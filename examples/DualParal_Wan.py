import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "../Checkpoint/"
os.environ["HF_HOME"] = "../Checkpoint/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "../Checkpoint/"
os.environ["TRANSFORMERS_CACHE"] = "../Checkpoint/"
# sys.path.append('../../')
import time
import copy
import argparse
import numpy as np
import multiprocessing
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from src.pipelines import DualParalWanPipeline, QueueLatents, Queue
from src.distribution_utils import RuntimeConfig, DistController, export_to_images, memory_check

def _parse_args():
    # For basic args
    parser = argparse.ArgumentParser(description="DualParal with Wan")
    parser.add_argument("--dtype", type=str, default="bf16",
        help="Model dtype (float64, float32, float16, fp32, fp16, half, bf16)")
    parser.add_argument("--seed", type=int, default=12345,
        help="The seed to use for generating the image or video.")
    parser.add_argument("--save_file", type=str, default="../results/",
        help="The file to save the generated image or video to.")
    parser.add_argument("--verbose", action="store_true", default=False, 
        help="Enable verbose mode")
    parser.add_argument("--export_image", action="store_true", default=False,
        help="Enable exporting video frames.")
        
    # For Wan-Video model
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Model Id for Wan-2.1 Video.")
    parser.add_argument("--height", type=int, default=480,
        help="Height of generating videos")
    parser.add_argument("--width", type=int, default=832,
        help="Width of generating videos")
    parser.add_argument("--sample_steps", type=int, default=50, 
        help="The sampling steps.")
    parser.add_argument("--sample_shift", type=float, default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument("--prompt", type=str, default="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
        help="The prompt to generate the image or video from.")
    
    # For DualParal
    parser.add_argument("--num_per_block", type=int, default=10,
        help="How many latents per block in DualParal.")
    parser.add_argument("--latents_num", type=int, default=30,
        help="How many latents to sample from a image or video. The total frames is equal to (latents_num-1)*4+1.")
    parser.add_argument("--num_cat", type=int, default=5,
        help="How many latents to concat in previous and backward blocks separately.")
        
    args = parser.parse_args()
    return args

def prepare_model(args, parallel_config, runtime_config):
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    model = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=runtime_config.dtype)
    model.scheduler = scheduler

    Pipe = DualParalWanPipeline(model, parallel_config, runtime_config)
    Pipe.to_device(device=parallel_config.device, dtype=runtime_config.dtype, non_blocking=True)

    num_channels_latents = Pipe.model.transformer.config.in_channels
    args.height = int(args.height) // Pipe.model.vae_scale_factor_spatial
    args.width = int(args.width) // Pipe.model.vae_scale_factor_spatial
    return Pipe, num_channels_latents

def update(ID, QueueWan, cnt_, Pipe, z):    
    test = QueueWan.get(ID)
    scheduler = Pipe.get_scheduler_dict(QueueWan.begin + ID)
    z = scheduler.step(z, Pipe.timesteps[test.denoise_time-1], test.z, return_dict=False)[0]
    test.z.copy_(z, non_blocking=True)

    if ID + QueueWan.begin in cnt_:
        cnt_ [ID + QueueWan.begin] += 1
    else:
        cnt_ [ID + QueueWan.begin] = 1

def main(args, rank, world_size):
    #---------------Model Preparation--------------------
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True 
    seed_everything(args.seed)
    parallel_config = DistController(rank, world_size)
    runtime_config = RuntimeConfig(args.seed, args.dtype)

    prompt = args.prompt
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    Pipe, out_channels = prepare_model(args, parallel_config, runtime_config)
    Pipe.get_model_args(
        prompt=prompt,
        num_inference_steps=args.sample_steps,
        negative_prompt=negative_prompt,
        guidance_scale=args.sample_guide_scale,
    )
    timesteps = Pipe.timesteps
    latent_size = (args.num_per_block, args.height, args.width)
    #---------------Warm Up--------------------
    print(f"[{Pipe.parallel_config.device}]--------------Warm Up-----------------")
    latent_size_warmup = (1, out_channels, args.num_per_block, args.height, args.width)
    Block_latents_size = None
    for iteration in range(1):
        warmup_block = QueueLatents(1, out_channels, latent_size).to(Pipe.parallel_config.device, Pipe.runtime_config.dtype, non_blocking=True)
        if Pipe.parallel_config.rank==0:
            output = Pipe.onestep(warmup_block, latent_size=warmup_block.z.size())
            Block_latents_size = output.size()
            Pipe.parallel_config.pipeline_send(tensor=output, dtype=Pipe.runtime_config.dtype, verbose=False)
            output = Pipe.parallel_config.pipeline_recv(dtype=Pipe.runtime_config.dtype, dimension=warmup_block.z.dim(), verbose=False)
        else:
            output = Pipe.parallel_config.pipeline_recv(dtype=Pipe.runtime_config.dtype, dimension=3, verbose=False)
            warmup_block.z = output
            Block_latents_size = output.size()
            output = Pipe.onestep(warmup_block, latent_size=latent_size_warmup)
            Pipe.parallel_config.pipeline_send(tensor=output, dtype=Pipe.runtime_config.dtype, verbose=False)
        del output
    print(f"[{Pipe.parallel_config.device}]-------------Warm Up End----------------------")
    
    #---------------DualParal--------------------
    cnt, video, cnt_ = 0, None, {}  # cnt for counting total latents in queue, video for concating video latents
    Pipe.parallel_config.init_buffer(timesteps)

    QueueWan = Queue(num_per_block=args.num_per_block, num_cat=args.num_cat, lenth_of_Queue=len(timesteps))
    Block_ref = copy.deepcopy(QueueWan)
    latent_tmp = (args.num_per_block + args.num_cat, args.height, args.width)
    test = QueueLatents(1, out_channels, latent_tmp)
    Block_ref.add_block(test)
    if Pipe.parallel_config.rank>0:
        num_frames = Block_latents_size[1]//args.num_per_block*Block_ref.get_size(0)
        tensor_shape = torch.tensor([Block_latents_size[0], num_frames, Block_latents_size[2]], dtype=torch.int64).contiguous()
        Pipe.parallel_config.modify_recv_queue(iteration=-1, idx=0, dtype=Pipe.runtime_config.dtype, tensor_shape=tensor_shape, verbose=False)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    start_time = time.time()
    template_z = torch.randn(1, out_channels, args.num_per_block + args.num_cat, args.height, args.width, pin_memory=True)
    for iteration in tqdm(range(len(timesteps)+args.latents_num//args.num_per_block)):
        Pipe.cache = []
        now_iteration = (iteration-1+2)%2
        prev_get = Pipe.parallel_config.init_get_start(now_iteration)
        if iteration == 1: prev_get = 0 # Corner Case
        
        # Change Time in Block Ref 
        if Pipe.parallel_config.rank > 0:
            for idx in range(Block_ref.end-Block_ref.begin, -1, -1):
                test = Block_ref.get(idx)
                test.denoise_time += 1
            if Block_ref.check_first(): Block_ref.del_prev_first()

        # Add block 
        add_block=False
        if cnt < args.latents_num:
            latent_tmp = latent_size
            if iteration==0:
                latent_tmp = (args.num_per_block + args.num_cat, args.height, args.width)
            select_indice = torch.randperm(template_z.size(2) - args.num_cat)
            template_z = torch.cat((template_z[:, :, -args.num_cat:], template_z[:, :, select_indice]), dim=2)
            test = QueueLatents(1, out_channels, latent_tmp, template_z).to(Pipe.parallel_config.device, Pipe.runtime_config.dtype, non_blocking=True)
            QueueWan.add_block(test)
            cnt += args.num_per_block
            add_block=True
            Pipe.add_scheduler(QueueWan.end)
            if cnt < args.latents_num:
                test = QueueLatents(1, out_channels, latent_size)
                Block_ref.add_block(test)
        if args.verbose: print(f"[{parallel_config.device}]--------DualParal in iteration-{iteration} is Begin with Queue from {QueueWan.begin} to {QueueWan.end}-----------")
        # Prepare for receving latents between two GPUs
        if Pipe.parallel_config.rank > 0:
            for idx in range(Block_ref.end-Block_ref.begin, -1, -1):
                num = Block_latents_size[1]//args.num_per_block * Block_ref.get_size(idx)
                tensor_shape = torch.tensor([Block_latents_size[0], num, Block_latents_size[2]], dtype=torch.int64).contiguous()
                Pipe.parallel_config.modify_recv_queue(iteration, idx, dtype=Pipe.runtime_config.dtype, tensor_shape=tensor_shape, verbose=False)

        # Del First Block in QueueWan
        del_begin = 0
        if QueueWan.check_first():
            QueueWan.del_prev_first()
            ID = QueueWan.begin - 2
            if ID >= 0: Pipe.del_scheduler(ID)
            del_begin = 1
        
        for idx in range(QueueWan.end-QueueWan.begin, -1, -1):
            now_end = ( idx==0 and iteration==(len(timesteps)+args.latents_num//args.num_per_block-1) )
            get_next = False
            # Receving 
            if Pipe.parallel_config.rank != 0:
                x = Pipe.parallel_config.recv_next(iteration-1, idx, queue_lenth=Block_ref.end-Block_ref.begin+1, 
                                                    force=True, end=(Block_ref.end-Block_ref.begin+1==0), verbose=False) 
                input_block_tmp = QueueWan.get(idx)
                input_block = QueueLatents(1, out_channels, None, None)
                input_block.denoise_time = input_block_tmp.denoise_time
                input_block.z = x
            else:
                latent_size_tmp = latent_size_warmup
                if QueueWan.begin+idx == 0:
                    latent_size_tmp = (1, out_channels, args.num_per_block + args.num_cat, args.height, args.width)
                tensor_shape = torch.tensor(latent_size_tmp, dtype=torch.int64).contiguous()
                Pipe.parallel_config.modify_recv_queue(iteration, idx, dtype=Pipe.runtime_config.dtype, tensor_shape=tensor_shape, verbose=False)
                force = False
                if prev_get >= 0:
                    # Make sure the gap between communication large than world_size to make sure output 
                    # is already come out from the last device
                    if (abs(prev_get+1 + QueueWan.end-idx+int(add_block)) <= Pipe.parallel_config.world_size or iteration==1)\
                        and idx == QueueWan.end-QueueWan.begin and add_block==True: 
                        get_next = True
                    else:
                        force = (now_iteration==(iteration+1)%2 and prev_get==idx)
                        z = Pipe.parallel_config.recv_next(now_iteration, prev_get, queue_lenth=QueueWan.end-QueueWan.begin+1, 
                                                            force=force, end=now_end, verbose=False)  
                        if z is not None:
                            get_next = True
                            ID = prev_get - del_begin
                            if now_iteration%2 == iteration%2:
                                ID = prev_get
                            update(ID, QueueWan, cnt_, Pipe, z)
                            prev_get -= 1
                else: get_next = True

                if prev_get < 0:
                    now_iteration = now_iteration^1
                    prev_get = Pipe.parallel_config.init_get_start(now_iteration) 
                input_block = QueueLatents(1, out_channels, None, None)
                L, R, latents_z, denoise_time = QueueWan.prepare_for_forward(idx)
                input_block.z = latents_z
                input_block.denoise_time = denoise_time
            
            size_frames = QueueWan.get_size(idx)
            latent_size_tmp = (latent_size_warmup[0], latent_size_warmup[1], size_frames, latent_size_warmup[3], latent_size_warmup[4])
            # Pipe.cache = []
            x = Pipe.onestep(input_block, latent_size=latent_size_tmp, latent_num=Block_latents_size[1], select=args.num_cat, select_all=args.num_per_block, verbose=False)
            
            if Pipe.parallel_config.rank != Pipe.parallel_config.world_size-1:
                # check recieving next 
                if parallel_config.rank==0 and not get_next:
                    z = Pipe.parallel_config.recv_next(now_iteration, prev_get, queue_lenth=QueueWan.end-QueueWan.begin+1, 
                                                        force=True, end=now_end, verbose=False)   
                    ID = prev_get - del_begin
                    if now_iteration%2 == iteration%2:
                        ID = prev_get
                    update(ID, QueueWan, cnt_, Pipe, z)
                    prev_get -= 1
                    if prev_get < 0:
                        now_iteration = now_iteration^1
                        prev_get = Pipe.parallel_config.init_get_start(now_iteration) 
                if args.verbose: print(f"[{Pipe.parallel_config.device}] ready to send X with size {x.size()}")
                Pipe.parallel_config.pipeline_isend(tensor=x, dtype=Pipe.runtime_config.dtype, verbose=False)
            else:
                size_latent = QueueWan.get_size(idx, itself=True)
                x = x[:, :, -size_latent:].clone().contiguous()
                if args.verbose: print(f"[{Pipe.parallel_config.device}] ready to send X with size {x.size()}, with sum {x.size()}")
                Pipe.parallel_config.pipeline_isend(tensor=x, dtype=Pipe.runtime_config.dtype, verbose=False)
            #Update Denoise_Time in Queue
            input_block = QueueWan.get(idx)
            input_block.denoise_time += 1

        # Extract First Block
        if del_begin==1 and Pipe.parallel_config.rank==0:
            first_block = QueueWan.get(-1)
            z = Pipe.parallel_config.recv_next(iteration-1, idx=0, verbose=False, queue_lenth=QueueWan.end-QueueWan.begin+1,
                                                force=True, end=(iteration==(len(timesteps)+args.latents_num//args.num_per_block-1)))
            if z is not None:
                prev_get = -1
                cnt_ [prev_get + QueueWan.begin] += 1
                scheduler = Pipe.get_scheduler_dict(QueueWan.begin - 1)
                z = scheduler.step(z, Pipe.timesteps[first_block.denoise_time-1], first_block.z, return_dict=False)[0]
                first_block.z = z
            video_ = first_block.z
            video = video_ if video is None else torch.cat((video, video_), dim=2)
            if args.verbose: print(f"[{Pipe.parallel_config.device}] Now {iteration} video size: ", video.size())
    
    torch.cuda.synchronize()
    print(f"[{Pipe.parallel_config.device}] Whole inference time {time.time()-start_time:.6f}s")
    if Pipe.parallel_config.rank==0:
        print("Video latents size: ", video.size())
        video = Pipe.get_video(video)[0]
        print(f"Final Video shape: {video.shape}")
        export_to_video(video, args.save_file+"output.mp4", fps=16)
        if args.export_image:
            export_to_images(video, args.save_file + "frames/")    

if __name__ == "__main__":
    args = _parse_args()
    multiprocessing.set_start_method('spawn')
    num_processes = torch.cuda.device_count()
    processes = []

    for rank in range(num_processes):
        p = multiprocessing.Process(target=main, args=(args, rank, num_processes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()