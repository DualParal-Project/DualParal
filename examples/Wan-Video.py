import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "../Checkpoint/"
os.environ["HF_HOME"] = "../Checkpoint/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "../Checkpoint/"
os.environ["TRANSFORMERS_CACHE"] = "../Checkpoint/"
import time
import torch
from transformers import T5TokenizerFast, UMT5EncoderModel
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

def batch_generate_video(model_id, prompt):
    ### infer config
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_frames = 33
    num_inference_steps = 50
    sample_fps = 16
    # height, width = 720, 1280
    height, width = 480, 832
    if model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers":
        guidance_scale = 4.0
        guidance_scale_2 = 3.0
        sample_shift = 12.0
        boundary_ratio = 0.875
    elif model_id == "Wan-AI/Wan2.1-T2V-14B-Diffusers":
        guidance_scale =5.0
        guidance_scale_2 = None
        sample_shift = 3.0
        boundary_ratio = None
    elif model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers":
        guidance_scale =5.0
        guidance_scale_2 = None
        sample_shift = 3.0
        boundary_ratio = None
    else:
        raise NotImplementedError

    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", flow_shift=sample_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, scheduler=scheduler, boundary_ratio=boundary_ratio, torch_dtype=dtype)
    # print(pipe.transformer2)
    ### Inference
    pipe = pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    export_to_video(output, "output.mp4", fps=sample_fps)

# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
# model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
batch_generate_video(model_id, prompt)