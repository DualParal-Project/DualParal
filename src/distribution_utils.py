import os
import time
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

class RuntimeConfig():
    def __init__(self, 
        seed: int = 42,
        dtype: torch.dtype = torch.float16
    ):
        self.seed = seed
        self.dtype = self.to_torch_dtype(dtype)
    
    def to_torch_dtype(self, dtype):
        if isinstance(dtype, torch.dtype):
            return dtype
        elif isinstance(dtype, str):
            dtype_mapping = {
                "float64": torch.float64,
                "float32": torch.float32,
                "float16": torch.float16,
                "fp32": torch.float32,
                "fp16": torch.float16,
                "half": torch.float16,
                "bf16": torch.bfloat16,
            }
            if dtype not in dtype_mapping:
                raise ValueError
            dtype = dtype_mapping[dtype]
            return dtype
        else:
            raise ValueError

class DistController(object):
    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{self.rank}")

        self.prev = self.rank-1 if self.rank-1>=0 else self.world_size-1
        self.next = self.rank+1 if self.rank+1<self.world_size else 0
        self.gpu_group_send = None
        self.gpu_group_receive = None
        self.buffer_recv = {0: [], 1: []}
        self.recv_queue = {0: [], 1: []}

        self.init_dist()
        if self.world_size>1: self.init_group()

    def init_get_start(self, iteration):
        for i in range(len(self.buffer_recv[iteration])-1, -1, -1):
            if self.recv_queue[iteration][i] is not None:
                return i
        return -1

    def init_buffer(self, timesteps):
        for i in range(len(timesteps)+1):
            for key in self.buffer_recv:
                self.buffer_recv[key].append(None)
                self.recv_queue[key].append(None)
        # print(f"[{self.device}] Init Buffer----")

    def modify_recv_queue(self, iteration, idx, tensor_shape=None, dtype=torch.float16, verbose=False):
        iteration = (iteration+2)%2
        
        assert self.buffer_recv[iteration][idx] is None, f"[{self.device}] Buffer_Queue at iteration {iteration} and index {idx} is not None!"
        assert self.recv_queue[iteration][idx] is None, f"[{self.device}] Recv_Queue at iteration {iteration} and index {idx} is not None!"
        
        tensor = torch.empty(tensor_shape.tolist(), device="cpu", dtype=dtype, pin_memory=True)  
        self.buffer_recv[iteration][idx] = tensor
        if verbose: print(f"[{self.device}] idx {idx} on iteration {iteration} modify with size {tensor_shape}")
   
    def recv_next(self, iteration, idx, queue_lenth=0, force=False, end=False, TIMEOUT=50, verbose=False):
        '''
        queue_lenth = Now queue lenth
        '''
        iteration = (iteration+2)%2
        req = self.recv_queue[iteration][idx]
        if req is None:
            if self.buffer_recv[iteration][idx] is None:
                # no tensor in buffer_recv means that 'idx' is already recieved
                return None
            else:
                self.buffer_recv[iteration][idx] = self.buffer_recv[iteration][idx].to(self.device, non_blocking=True).contiguous()
                req = self._pipeline_irecv(self.buffer_recv[iteration][idx])

        if verbose: start_time = time.time()
        if not req.is_completed() and not force:
            return None
        req.wait()
        ans = self.buffer_recv[iteration][idx]

        self.recv_queue[iteration][idx] = None
        self.buffer_recv[iteration][idx] = None
        if verbose: print(f"[{self.device}] Request status of idx{idx} after wait:", req.is_completed(), " with size: ", ans.size(), f" with time: {time.time()-start_time:.6f}s")
        if end: return ans

        if verbose: start = time.time()
        next_id = idx-1 #previous one in backward order
        if next_id < 0:
            next_id = 0 if queue_lenth==0 else (next_id + queue_lenth)%queue_lenth
            self.buffer_recv[iteration^1][next_id] = self.buffer_recv[iteration^1][next_id].to(self.device, non_blocking=True)
            self.recv_queue[iteration^1][next_id] = self._pipeline_irecv(self.buffer_recv[iteration^1][next_id])
            tmp = self.recv_queue[iteration^1][next_id]
        else:
            self.buffer_recv[iteration][next_id] = self.buffer_recv[iteration][next_id].to(self.device, non_blocking=True)
            self.recv_queue[iteration][next_id] = self._pipeline_irecv(self.buffer_recv[iteration][next_id],)
            tmp = self.recv_queue[iteration][next_id]
        if verbose: print(f"[{self.device}] Stream status {tmp.is_completed()} recieving {time.time()-start:.6f}s")
        return ans
 
    def pipeline_isend(self, tensor, dtype, verbose=False) -> None:
        tensor_shape = tensor.size()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()  
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        req = self._pipeline_isend(tensor)

    
    def _pipeline_irecv(self, tensor: torch.tensor):
        return torch.distributed.irecv(
            tensor,
            src=self.prev,
            group=self.gpu_group_receive,
        )
    
    def _pipeline_isend(self, tensor: torch.tensor):
        return torch.distributed.isend(
            tensor,
            dst=self.next,
            group=self.gpu_group_send
        )

    def init_dist(self):
        torch.cuda.set_device(self.device)
        print(f"Rank {self.rank} (world size {self.world_size}, with prev {self.prev} and next {self.next}) is running.")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    def init_group(self):
        group = list(range(self.world_size))
        if len(group) > 2 or len(group) == 1:
            device_group = torch.distributed.new_group(group, backend="nccl")
            self.gpu_group_receive = device_group
            self.gpu_group_send = device_group
        elif len(group) == 2:
            # when pipeline parallelism is 2, we need to create two groups to avoid
            #   communication stall.
            # *_group_0_1 represents the group for communication from device 0 to 
            #   device 1.
            # *_group_1_0 represents the group for communication from device 1 to
            #   device 0.
            device_group_0_1 = torch.distributed.new_group(group, backend="nccl")
            device_group_1_0 = torch.distributed.new_group(group, backend="nccl")
            groups = [device_group_0_1, device_group_1_0]
            self.gpu_group_send = groups[self.rank]
            self.gpu_group_receive = groups[(self.rank+1)%2]

    def pipeline_send(self, tensor: torch.Tensor, dtype, verbose=False) -> None:
        if verbose: start_time = time.time()

        tensor_shape = torch.tensor(tensor.size(), device=tensor.device, dtype=torch.int64).contiguous()
        if verbose: print(f"[{self.device}] Send size {tensor_shape} with dimension {tensor.dim()}")
        self._pipeline_isend(tensor_shape).wait()
        if verbose: print(f"[{self.device}] Success Send size {tensor_shape}")
        tensor = tensor.contiguous().to(dtype)
        self._pipeline_isend(tensor).wait()

        del tensor_shape
        del tensor
        if verbose: print(f"[{self.device}] Sending Tensor with shape {tensor.size()} ({tensor.device}, {tensor.dtype}, {tensor.sum()}) in {time.time()-start_time:.6f}s")

    def pipeline_recv(self, dtype, dimension=3, verbose=False) -> torch.Tensor:
        if verbose: start_time = time.time()

        tensor_shape = torch.empty(dimension, device=self.device, dtype=torch.int64)  
        if verbose: print(f"[{self.device}] Ready size {tensor_shape} with dimension {dimension}")
        self._pipeline_irecv(tensor_shape).wait()
        if verbose: print(f"[{self.device}] Got size {tensor_shape}")
        tensor = torch.empty(tensor_shape.tolist(), device=self.device, dtype=dtype)  # 假设数据类型为float32，调整为适合的类型
        self._pipeline_irecv(tensor).wait()

        del tensor_shape
        if verbose: print(f"[{self.device}] Receiving Tensor with shape {tensor.shape}(dtype {dtype}, sum {tensor.sum()}) in {time.time()-start_time:.6f}s")
        return tensor

def export_to_images(video_frames, output_dir: str = None,):
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    for i, frame in enumerate(video_frames):
        image_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame).save(image_path)
        output_paths.append(image_path)

    return output_paths

def memory_check(device, info = ""):
    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    max_allocated_memory = torch.cuda.max_memory_allocated()
    max_cached_memory = torch.cuda.max_memory_reserved()

    print(f"[{device}] {info} Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"[{device}] {info} Cached Memory: {cached_memory / (1024 ** 2):.2f} MB")
    print(f"[{device}] {info} Max Allocated Memory: {max_allocated_memory / (1024 ** 2):.2f} MB")
    print(f"[{device}] {info} Max Cached Memory: {max_cached_memory / (1024 ** 2):.2f} MB")

    