import torch
from src.distribution_utils import DistController, RuntimeConfig

class QueueLatents(object):
    def __init__(self, 
        batch_size, 
        out_channels,
        latent_size,
        z = None
    ):
        self.z = torch.randn(batch_size, out_channels, *latent_size, pin_memory=True) if latent_size is not None else None
        self.z = z[:, :, -latent_size[0]:].clone() if z is not None else self.z
        self.denoise_time = 0

    def to(self, *args, **kwargs):
        self.z = self.z.to(*args, **kwargs).contiguous() if self.z is not None else None
        return self

class Queue(object):
    def __init__(self, num_per_block=10, num_cat=5, lenth_of_Queue=50):
        self.num_per_block = num_per_block
        self.num_cat = num_cat
        self.lenth_of_Queue = lenth_of_Queue
        self.begin, self.end = 0, -1
        self.queue = []
    
    def get(self, idx):
        return self.queue[idx+self.begin]

    def get_size(self, idx, itself=False):
        idx = self.begin+idx
        if itself:
            return self.queue[idx].z.size(2)
        
        prev = self.queue[idx-1] if idx-1>=0 else None
        L, R = 0, self.queue[idx].z.size(2)
        if prev is not None:
            R += self.num_cat
        return R

    def prepare_for_forward(self, idx):
        idx = self.begin + idx
        prev = self.queue[idx - 1] if idx-1 >=0 else None
        now = self.queue[idx].z.clone()
        L, R = 0, now.size(2)
        

        if prev is not None and self.num_cat > 0:
            now = torch.cat((prev.z[:,:,-self.num_cat:], now), dim=2)
            now = now[:, :, -self.num_cat - now.size(2):]
            L, R = L + self.num_cat, R + self.num_cat
        
        return L, R, now, self.queue[idx].denoise_time

    def update(self, idx, block):
        self.queue[idx + self.begin].z.copy_(block.z, non_blocking=True)

    def add_block(self, block):
        self.queue.append(block)
        self.end += 1
    
    def check_first(self, is_first=False):
        if self.begin>self.end:
            return False
        first_block = self.queue[self.begin]
        return first_block.denoise_time==(self.lenth_of_Queue - int(is_first==True))
    
    def del_prev_first(self):
        '''
        Followed by check_first()
        Self.begin add 1
        '''
        if self.begin-1>=0 and self.queue[self.begin-1] is not None:
            self.queue[self.begin-1] = None
        self.begin += 1

    def print_queue(self, device):
        print(f"[{device}] LOOK Queue: ", end=" ")
        for i in range(self.begin, self.end+1):
            print(self.queue[i].denoise_time, end=", ")
        print()

class DualParalPipelineBaseWrapper(object):
    def __init__(
        self,
        parallel_config: DistController,
        runtime_config: RuntimeConfig,
    ):
        self.runtime_config = runtime_config
        self.parallel_config = parallel_config
    
    def _get_blocks_range(self, transformer):
        world_size = self.parallel_config.world_size
        local_rank = self.parallel_config.rank

        lenth_of_blocks = len(transformer)
        lenth_of_blocks_per_device = lenth_of_blocks//world_size
        mod_of_blocks_per_device = lenth_of_blocks%world_size
        range_of_block = {}
        start_q, end_q = 0, lenth_of_blocks_per_device
        for i in range(world_size):
            end_q = end_q + (mod_of_blocks_per_device>0)
            range_of_block[i] = (start_q, end_q)
            start_q = end_q
            end_q = end_q +lenth_of_blocks_per_device
            mod_of_blocks_per_device -= 1

        start_range, final_range = range_of_block[local_rank][0], range_of_block[local_rank][1]
        if self.parallel_config.rank==self.parallel_config.world_size-1:
            final_range = lenth_of_blocks
        range_of_blocks = range(start_range, final_range)
        return range_of_blocks

    def _split_transformer_backbone(self):
        range_of_blocks = self._get_blocks_range(self.transformer)
        self.transformer_ = self.transformer[range_of_blocks.start:range_of_blocks.stop]
        del self.transformer

        if self.transformer2 is not None:
            range_of_blocks = self._get_blocks_range(self.transformer2)
            self.transformer2_ = self.transformer2[range_of_blocks.start:range_of_blocks.stop]
            del self.transformer2
        else:
            self.transformer2_ = None

        self.transformer = self.transformer_
        self.transformer2 = self.transformer2_
        del self.transformer_
        del self.transformer2_
        
    def forward(self):
        pass
    
    def to(self, *args, **kwargs):
        pass