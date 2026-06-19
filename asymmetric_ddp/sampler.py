import torch
from torch.utils.data import Sampler
import torch.distributed as dist

class AsymmetricDistributedSampler(Sampler):
    """
    A custom DistributedSampler that yields different amounts of data per GPU
    based on the hardware's auto-profiled compute power weight.
    """
    def __init__(self, dataset, weights, rank=None, world_size=None, global_batch_size=None):
        if rank is None:
            rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()
            
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.global_batch_size = global_batch_size
        self.weight = weights[self.rank]
        
        self.local_batch_size = max(1, int(round(self.global_batch_size * self.weight)))
        total_batches = len(dataset) // self.global_batch_size
        self.num_samples = total_batches * self.local_batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        local_indices = []
        idx = 0
        while idx < len(self.dataset):
            chunk = indices[idx:idx + self.local_batch_size]
            if len(chunk) == self.local_batch_size:
                local_indices.extend(chunk)
            idx += self.global_batch_size
            
        local_indices = local_indices[:self.num_samples]
        return iter(local_indices)

    def __len__(self):
        return self.num_samples
