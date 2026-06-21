import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from asymmetric_ddp import AsymmetricDistributedSampler, scale_loss_for_asymmetric_ddp

warnings.filterwarnings("ignore")

class DummyTransformer(nn.Module):
    def __init__(self, d_model=1024, num_layers=12):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)) 
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 10)

    def forward(self, x):
        return self.classifier(self.layers(x).mean(dim=1))

def run_training_loop(mode, dtype_str, rank, world_size, local_rank, device):
    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    autocast_dtype = dtype_map[dtype_str]
    use_autocast = autocast_dtype is not None
    
    weights = [1.0 / world_size] * world_size
    GLOBAL_BATCH_SIZE = 256
    SEQ_LEN = 512
    
    json_path = os.path.join(os.getcwd(), f"cluster_weights_{dtype_str}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            weights = data["weights"]
            GLOBAL_BATCH_SIZE = data.get("optimal_global_batch_size", 256)
            SEQ_LEN = data.get("optimal_seq_len", 512)
            
    if mode == "naive":
        GLOBAL_BATCH_SIZE = min(GLOBAL_BATCH_SIZE, 32 * world_size)
    
    dataset = TensorDataset(torch.randn(10000, SEQ_LEN, 1024), torch.randint(0, 10, (10000,)))
    
    if mode == "naive":
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        local_batch_size = max(1, GLOBAL_BATCH_SIZE // world_size)
    else:
        sampler = AsymmetricDistributedSampler(dataset, weights, rank, world_size, GLOBAL_BATCH_SIZE)
        local_batch_size = sampler.local_batch_size
        
    dataloader = DataLoader(
        dataset, 
        batch_size=local_batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    model = DummyTransformer().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype_str == "fp16"))
    
    dist.barrier()
    if rank == 0:
        print(f"\n--- [ {dtype_str.upper()} | {mode.upper()} ] (Global Batch: {GLOBAL_BATCH_SIZE}, SeqLen: {SEQ_LEN}) ---")
        
    start_time = time.time()
    total_samples = 0
    
    for step, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_autocast):
            outputs = ddp_model(batch_x)
            loss = criterion(outputs, batch_y)
            if mode == "asymmetric":
                loss = scale_loss_for_asymmetric_ddp(loss, local_batch_size, GLOBAL_BATCH_SIZE, world_size)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_samples += GLOBAL_BATCH_SIZE
        if step >= 20: 
            break
            
    dist.barrier()
    elapsed = time.time() - start_time
    tps = total_samples / elapsed
    
    if rank == 0:
        print(f"Result: {tps:.2f} Samples/sec")

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group("nccl", device_id=device)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    if rank == 0:
        print("\n==================================================")
        print("--- FULL CLUSTER BENCHMARK SUITE ---")
        
    precisions = ["fp32", "fp16", "bf16"]
    modes = ["naive", "asymmetric"]
    
    for p in precisions:
        if p == "bf16" and not torch.cuda.is_bf16_supported():
            continue
        for m in modes:
            run_training_loop(m, p, rank, world_size, local_rank, device)
            
    if rank == 0:
        print("\n==================================================")
        print("Benchmark Complete!")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
