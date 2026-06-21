import os
import time
import json
import torch
import torch.distributed as dist
import torch.nn as nn

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

def profile_compute(device, dtype=torch.float32, warmup=3, active=10):
    # Dummy tensor sizes for compute
    N = 4096
    A = torch.randn(N, N, device=device, dtype=dtype)
    B = torch.randn(N, N, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize(device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(active):
        torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize(device)
    
    time_ms = start_event.elapsed_time(end_event)
    ops_per_matmul = 2.0 * N * N * N
    total_ops = ops_per_matmul * active
    
    tflops = (total_ops / (time_ms / 1000.0)) / 1e12
    return tflops

def profile_vram(device, dtype=torch.float32, input_dim=1024, hidden_dim=4096):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, input_dim)
    ).to(device).to(dtype)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    max_safe_batch = 0
    current_batch = 32
    
    while True:
        try:
            x = torch.randn(current_batch, input_dim, device=device, dtype=dtype)
            target = torch.randn(current_batch, input_dim, device=device, dtype=dtype)
            
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            
            max_safe_batch = current_batch
            current_batch += 32
            
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            else:
                raise e
                
    return max_safe_batch

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group("nccl", device_id=device)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    
    if rank == 0:
        print("\n==================================================")
        print("--- STARTING AUTOMATED COMPUTE & VRAM PROFILER ---")
        print("==================================================")
        
    for dtype_str, target_dtype in dtypes.items():
        if dtype_str == "bf16" and not torch.cuda.is_bf16_supported():
            continue
            
        dist.barrier()
        if rank == 0: print(f"\n--- Profiling {dtype_str.upper()} ---")
            
        tflops = profile_compute(device, dtype=target_dtype)
        max_local_batch = profile_vram(device, dtype=target_dtype)
        
        # Bulletproof tensor serialization for names to completely avoid NCCL all_gather_object deadlocks
        name_str = torch.cuda.get_device_name(device)
        name_bytes = bytearray(name_str.encode("utf-8"))
        name_bytes.extend([0] * (256 - len(name_bytes)))
        name_tensor = torch.tensor(name_bytes, device=device, dtype=torch.uint8)
        
        gathered_names = [torch.zeros(256, device=device, dtype=torch.uint8) for _ in range(world_size)]
        dist.all_gather(gathered_names, name_tensor)
        
        # Gather TFLOPS and VRAM exactly as before
        tflops_tensor = torch.tensor([tflops, max_local_batch], device=device, dtype=torch.float32)
        gathered_stats = [torch.zeros(2, device=device, dtype=torch.float32) for _ in range(world_size)]
        dist.all_gather(gathered_stats, tflops_tensor)
        
        if rank == 0:
            scores = [t[0].item() for t in gathered_stats]
            vrams = [t[1].item() for t in gathered_stats]
            
            names = []
            for t in gathered_names:
                byte_list = t.cpu().tolist()
                names.append(bytes(byte_list).decode("utf-8").strip("\x00"))
            
            total_tflops = sum(scores)
            weights = [s / total_tflops for s in scores]
            
            max_globals = [int(vrams[i] / weights[i]) for i in range(world_size)]
            safe_global_batch = min(max_globals)
            safe_global_batch = max(32, (safe_global_batch // 32) * 32)
            
            print(f"Total Cluster Compute: {total_tflops:.2f} TFLOPS")
            print(f"Optimal Auto-Detected Global Batch Size: {safe_global_batch}")
            
            print("\n--- GPU Breakdown ---")
            for i in range(world_size):
                print(f"Rank {i} [{names[i]}]: {scores[i]:.2f} TFLOPS | Max Local Batch: {vrams[i]:.0f} | Compute Weight: {weights[i]*100:.1f}%")
            print("---------------------\n")
            
            json_path = os.path.join(os.getcwd(), f"cluster_weights_{dtype_str}.json")
            with open(json_path, "w") as f:
                json.dump({
                    "dtype": dtype_str, 
                    "weights": weights, 
                    "world_size": world_size,
                    "optimal_global_batch_size": safe_global_batch,
                    "optimal_seq_len": 512
                }, f, indent=4)
                
    if rank == 0: print("\nAll precisions profiled! Ratios and Auto-Batch Sizes saved.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
