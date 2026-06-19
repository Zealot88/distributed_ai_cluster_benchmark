# Asymmetric DDP: Heterogeneous Cluster Benchmark 🚀

A PyTorch framework that unlocks **100% hardware utilization** across highly heterogeneous GPU clusters (mixing Pascal, Ampere, and Ada architectures on the same network).

## The "Straggler Effect" Problem
Standard PyTorch Distributed Data Parallel (DDP) assumes all GPUs in a cluster are completely identical. If you mix an RTX 4070 Super with an old Tesla P4 over a 100GbE network, DDP assigns them the exact same batch size and forces them to synchronize. The 4070 Super finishes its batch in 10 milliseconds, and then sits completely idle waiting for the older P4 to finish its math. The fastest GPUs in your cluster drop to idle wattage, crippling your throughput.

## The Solution: 3.24x Performance Multiplier
This repository solves the Straggler Effect using **Asymmetric Batch Sizing** combined with a mathematical **Gradient Loss Scaling Hook**, perfectly balancing the computational load across wildly varying architectures without requiring custom C++ NCCL hooks.

### Real-World Benchmark Results
Testing across an 8-GPU cluster (2x RTX A4500, 1x RTX 4070 Super, 1x RTX 3080, 4x Tesla P4) over a 100GbE Mellanox RoCEv2 (RDMA) fabric:

| Precision | NAIVE DDP | ASYMMETRIC DDP | Improvement |
|-----------|-----------|----------------|-------------|
| **FP32**  | 59.15 Samples/s | 93.69 Samples/s | **+1.58x** |
| **FP16**  | 79.51 Samples/s | 257.77 Samples/s | **+3.24x** |
| **BF16**  | 58.81 Samples/s | 257.10 Samples/s | **+4.37x** |

*(Note the massive multiplier in Mixed Precision! The framework correctly auto-detects that Pascal P4s lack Tensor Cores and completely offloads the FP16 matrix math to the Ada/Ampere hardware.)*

---

## 🛠️ Repository Structure

```text
├── asymmetric_ddp/
│   ├── __init__.py
│   ├── sampler.py        # Asymmetric Data Sampler
│   └── scaler.py         # Mathematical Loss Scaling Hook
├── scripts/
│   ├── profiler.py       # Auto-detects TFLOPS and Max VRAM batch size
│   └── benchmark.py      # Automated testing suite
├── README.md
└── requirements.txt
```

---

## ⚙️ How It Works

### 1. `profiler.py` (The Calibration Step)
Runs a raw TFLOPS matrix stress test on every GPU to generate a cluster weighting. It then iteratively loads a real PyTorch Transformer model into VRAM and exponentially scales the batch size until the GPU throws a `CUDA Out of Memory` error. It combines the compute ratio and VRAM limits to output a mathematically perfect, cluster-wide `GLOBAL_BATCH_SIZE`.

### 2. `AsymmetricDistributedSampler`
Intercepts the PyTorch DataLoader and partitions the global batch unevenly. If the 4070 Super has 25% of the cluster's compute power, it is fed exactly 25% of the batch.

### 3. `scale_loss_for_asymmetric_ddp()` (The Magic Math)
If the 4070 Super trains on 64 images and the P4 trains on 2 images, standard DDP will blindly average their gradients 50/50, corrupting the model. We intercept the loss *before* `loss.backward()` and mathematically scale it by `(local_batch_size * world_size) / global_batch_size`. This mathematically tricks PyTorch's native C++ NCCL engine into reducing the true global mean!

---

## 🚀 Quick Start

### 1. Auto-Profile Your Cluster
```bash
# Run on all nodes via torch.distributed.run
python -m torch.distributed.run --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.0.100.2" --master_port=29500 scripts/profiler.py
```

### 2. Run the Benchmark
```bash
# Run on all nodes to test Naive vs Asymmetric
python -m torch.distributed.run --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.0.100.2" --master_port=29500 scripts/benchmark.py
```

### RDMA (RoCEv2) Note:
If you are running InfiniBand/RoCEv2, ensure you export `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` so NCCL routes traffic over the correct hardware Queue Pairs!
