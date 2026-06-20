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

## ⚙️ Installation & Legacy Hardware Notice

### Legacy Hardware Support (Pascal / CUDA 6.1)
Standard pre-compiled PyTorch 2.x binaries **do not support** Pascal architectures (`sm_61`) like the Tesla P4 or GTX 1080. If you try to run PyTorch on these GPUs, it will crash. 
To run this framework on older hardware, you must build PyTorch from source:

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6;8.9"
export USE_ROCM=0
python setup.py install
```
### Requirements & Installation
To install the package and its global CLI globally:
```bash
git clone https://github.com/Zealot88/distributed_ai_cluster_benchmark.git
cd distributed_ai_cluster_benchmark
pip install -e .
```

---

## 🚀 Quick Start

### Single-Node Testing
If you are running everything on a single machine, you don't need to configure master IPs.

```bash
# 1. Profile the GPUs
asym-ddp profile --gpus <NUM_GPUS>

# 2. Run the Benchmark
asym-ddp benchmark --gpus <NUM_GPUS>
```

### Multi-Node Cluster
If you are syncing multiple machines over a network, use the CLI options to specify the node ranks and master IP.

**1. Auto-Profile Your Cluster**
*(Run on Node 1 Master)*
```bash
asym-ddp profile --gpus 4 --nodes 2 --node-rank 0 --master-addr "<MASTER_IP>"
```
*(Run on Node 2 Worker)*
```bash
asym-ddp profile --gpus 4 --nodes 2 --node-rank 1 --master-addr "<MASTER_IP>"
```

**2. Run the Benchmark**
*(Run on Node 1 Master)*
```bash
asym-ddp benchmark --gpus 4 --nodes 2 --node-rank 0 --master-addr "<MASTER_IP>"
```
*(Run on Node 2 Worker)*
```bash
asym-ddp benchmark --gpus 4 --nodes 2 --node-rank 1 --master-addr "<MASTER_IP>"
```

---

## 🌐 Advanced Network Setup (RDMA / RoCEv2)
If you are running InfiniBand or 100GbE RoCEv2 networks, standard PyTorch/NCCL may pick the wrong physical interface or the wrong IPv4 hardware address (GID Index). To unlock the raw hardware-offloaded speeds, prefix your `torchrun` commands with the following exports:

```bash
# Enable RDMA (0 = Enable, 1 = Disable to force standard TCP)
export NCCL_IB_DISABLE=0

# Bind to the correct physical network cards
export NCCL_SOCKET_IFNAME=<NETWORK_INTERFACE>
export GLOO_SOCKET_IFNAME=<NETWORK_INTERFACE>

# Specify the explicit Mellanox Hardware Device & Port
export NCCL_IB_HCA=<MELLANOX_DEVICE_NAME>:1

# Specify the GID Index containing your IPv4 address (usually 3 for RoCEv2)
export NCCL_IB_GID_INDEX=3
```
*(To find your exact GID Index, run `grep -H "<IP_HEX>" /sys/class/infiniband/*/ports/1/gids/*` on your machine.)*

---

## 🤝 Contributing
Contributions are extremely welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to submit pull requests or report bugs.

## 📄 License & Authorship
This project was authored by [Zealot88](https://github.com/Zealot88) and is released under the [MIT License](LICENSE).
