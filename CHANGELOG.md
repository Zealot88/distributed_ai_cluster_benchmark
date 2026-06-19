# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-06-19
### Added
- Mathematical loss scaling for exact global gradient equivalence.
- Custom `AsymmetricDistributedSampler` to partition datasets by TFLOPS weights.
- Automated compute profiler to measure raw FP32/FP16/BF16 capabilities.
- Automated VRAM profiler to dynamically find max safe batch size for heterogeneous clusters.
- `benchmark.py` showing a 3.24x performance jump on a heterogeneous (Ada/Ampere/Pascal) cluster over 100G RoCEv2.
- Fully automated Precision-Switching loop (`autocast`) to accurately benchmark Tensor Cores.
