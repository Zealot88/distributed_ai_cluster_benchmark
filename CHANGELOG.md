# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-06-20
### Added
- Converted framework to a proper pip package with a global executable CLI `asym-ddp`.
- Injected `dist.all_gather_object` logic to explicitly broadcast specific GPU hardware strings (e.g., "NVIDIA GeForce RTX 4070 Super") to the terminal breakdown.

### Fixed
- Bypassed default Python `console_scripts` packaging engine (removed `pkg_resources` wrapper) that implicitly crashed Intel MKL OpenMP threading during nested process spawning.
- Added explicit `MKL_THREADING_LAYER="GNU"` injection directly into the CLI wrapper environment parameters.
- Patched silent multi-node deadlock caused by implicit NCCL `cuda:0` overriding by actively binding `torch.cuda.set_device` prior to process serialization commands.

## [0.1.0] - 2026-06-19
### Added
- Mathematical loss scaling for exact global gradient equivalence.
- Custom `AsymmetricDistributedSampler` to partition datasets by TFLOPS weights.
- Automated compute profiler to measure raw FP32/FP16/BF16 capabilities.
- Automated VRAM profiler to dynamically find max safe batch size for heterogeneous clusters.
- `benchmark.py` showing a 3.24x performance jump on a heterogeneous (Ada/Ampere/Pascal) cluster over 100G RoCEv2.
- Fully automated Precision-Switching loop (`autocast`) to accurately benchmark Tensor Cores.
