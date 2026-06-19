# Contributing to Asymmetric DDP

Thank you for your interest in contributing! This project aims to help the community maximize hardware utilization across heterogeneous AI clusters. Whether you are fixing a bug, adding support for new hardware architectures, or improving the documentation, your help is welcome!

## How to Contribute

1. **Fork the Repository:** Start by forking the repo to your own GitHub account.
2. **Clone Locally:** Clone your fork to your heterogeneous cluster.
3. **Create a Branch:** Create a feature branch (`git checkout -b feature/amazing-new-feature`).
4. **Test Your Changes:** 
   - Run `profiler.py` to ensure the mathematical batch scaling remains accurate.
   - Run `benchmark.py` in both `naive` and `asymmetric` modes to guarantee performance regressions haven't been introduced.
5. **Commit:** Commit your changes clearly (`git commit -m "Add support for Hopper FP8 profiling"`).
6. **Push:** Push to your fork (`git push origin feature/amazing-new-feature`).
7. **Pull Request:** Open a Pull Request against the `master` branch of this repository.

## Reporting Bugs

If you run into an NCCL crash, an InfiniBand GID issue, or the mathematics of the gradient scaling fail on your specific hardware, please open an Issue! 

When reporting a bug, please include:
- Your exact cluster hardware topology (e.g., Node 1: 2x A100, Node 2: 4x V100).
- The network fabric you are using (TCP, RoCEv2, InfiniBand).
- The exact traceback of the error (remember to scroll up past the PyTorch `ChildFailedError` to find the root `RuntimeError`!).

## Code of Conduct
Please be respectful to all members of the open-source community. We are all here to learn and build faster systems together!
