from setuptools import setup, find_packages

setup(
    name="asymmetric_ddp",
    version="1.0.0",
    description="Asymmetric DDP: Heterogeneous Cluster Benchmark",
    author="Zealot88",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "asym-ddp=asymmetric_ddp.cli:main",
        ],
    },
)
