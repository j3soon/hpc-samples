FROM nvcr.io/nvidia/nvhpc:24.5-devel-cuda_multi-ubuntu22.04

# Common Tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    htop \
    iputils-ping \
    nano \
    net-tools \
    tmux \
    tree \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Set environment variables
ENV CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
ENV LIBRARY_PATH=$LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/lib64

# Clone cuda-samples
RUN git clone --depth 1 -b v12.5 https://github.com/NVIDIA/cuda-samples

WORKDIR /workspace
