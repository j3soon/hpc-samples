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

# Clone cuda-samples
RUN git clone --depth 1 -b v12.5 https://github.com/NVIDIA/cuda-samples

WORKDIR /workspace
