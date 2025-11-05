FROM nvcr.io/nvidia/nvhpc:25.9-devel-cuda13.0-ubuntu24.04

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
ENV CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0
ENV LIBRARY_PATH=$LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64

# Clone cuda-samples
RUN git clone --depth 1 -b v13.0 https://github.com/NVIDIA/cuda-samples
RUN cd /root/cuda-samples \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc)

WORKDIR /workspace
