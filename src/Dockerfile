FROM nvcr.io/nvidia/nvhpc:25.9-devel-cuda13.0-ubuntu24.04

# Common Tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    htop \
    iputils-ping \
    nano \
    net-tools \
    plocate \
    tmux \
    tree \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Set environment variables for cuda-samples
ENV CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0
ENV LIBRARY_PATH=$LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64
# Clone cuda-samples
RUN git clone --depth 1 -b v13.0 https://github.com/NVIDIA/cuda-samples
RUN cd /workspace/cuda-samples \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc)

# Set environment variables for nccl-tests
ENV CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0
ENV NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/comm_libs/13.0/nccl
ENV MPI_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/comm_libs/13.0/hpcx/hpcx-2.24/ompi
# Clone nccl-tests
RUN git clone --depth 1 -b v2.17.6 https://github.com/NVIDIA/nccl-tests
RUN cd /workspace/nccl-tests \
    && make -j$(nproc)
RUN cd /workspace/nccl-tests \
    && make -j$(nproc) MPI=1 NAME_SUFFIX=_mpi
