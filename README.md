# HPC Samples

## Prerequisites

- Ubuntu 22.04 or 24.04
- [NVIDIA Driver](https://ubuntu.com/server/docs/nvidia-drivers-installation)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Follow [this post](https://tutorial.j3soon.com/docker/nvidia-gpu-support/) for the installation instructions.

## Clone the repository

```sh
git clone https://github.com/j3soon/hpc-samples.git
cd hpc-samples
```

## Docker Environment

We use the [`nvidia/nvhpc`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nvhpc/tags) NGC image as the base image. See [the documentation](https://docs.nvidia.com/hpc-sdk/hpc-sdk-container/index.html) for more details.

```sh
cd src

docker build -f Dockerfile_cuda13.0 -t j3soon/hpc-samples:nvhpc-25.9-devel-cuda13.0-ubuntu24.04 .
docker build -f Dockerfile_cuda12.9 -t j3soon/hpc-samples:nvhpc-25.7-devel-cuda12.9-ubuntu24.04 .
docker build -f Dockerfile_cuda12.4 -t j3soon/hpc-samples:nvhpc-24.5-devel-cuda12.4-ubuntu22.04 .

docker run --rm -it --gpus all -v $PWD:/app j3soon/hpc-samples:nvhpc-25.9-devel-cuda13.0-ubuntu24.04
docker run --rm -it --gpus all -v $PWD:/app j3soon/hpc-samples:nvhpc-25.7-devel-cuda12.9-ubuntu24.04
docker run --rm -it --gpus all -v $PWD:/app j3soon/hpc-samples:nvhpc-24.5-devel-cuda12.4-ubuntu22.04
```

## Examples

### Built-in Examples

To compile, run, and clean the built-in examples at `/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples`, you can use the following commands:

```sh
# C++ Standard Parallelism
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/stdpar/stdblas
make all
# OpenACC Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/OpenACC/samples
make all
# OpenMP Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/OpenMP
make all
# CUDA-Libraries Examples
# - cuBLAS
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/CUDA-Libraries/cuBLAS
make all
# - cuFFT
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/CUDA-Libraries/cuFFT
make all
# - cuRAND
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/CUDA-Libraries/cuRAND
make all
# - cuSPARSE
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/CUDA-Libraries/cuSPARSE
make all
# - thrust
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/CUDA-Libraries/thrust
make all
# MPI Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/MPI
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
make all

# CUDA-Fortran Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/CUDA-Fortran/CUDA-Fortran-Book
make all
# AutoPar Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/AutoPar
make all
# F2003 Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/F2003
make all
# NVLAmath Examples
cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/examples/NVLAmath
make all
```

### CUDA Samples

[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) has been pre-built and included in the docker image at `/workspace/cuda-samples`. For example, to run the `deviceQuery` example, you can run the following command:

```sh
/workspace/cuda-samples/build/Samples/1_Utilities/deviceQuery/deviceQuery
```

or the `p2pBandwidthLatencyTest` example to test the [GPU-to-GPU communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#gpu-to-gpu-communication):

```sh
/workspace/cuda-samples/build/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
```

See the full list of examples [here](https://github.com/NVIDIA/cuda-samples#samples-list).

> If you are using a custom docker image, follow the official instructions:
> ```sh
> git clone https://github.com/NVIDIA/cuda-samples
> cd cuda-samples
> git checkout v13.0  # Replace with the CUDA version matching your image
> mkdir build && cd build
> cmake ..
> make -j$(nproc)
> ```
> You might also need to set `CUDA_PATH` and `LIBRARY_PATH` according to your environment if the build fails.

### NCCL Tests

[NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) has been pre-built and included in the docker image at `/workspace/nccl-tests`. For example, to run the `all_reduce_perf` test, you can run the following command:

```sh
cd /workspace/nccl-tests
# single node 8 GPUs
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
# two node 16 GPUs
mpirun -np 16 -N 2 ./build/all_reduce_perf_mpi -b 8 -e 8G -f 2 -g 1
```

or with Slurm:

```sh
# Enroot+Pyxis
srun -N 2 --ntasks-per-node=8 --mpi=pmix \
  --container-image=j3soon/hpc-samples:nvhpc-24.5-devel-cuda12.4-ubuntu22.04 \
  /usr/local/bin/hpcx-entrypoint.sh \
  /workspace/nccl-tests/build/all_reduce_perf_mpi -b 8 -e 8G -f 2 -g 1
# Apptainer/Singularity (To be confirmed)
singularity pull docker://j3soon/hpc-samples:nvhpc-24.5-devel-cuda12.4-ubuntu22.04
singularity build --sandbox hpc-samples-cuda12/ hpc-samples_nvhpc-24.5-devel-cuda12.4-ubuntu22.04.sif
srun -N 2 --ntasks-per-node 8 --mpi=pmix --gres=gpu:8 \
  singularity exec --nv hpc-samples-cuda12/ \
  /usr/local/bin/hpcx-entrypoint.sh \
  /workspace/nccl-tests/build/all_reduce_perf_mpi -b 8 -e 8G -f 2 -g 1
```

or with [debug flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug-subsys):

```sh
cd /workspace/nccl-tests
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

> If you are using a custom docker image, follow the official instructions:
> ```sh
> git clone https://github.com/NVIDIA/nccl-tests
> cd nccl-tests
> git checkout v2.17.6  # Replace with the NCCL version matching your image
> make -j$(nproc)
> make -j$(nproc) MPI=1 NAME_SUFFIX=_mpi
> ```
> You might also need to set `CUDA_HOME`, `NCCL_HOME`, and `MPI_HOME` according to your environment if the build fails.

### NVBandwidth

[NVIDIA/nvbandwidth](https://github.com/NVIDIA/nvbandwidth) has been pre-built and included in the docker image at `/workspace/nvbandwidth`. For example, to run the `nvbandwidth` tool, you can run the following command:

```sh
cd /workspace/nvbandwidth
./nvbandwidth
```

or verbose mode:

```sh
./nvbandwidth -v
```

or single test case:

```sh
./nvbandwidth -t device_to_device_memcpy_read_ce
```

or the multi-node version:

```sh
cd /workspace/nvbandwidth_mpi
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -n 4 ./nvbandwidth -p multinode
```

> If you are using a custom docker image, follow the official instructions:
> ```sh
> git clone https://github.com/NVIDIA/nvbandwidth
> cd nvbandwidth
> git checkout v0.8  # Replace with the NVBandwidth version matching your image
> cp -r . ../nvbandwidth_mpi
> apt-get update && apt-get install -y libboost-program-options-dev
> cmake .
> make -j$(nproc)
> cd ../nvbandwidth_mpi
> cmake -DMULTINODE=1 .
> make -j$(nproc)
> ```

### CUDA Library Samples

[NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples) is not yet included.

### Compute Sanitizer Samples

[NVIDIA/compute-sanitizer-samples](https://github.com/NVIDIA/compute-sanitizer-samples) is not yet included.

### Multi GPU Programming Models

[NVIDIA/multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models) is not yet included.

## Tools

### NVIDIA-SMI

Use the [`nvidia-smi`](https://docs.nvidia.com/deploy/nvidia-smi/index.html) tool to query GPU status.

Check local GPU [topology status](https://docs.nvidia.com/multi-node-nvlink-systems/mnnvl-user-guide/verifying.html#topology-status):

```sh
nvidia-smi topo -p2p n
```

Topology connections and affinities matrix between the GPUs and NICs in the system:

```sh
nvidia-smi topo -m
```

### Compute Sanitizer

Use [`compute-sanitizer`](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) to detect CUDA errors.

```sh
compute-sanitizer ./a.out
```

## Nsight Guided Profiling

See [nsight-guided-profiling.md](nsight-guided-profiling.md) for more details.
