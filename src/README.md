# Source of HPC Notes

## Prerequisites

- Ubuntu 22.04
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
docker build -t j3soon/hpc-samples .
docker run --rm -it --gpus all -v $PWD:/workspace j3soon/hpc-samples
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

[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) has been pre-built and included in the docker image at `/root/cuda-samples`. For example, to run the `deviceQuery` example, you can run the following command:

```sh
~/cuda-samples/build/Samples/1_Utilities/deviceQuery/deviceQuery
```

See the full list of examples [here](https://github.com/NVIDIA/cuda-samples#samples-list).
