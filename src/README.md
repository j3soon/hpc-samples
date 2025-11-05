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

```sh
docker build -t j3soon/hpc-samples .
docker run --rm -it --gpus all -v $PWD:/workspace j3soon/hpc-samples
```
