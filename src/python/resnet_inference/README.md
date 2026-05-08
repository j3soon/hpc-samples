# ResNet152 Inference

Step-by-step ResNet152 inference profiling examples for PyTorch and Nsight Systems.

## Docker Environment

This sample uses the [NVIDIA PyTorch NGC image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?version=25.06-py3) as the base image:

```sh
cd src/python/resnet_inference

docker build -t j3soon/hpc-samples:resnet-inference .

docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -v $PWD:/workspace \
  -v .cache:/root/.cache \
  j3soon/hpc-samples:resnet-inference
```

## Run

Inside the container:

```sh
python 01_pytorch_basic.py
```

## Profile

Inside the container:

```sh
mkdir -p profiles

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=./profiles/01_pytorch_basic.nsys-rep \
  --force-overwrite=true \
  python 01_pytorch_basic.py
```

For more `nsys` flags, see the [documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options).
