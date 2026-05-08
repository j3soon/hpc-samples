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
python 02_pytorch_batch.py
```

## Profile

Inside the container, follow the nsys [command line examples](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines) and [python profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling):

```sh
mkdir -p profiles

FILE=01_pytorch_basic
nsys profile \
  --cudabacktrace=none \
  --output=./profiles/${FILE}.nsys-rep \
  python ${FILE}.py
```

<!--
```sh
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx,python-gil --pytorch=functions-trace,autograd-nvtx,autograd-shapes-nvtx \
  --cudabacktrace=all --python-backtrace=cuda --python-sampling=true \
  --output=./profiles/${FILE}.nsys-rep \
  python ${FILE}.py
```
-->

For more `nsys` flags, see the [documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options).
