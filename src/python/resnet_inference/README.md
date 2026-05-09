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
python 03_pytorch_loader_tuning.py
python 04_pytorch_non_blocking.py
python 05_pytorch_gpu_accum.py
python 06_pytorch_cuda_stream.py
TORCH_LOGS="cudagraphs,recompiles" python 07-1_pytorch_compile.py
TORCH_LOGS="cudagraphs,recompiles" python 07-2_pytorch_cuda_graph_fix.py
```

## Profile

Inside the container, follow the nsys [command line examples](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines) and [python profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling):

```sh
mkdir -p profiles

FILES=(
  01_pytorch_basic
  02_pytorch_batch
  03_pytorch_loader_tuning
  04_pytorch_non_blocking
  05_pytorch_gpu_accum
  06_pytorch_cuda_stream
  07-1_pytorch_compile
  07-2_pytorch_cuda_graph_fix
)

for FILE in "${FILES[@]}"; do
  nsys profile \
    --cudabacktrace=none \
    --output=./profiles/${FILE}.nsys-rep \
    python ${FILE}.py
done
```

<!--
```sh
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx,python-gil --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=all --python-backtrace=cuda --python-sampling=true \
  --output=./profiles/${FILE}.nsys-rep \
  python ${FILE}.py
```
-->

For more `nsys` flags, see the [documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options).
