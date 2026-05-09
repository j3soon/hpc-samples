# ResNet152 Inference

Step-by-step ResNet152 inference profiling examples for PyTorch and Nsight Systems.

## Docker Environment

This sample uses the [NVIDIA PyTorch NGC image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?version=25.06-py3) as the base image:

```sh
cd src/python/resnet_inference

docker build -f Dockerfile_25_06 -t j3soon/hpc-samples:resnet-inference-25.06 .
docker build -f Dockerfile_26_04 -t j3soon/hpc-samples:resnet-inference-26.04 .

docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -v $PWD:/workspace \
  -v .cache:/root/.cache \
  j3soon/hpc-samples:resnet-inference-26.04
```

## Run

Inside the container:

```sh
python 01_base.py
python 02_batch.py
python 03_dataloader_tuning.py
python 04_non_blocking_copy.py
python 05_gpu_accum.py
python 06_cuda_stream_prefetch.py
TORCH_LOGS="cudagraphs,recompiles" python 07_1_torch_compile.py
TORCH_LOGS="cudagraphs,recompiles" python 07_2_cuda_graph_fix.py
```

## Profile

Inside the container, follow the nsys [command line examples](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines) and [python profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling):

```sh
mkdir -p profiles

FILES=(
  01_base
  02_batch
  03_dataloader_tuning
  04_non_blocking_copy
  05_gpu_accum
  06_cuda_stream_prefetch
  07_1_torch_compile
  07_2_cuda_graph_fix
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
