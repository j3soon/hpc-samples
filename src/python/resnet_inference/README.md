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
python 04_1_gpu_accuracy_accum.py
python 04_2_non_blocking_h2d.py
python 05_cuda_stream_prefetch.py
TORCH_LOGS="cudagraphs,recompiles" python 06_1_torch_compile.py
TORCH_LOGS="cudagraphs,recompiles" python 06_2_cuda_graph_fix.py
TORCH_LOGS="cudagraphs,recompiles" python 06_3_benchmark.py
TORCH_LOGS="cudagraphs,recompiles" python 07_1_amp_fp32.py
TORCH_LOGS="cudagraphs,recompiles" python 07_2_amp_tf32.py
TORCH_LOGS="cudagraphs,recompiles" python 07_3_amp_bf16.py
TORCH_LOGS="cudagraphs,recompiles" python 07_4_amp_fp16.py
python 08_1_tensorrt_fp16.py
python 08_2_tensorrt_best.py
python 09_dali.py
python 10_channels_last.py
```

## Profile

Inside the container, follow the nsys [command line examples](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines) and [python profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling):

```sh
mkdir -p profiles

FILES=(
  01_base
  02_batch
  03_dataloader_tuning
  04_1_gpu_accuracy_accum
  04_2_non_blocking_h2d
  05_cuda_stream_prefetch
  06_1_torch_compile
  06_2_cuda_graph_fix
  06_3_benchmark
  07_1_amp_fp32
  07_2_amp_tf32
  07_3_amp_bf16
  07_4_amp_fp16
  08_1_tensorrt_fp16
  08_2_tensorrt_best
  09_dali
  10_channels_last
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
