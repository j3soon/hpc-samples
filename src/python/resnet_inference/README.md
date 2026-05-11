# ResNet152 Inference

## Prerequisites

- [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started)

Downloading the two Nsight GUIs are sufficient, as we have provide pre-profiled reports for the examples in the repository.

## (Optional) Profiler and Container Setup

Launch container with [`SYS_ADMIN` caps](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#container-and-scheduler-support):

```sh
cd src/python/resnet_inference

# docker build -f Dockerfile_25_06 -t j3soon/hpc-samples:resnet-inference-25.06 .
docker build -f Dockerfile_26_04 -t j3soon/hpc-samples:resnet-inference-26.04 .

docker run --rm -it --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -v $PWD:/workspace \
  -v .cache:/root/.cache \
  j3soon/hpc-samples:resnet-inference-26.04
```

## ResNet152 Inference

In the container:

```sh
cd /workspace
```

and run all tests:

```sh
./run_all.sh
```

and run all profiles:

```sh
./profile_all.sh
```

If you don't have an environment, download the reports from [here](https://github.com/j3soon/hpc-samples/releases).

- [01_base.py](01_base.py)

  * Baseline: batch size 1 PyTorch inference.

- [02_batch.py](02_batch.py)

  * Improved: batched inference with batch size 64.

- [03_dataloader_tuning.py](03_dataloader_tuning.py)

  * Improved: DataLoader workers, pinned memory, and prefetching.

- [04_1_gpu_accuracy_accum.py](04_1_gpu_accuracy_accum.py)

  * Improved: keep accuracy accumulation on GPU.

- [04_2_non_blocking_h2d.py](04_2_non_blocking_h2d.py)

  * Improved: non-blocking host-to-device copies.

- [05_cuda_stream_prefetch.py](05_cuda_stream_prefetch.py)

  * Improved: dedicated CUDA stream for prefetching.

- [06_1_torch_compile.py](06_1_torch_compile.py)

  * Improved: `torch.compile`, with 2 caveats (last batch and accidental re-compile) that will be fixed in the next example.

- [06_2_cuda_graph_fix.py](06_2_cuda_graph_fix.py)

  * Improved: stable shape for CUDA graph capture.

- [06_3_benchmark.py](06_3_benchmark.py)

  * Improved: enable cudnn backend autotuning.

- [07_1_amp_fp32.py](07_1_amp_fp32.py)

  * Baseline: compiled FP32 path with TF32 disabled.

- [07_2_amp_tf32.py](07_2_amp_tf32.py)

  * Improved: TF32, which is essentially same as `06_3_benchmark.py`.

- [07_3_amp_bf16.py](07_3_amp_bf16.py)

  * Improved: BF16 autocast.

- [07_4_amp_fp16.py](07_4_amp_fp16.py)

  * Improved: FP16 autocast.

- [08_1_tensorrt_fp16.py](08_1_tensorrt_fp16.py)

  * Improved: ONNX export and TensorRT FP16 engine (using `FP32+FP16` in our case).

- [08_2_tensorrt_best.py](08_2_tensorrt_best.py)

  * Improved: TensorRT `--best` engine (using `FP32+FP16+BF16+INT8` in our case).

- [09_dali.py](09_dali.py)

  * Improved: NVIDIA DALI image pipeline.

- [10_channels_last.py](10_channels_last.py)

  * Improved: DALI with TensorRT HWC input format.
