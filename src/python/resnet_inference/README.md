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

If you don't have an environment, download the `resnet_inference_L40_results.zip` reports from [here](https://github.com/j3soon/hpc-samples/releases).

- [01_base.py](01_base.py) (~67 img/s)

  * Baseline: batch size 1 PyTorch inference.

- [02_batch.py](02_batch.py) (~194 img/s)

  * Improved: batched inference with batch size 64.

- [03_dataloader_tuning.py](03_dataloader_tuning.py) (~1104 img/s)

  * Improved: DataLoader workers, pinned memory, and prefetching.

- [04_1_gpu_accuracy_accum.py](04_1_gpu_accuracy_accum.py) (~1106 img/s)

  * Improved: keep accuracy accumulation on GPU.

- [04_2_non_blocking_h2d.py](04_2_non_blocking_h2d.py) (~1134 img/s)

  * Improved: non-blocking host-to-device copies.

- [05_cuda_stream_prefetch.py](05_cuda_stream_prefetch.py) (~1154 img/s)

  * Improved: dedicated CUDA stream for prefetching.

- [06_1_torch_compile.py](06_1_torch_compile.py) (~1031 img/s)

  * Improved: `torch.compile`, with 2 caveats (last batch and accidental re-compile) that will be fixed in the next example.

- [06_2_cuda_graph_fix.py](06_2_cuda_graph_fix.py) (~1555 img/s)

  * Improved: stable shape for CUDA graph capture.

- [06_3_benchmark.py](06_3_benchmark.py) (~1562 img/s)

  * Improved: enable cudnn backend autotuning.

- [07_1_amp_fp32.py](07_1_amp_fp32.py) (~774 img/s)

  * Baseline: compiled FP32 path with TF32 disabled.

- [07_2_amp_tf32.py](07_2_amp_tf32.py) (~1557 img/s)

  * Improved: TF32, which is essentially same as `06_3_benchmark.py`.

- [07_3_amp_bf16.py](07_3_amp_bf16.py) (~3280 img/s)

  * Improved: BF16 autocast.

- [07_4_amp_fp16.py](07_4_amp_fp16.py) (~3292 img/s)

  * Improved: FP16 autocast.

- [08_1_tensorrt_fp16.py](08_1_tensorrt_fp16.py) (~4715 img/s)

  * Improved: ONNX export and TensorRT FP16 engine (using `FP32+FP16` in our case).

- [08_2_tensorrt_best.py](08_2_tensorrt_best.py) (~5793 img/s)

  * Improved: TensorRT `--best` engine (using `FP32+FP16+BF16+INT8` in our case).

- [09_dali.py](09_dali.py) (~9141 img/s)

  * Improved: NVIDIA DALI image pipeline.

- [10_channels_last.py](10_channels_last.py) (~9249 img/s)

  * Improved: DALI with TensorRT HWC input format.

Further optimizations may include using [lower-precision model with calibration data](https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer/), more optimized pipeline, data format, or code-level accuracy calculation improvements.

**Runtime statistics summary**:

| Step | Throughput |
| --- | ---: |
| 01_base | ~67 img/s |
| 02_batch | ~194 img/s |
| 03_dataloader_tuning | ~1104 img/s |
| 04_1_gpu_accuracy_accum | ~1106 img/s |
| 04_2_non_blocking_h2d | ~1134 img/s |
| 05_cuda_stream_prefetch | ~1154 img/s |
| 06_1_torch_compile | ~1031 img/s |
| 06_2_cuda_graph_fix | ~1555 img/s |
| 06_3_benchmark | ~1562 img/s |
| 07_1_amp_fp32 | ~774 img/s |
| 07_2_amp_tf32 | ~1557 img/s |
| 07_3_amp_bf16 | ~3280 img/s |
| 07_4_amp_fp16 | ~3292 img/s |
| 08_1_tensorrt_fp16 | ~4715 img/s |
| 08_2_tensorrt_best | ~5793 img/s |
| 09_dali | ~9141 img/s |
| 10_channels_last | ~9249 img/s |

Raw results:

```
Testing 01_base...
throughput: 66.97 img/s, latency for 9999 images: 149308.425 ms, images: 10000, top-1: 66.97%, top-5: 87.22%
Testing 02_batch...
throughput: 194.00 img/s, latency for 9936 images: 51216.489 ms, images: 10000, top-1: 66.96%, top-5: 87.22%
Testing 03_dataloader_tuning...
throughput: 1103.98 img/s, latency for 9936 images: 9000.136 ms, images: 10000, top-1: 66.96%, top-5: 87.22%
Testing 04_1_gpu_accuracy_accum...
throughput: 1106.01 img/s, latency for 9936 images: 8983.622 ms, images: 10000, top-1: 66.96%, top-5: 87.22%
Testing 04_2_non_blocking_h2d...
throughput: 1133.68 img/s, latency for 9936 images: 8764.415 ms, images: 10000, top-1: 66.96%, top-5: 87.22%
Testing 05_cuda_stream_prefetch...
throughput: 1153.51 img/s, latency for 9936 images: 8613.738 ms, images: 10000, top-1: 66.96%, top-5: 87.22%
Testing 06_1_torch_compile...
throughput: 1030.51 img/s, latency for 9936 images: 9641.801 ms, images: 10000, top-1: 66.92%, top-5: 87.23%
Testing 06_2_cuda_graph_fix...
throughput: 1555.13 img/s, latency for 9936 images: 6389.174 ms, images: 10000, top-1: 66.92%, top-5: 87.23%
Testing 06_3_benchmark...
throughput: 1561.63 img/s, latency for 9936 images: 6362.587 ms, images: 10000, top-1: 66.93%, top-5: 87.23%
Testing 07_1_amp_fp32...
throughput: 773.80 img/s, latency for 9936 images: 12840.593 ms, images: 10000, top-1: 66.97%, top-5: 87.22%
Testing 07_2_amp_tf32...
throughput: 1556.77 img/s, latency for 9936 images: 6382.464 ms, images: 10000, top-1: 66.93%, top-5: 87.23%
Testing 07_3_amp_bf16...
throughput: 3280.08 img/s, latency for 9936 images: 3029.197 ms, images: 10000, top-1: 66.87%, top-5: 87.23%
Testing 07_4_amp_fp16...
throughput: 3292.26 img/s, latency for 9936 images: 3017.990 ms, images: 10000, top-1: 66.87%, top-5: 87.23%
Testing 08_1_tensorrt_fp16...
throughput: 4714.82 img/s, latency for 9936 images: 2107.396 ms, images: 10000, top-1: 66.97%, top-5: 87.28%
Testing 08_2_tensorrt_best...
throughput: 5792.71 img/s, latency for 9936 images: 1715.260 ms, images: 10000, top-1: 63.89%, top-5: 85.17%
Testing 09_dali...
throughput: 9141.11 img/s, latency for 9936 images: 1086.957 ms, images: 10000, top-1: 64.03%, top-5: 85.35%
Testing 10_channels_last...
throughput: 9248.51 img/s, latency for 9936 images: 1074.336 ms, images: 10000, top-1: 64.03%, top-5: 85.35%
```
