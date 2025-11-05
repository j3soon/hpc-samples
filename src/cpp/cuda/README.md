# CUDA C++

```sh
# in docker container
cd /workspace/cpp/cuda
```

## Examples

### Hello World

```sh
nvcc hello.cu
./a.out
```

```sh
nvcc real-hello-world.cu
./a.out
```

If there is no output from the first example, and the output is `Hello Hello` in the second example, you may have been using an outdated driver, consider update your GPU driver to the latest release. The error could be observed by using `compute-sanitizer`. Alternatively, you can try to use `nvcc` with the `-arch` flag to specify the architecture of the GPU:

```sh
nvcc -arch=native <SOURCE_FILE>
```

Refs:

- https://github.com/NVIDIA/cuda-samples
- https://github.com/NVIDIA/CUDALibrarySamples

### CUDA Samples

[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) is included in the docker image at `/root/cuda-samples`. For example, to run the `deviceQuery` example, you can run the following command:

```sh
~/cuda-samples/build/Samples/1_Utilities/deviceQuery/deviceQuery
```

See the full list of examples [here](https://github.com/NVIDIA/cuda-samples#samples-list).

## Tools

```
compute-sanitizer ./a.out
```

Refs:

- https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- https://github.com/NVIDIA/compute-sanitizer-samples
