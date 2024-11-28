# CUDA C++

## Docker Environment

```sh
docker run --rm -it --gpus all -v $PWD:/workspace j3soon/hpc-notes
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

If there is no output from the first example, you may have been using an outdated driver, consider update your GPU driver to the latest release. The error could be observed by using `compute-sanitizer`.
