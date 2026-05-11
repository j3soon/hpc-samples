import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nvtx
import tensorrt as trt
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152
from imagenetv2_pytorch import ImageNetV2Dataset

torch.backends.cudnn.benchmark = True


DATA_DIR = "./data"
RESULTS_DIR = "./results"
BATCH_SIZE = 64
MAX_IMAGES = 10000
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
WARMUP_RUNS = 3
ONNX_PATH = Path(RESULTS_DIR) / f"resnet152_bs{BATCH_SIZE}.onnx"
ENGINE_PATH = Path(RESULTS_DIR) / f"resnet152_bs{BATCH_SIZE}_best.engine"


# Ref: https://pytorch.org/hub/pytorch_vision_resnet/
def imagenet_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def padded_collate(batch):
    images, targets = default_collate(batch)

    if images.shape[0] < BATCH_SIZE:
        pad_size = BATCH_SIZE - images.shape[0]
        image_padding = images.new_zeros((pad_size, *images.shape[1:]))
        target_padding = targets.new_zeros((pad_size,))
        images = torch.cat((images, image_padding), dim=0)
        targets = torch.cat((targets, target_padding), dim=0)

    return images, targets


# Ref: https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            with nvtx.annotate("dataloader_next"):
                self.next_images, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_images = None
            self.next_targets = None
            return

        with torch.cuda.stream(self.stream):
            with nvtx.annotate("prefetch_h2d"):
                self.next_images = self.next_images.to(self.device, non_blocking=True)
                self.next_targets = self.next_targets.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images = self.next_images
        targets = self.next_targets
        if images is not None:
            images.record_stream(torch.cuda.current_stream())
            targets.record_stream(torch.cuda.current_stream())
        self.preload()
        return images, targets


def export_onnx(device):
    if ONNX_PATH.exists():
        return

    model = resnet152(weights=ResNet152_Weights.DEFAULT).to(device).eval()
    dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=21,
    )


def build_engine():
    if ENGINE_PATH.exists():
        return True

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        print("skip: trtexec not found")
        return False

    # Ref: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#convert-the-model
    # Ref: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html
    cmd = [
        trtexec,
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={ENGINE_PATH}",
        "--best",
        "--skipInference",
    ]
    subprocess.run(cmd, check=True)
    return True


def load_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    # Ref: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html#deserializing-a-plan
    with ENGINE_PATH.open("rb") as f:
        model_data = f.read()
    engine = runtime.deserialize_cuda_engine(model_data)

    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    return engine, input_name, output_name


class TensorRTModel:
    def __init__(self, device):
        engine, input_name, output_name = load_engine()

        # Ref: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html#performing-inference
        self.context = engine.create_execution_context()
        self.context.set_input_shape(input_name, (BATCH_SIZE, 3, 224, 224))
        output_shape = tuple(self.context.get_tensor_shape(output_name))
        output_dtype = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
        }[engine.get_tensor_dtype(output_name)]

        self.input_name = input_name
        self.output_name = output_name
        self.logits = torch.empty(output_shape, device=device, dtype=output_dtype)
        self.stream = torch.cuda.Stream()

    def __call__(self, images):
        self.stream.wait_stream(torch.cuda.current_stream())
        images.record_stream(self.stream)
        self.logits.record_stream(self.stream)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, self.logits.data_ptr())
        with torch.cuda.stream(self.stream):
            assert self.context.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.stream)
        return self.logits


def main():
    assert torch.cuda.is_available(), "CUDA required."

    max_images = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_IMAGES
    device = torch.device("cuda")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    export_onnx(device)
    assert build_engine(), "Failed to build TensorRT engine"
    model = TensorRTModel(device)

    dataset = ImageNetV2Dataset(
        variant="matched-frequency",
        transform=imagenet_preprocess(),
        location=DATA_DIR,
    )
    dataset = Subset(dataset, range(max_images))
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=padded_collate,
    )

    with torch.inference_mode():
        dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
        with nvtx.annotate("warmup"):
            for _ in range(WARMUP_RUNS):
                model(dummy)
    torch.cuda.synchronize()

    top1_correct_gpu = torch.zeros((), device=device)
    top5_correct_gpu = torch.zeros((), device=device)

    with torch.inference_mode():
        def run_batch(images, targets):
            nonlocal top1_correct_gpu, top5_correct_gpu

            with nvtx.annotate("forward_pass"):
                logits = model(images)

            with nvtx.annotate("accuracy"):
                _, pred = logits.topk(5, dim=1)
                matches = pred.eq(targets.view(-1, 1))
                top1_correct_gpu += matches[:, :1].sum()
                top5_correct_gpu += matches.sum()

        prefetcher = DataPrefetcher(loader, device)
        images, targets = prefetcher.next()
        with nvtx.annotate("first_batch"):
            run_batch(images, targets)

        torch.cuda.synchronize()
        start = time.perf_counter()

        images, targets = prefetcher.next()
        while images is not None:
            run_batch(images, targets)
            images, targets = prefetcher.next()

        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    latency_ms = elapsed_s * 1000.0
    timed_images = len(dataset) - BATCH_SIZE
    fps = timed_images / elapsed_s
    top1_correct = top1_correct_gpu.item()
    top5_correct = top5_correct_gpu.item()
    top1 = 100.0 * top1_correct / len(dataset)
    top5 = 100.0 * top5_correct / len(dataset)

    print(
        f"throughput: {fps:.2f} img/s, latency for {timed_images} images: "
        f"{latency_ms:.3f} ms, images: {len(dataset)}, top-1: {top1:.2f}%, "
        f"top-5: {top5:.2f}%"
    )


if __name__ == "__main__":
    main()
