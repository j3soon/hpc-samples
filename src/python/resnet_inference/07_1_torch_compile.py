import os
import time

import nvtx
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152
from imagenetv2_pytorch import ImageNetV2Dataset


DATA_DIR = "./data"
MAX_IMAGES = 10000
BATCH_SIZE = 128
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
WARMUP_RUNS = 3


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


# Ref: https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
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


def main():
    assert torch.cuda.is_available(), "CUDA required."

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"num workers: {NUM_WORKERS}")
    print(f"prefetch factor: {PREFETCH_FACTOR}")

    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device).eval()
    # Ref: https://docs.pytorch.org/docs/2.11/generated/torch.compile.html?utm_source=chatgpt.com
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset = ImageNetV2Dataset(
        variant="matched-frequency",
        transform=imagenet_preprocess(),
        location=DATA_DIR,
    )
    dataset = Subset(dataset, range(MAX_IMAGES))
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
    with torch.inference_mode():
        with nvtx.annotate("warmup"):
            for _ in range(WARMUP_RUNS):
                model(dummy)
    torch.cuda.synchronize()

    top1_correct_gpu = torch.zeros((), device=device)
    top5_correct_gpu = torch.zeros((), device=device)

    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()

        prefetcher = DataPrefetcher(loader, device)
        images, targets = prefetcher.next()
        while images is not None:
            with nvtx.annotate("forward_pass"):
                logits = model(images)

            with nvtx.annotate("accuracy"):
                _, pred = logits.topk(5, dim=1)
                matches = pred.eq(targets.view(-1, 1))
                top1_correct_gpu += matches[:, :1].sum()
                top5_correct_gpu += matches.sum()

            images, targets = prefetcher.next()

        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    latency_ms = elapsed_s * 1000.0
    fps = len(dataset) / elapsed_s
    top1_correct = top1_correct_gpu.item()
    top5_correct = top5_correct_gpu.item()
    top1 = 100.0 * top1_correct / len(dataset)
    top5 = 100.0 * top5_correct / len(dataset)

    print(f"images: {len(dataset)}")
    print(f"latency for {len(dataset)} images: {latency_ms:.3f} ms")
    print(f"throughput: {fps:.2f} img/s")
    print(f"top-1: {top1:.2f}%")
    print(f"top-5: {top5:.2f}%")


if __name__ == "__main__":
    main()
