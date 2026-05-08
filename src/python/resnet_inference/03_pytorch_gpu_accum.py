import argparse
import os
import time

import nvtx
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152
from imagenetv2_pytorch import ImageNetV2Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="03 PyTorch ResNet152 inference with GPU metric accumulation")
    parser.add_argument("--data-dir", default="./data", help="Directory for ImageNet-V2 download/cache")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    return parser.parse_args()


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


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required."

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"batch size: {args.batch_size}")

    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device).eval()
    os.makedirs(args.data_dir, exist_ok=True)
    dataset = ImageNetV2Dataset(
        variant="matched-frequency",
        transform=imagenet_preprocess(),
        location=args.data_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    top1_correct_gpu = torch.zeros((), device=device)
    top5_correct_gpu = torch.zeros((), device=device)
    n_samples = 0

    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()

        for images, targets in loader:
            with nvtx.annotate("h2d_transfer"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            with nvtx.annotate("forward_pass"):
                logits = model(images)

            _, pred = logits.topk(5, dim=1)
            matches = pred.eq(targets.view(-1, 1))
            top1_correct_gpu += matches[:, :1].sum()
            top5_correct_gpu += matches.sum()
            n_samples += targets.numel()

        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start

    latency_ms = elapsed_s * 1000.0
    fps = n_samples / elapsed_s
    top1_correct = top1_correct_gpu.item()
    top5_correct = top5_correct_gpu.item()
    top1 = 100.0 * top1_correct / n_samples
    top5 = 100.0 * top5_correct / n_samples

    print(f"images: {n_samples}")
    print(f"total latency: {latency_ms:.3f} ms")
    print(f"throughput: {fps:.2f} img/s")
    print(f"top-1: {top1:.2f}%")
    print(f"top-5: {top5:.2f}%")


if __name__ == "__main__":
    main()
