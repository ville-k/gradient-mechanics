import argparse
import logging
import time

from gradient_mechanics.data import torch_loading, torchdata_loading
from gradient_mechanics.data import transforms
from gradient_mechanics.data import video_transforms
from tests import video_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark VideoDataset")
    parser.add_argument("video_file_path", type=str)
    parser.add_argument(
        "--dataloader-cls", type=str, default="torch", choices=["torch", "torchdata"]
    )
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    video_file_path = args.video_file_path
    episode_length = args.episode_length
    episode_stride = args.episode_stride
    batch_size = args.batch_size
    num_workers = args.num_workers
    device_id = args.device_id
    if args.dataloader_cls == "torchdata":
        dataloader_cls = torchdata_loading.GPUDataLoader
    elif args.dataloader_cls == "torch":
        dataloader_cls = torch_loading.GPUDataLoader
    else:
        raise ValueError(f"Invalid dataloader class: {args.dataloader_cls}")

    dataset = video_dataset.VideoDataset(
        video_file_path, episode_length=episode_length, episode_stride=episode_stride
    )

    gpu_transforms = [
        video_transforms.DecodeVideo(device_id=device_id),
        transforms.ToTensor(device_id=device_id),
    ]
    loader = torch_loading.GPUDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # GPU Specific
        gpu_device=device_id,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )

    started_at = time.perf_counter()
    samples_loaded = 0
    batches_loaded = 0
    load_gaps = []

    load_started_at = time.perf_counter()
    for i, batch in enumerate(loader):
        load_ended_at = time.perf_counter()
        load_gaps.append(load_ended_at - load_started_at)

        load_started_at = time.perf_counter()
        print(f"Batch: {i} - {len(batch)} frames in {load_gaps[-1]:.4f} seconds")
        batches_loaded += 1
        for sample in batch:
            samples_loaded += sample.shape[0]

    ended_at = time.perf_counter()

    throughput = samples_loaded / (ended_at - started_at)
    print(f"Samples loaded: {samples_loaded}")
    print(f"Time taken: {ended_at - started_at:.2f} seconds")
    print(f"Throughput: {throughput:.2f} samples per second")
    print(
        f"Throughput: {batches_loaded / (ended_at - started_at):.2f} batches per second"
    )

    warmup = 2
    load_gaps = load_gaps[warmup:]
    print(f"Mean load gap: {sum(load_gaps) / len(load_gaps):.4f}")
    print(f"Max load gap: {max(load_gaps):.4f}")
    print(f"Min load gap: {min(load_gaps):.4f}")
    print(f"Num workers: {num_workers}")
    # print out the dataloader class and module
    print(f"Dataloader class: {dataloader_cls.__name__}")
    print(f"Dataloader module: {dataloader_cls.__module__}")
