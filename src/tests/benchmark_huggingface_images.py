import logging
import time
import dataclasses
from typing import List

import cvcuda
from datasets import load_dataset, Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import profiler
from torch.utils.tensorboard import SummaryWriter

from rich.console import Console
from rich.table import Table

from gradient_mechanics.data.torch_loading import GPUDataLoader
from gradient_mechanics.data import transforms


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BenchmarkResult:
    description: str
    time_taken: float
    images_loaded: int
    throughput: float


def summarize_benchmark_results(
    baseline: BenchmarkResult, results: List[BenchmarkResult]
) -> Table:
    """
    Summarize the benchmark results using a rich.Table  with deltas to the baseline.

    Args:
        baseline (BenchmarkResult): The baseline benchmark result.
        results (List[BenchmarkResult]): List of benchmark results to summarize.
    """
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Description", justify="center")
    table.add_column("Time Taken (s)", justify="center")
    table.add_column("Samples Loaded", justify="center")
    table.add_column("Throughput (samples/s)", justify="center")

    def format_delta(value, baseline_value):
        delta = value - baseline_value
        if delta == 0:
            return f"{value:.0f}"
        elif delta > 0:
            return f"[green]{value:.0f} (+{delta:.0f})[/green]"
        else:
            return f"[red]{value:.0f} ({delta:.0f})[/red]"

    table.add_row(
        baseline.description,
        f"{baseline.time_taken:.0f}",
        f"{baseline.images_loaded}",
        f"{baseline.throughput:.0f}",
    )

    for result in results:
        table.add_row(
            result.description,
            format_delta(result.time_taken, baseline.time_taken),
            format_delta(result.images_loaded, baseline.images_loaded),
            format_delta(result.throughput, baseline.throughput),
        )

    return table


def benchmark(
    num_workers: int = 12,
    batch_size: int = 140,
    decode_image: bool = False,
    decode_image_on_gpu: bool = False,
    include_model: bool = False,
    profile: bool = False,
) -> BenchmarkResult:
    benchmark_name = f"decode_image={decode_image}, decode_image_on_gpu={decode_image_on_gpu}, num_workers={num_workers}, batch_size={batch_size}, include_model={include_model}"

    if profile:
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/log_dir"),
        ) as prof:
            with profiler.record_function(benchmark_name):
                return run_benchmark(
                    num_workers=num_workers,
                    batch_size=batch_size,
                    decode_image=decode_image,
                    decode_image_on_gpu=decode_image_on_gpu,
                    include_model=include_model,
                    profiler=prof,
                )
    else:
        return run_benchmark(
            num_workers=num_workers,
            batch_size=batch_size,
            decode_image=decode_image,
            decode_image_on_gpu=decode_image_on_gpu,
            include_model=include_model,
        )


def run_benchmark(
    num_workers: int,
    batch_size: int,
    decode_image: bool = False,
    decode_image_on_gpu: bool = False,
    include_model: bool = False,
    profiler: profiler.profile = None,
) -> BenchmarkResult:
    dataset = load_dataset("beans", split="train").cast_column(
        "image", Image(decode=decode_image)
    )
    dataset.set_format(type="numpy", columns=["image", "labels"])

    model = None
    optimizer = None
    loss = None
    if include_model:
        model = torchvision.models.resnet50(weights=None)
        model = torch.nn.Sequential(model, torch.nn.Linear(1000, 3)).to("cuda:0")
        if decode_image_on_gpu:
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = torch.nn.CrossEntropyLoss()

    images_loaded = 0
    started_at = time.perf_counter()

    if decode_image_on_gpu:

        def transform_fn(samples):
            images = []
            # Convert to EncodedImage that can be used with GPU decoding
            for sample in samples["image"]:
                mutable_array = bytearray(sample["bytes"])
                image_tensor = torch.frombuffer(mutable_array, dtype=torch.uint8)
                images.append(transforms.EncodedImage(buffer=image_tensor))

            return {
                "image": images,
                "labels": samples["labels"],
            }

        dataset.set_transform(transform_fn)

        device_id = 0
        torch_stream = torch.cuda.Stream(device=device_id)
        gpu_transforms = [
            transforms.JPEGDecode(device_id=device_id, stream=torch_stream.cuda_stream),
            transforms.Resize(
                device_id=device_id,
                width=224,
                height=224,
                interpolation=cvcuda.Interp.CUBIC,
                stream=torch_stream.cuda_stream,
            ),
            transforms.ToTensor(device_id=device_id, stream=torch_stream.cuda_stream),
            transforms.AsyncH2D(device_id=device_id, stream=torch_stream.cuda_stream),
        ]
        dataloader = GPUDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            # GPU Specific
            gpu_device=device_id,
            gpu_prefetch_factor=1,
            gpu_num_threads=1,
            gpu_transforms=gpu_transforms,
        )
    elif decode_image:
        cpu_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (224, 224),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )

        def transform_fn(samples):
            images = [cpu_transforms(sample) for sample in samples["image"]]
            return {
                "image": images,
                "labels": samples["labels"],
            }

        dataset.set_transform(transform_fn)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        raise ValueError("Either decode_image or decode_image_on_gpu must be True.")

    load_gap_durations = []
    step_durations = []
    train_step_ended_at = None
    for epoch in range(3):
        for batch in dataloader:
            images = batch["image"]
            labels = batch["labels"]
            if decode_image:
                images = images.to("cuda:0")
                labels = labels.to("cuda:0")
            images_loaded += images.shape[0]
            batch_load_ended_at = time.perf_counter()
            if train_step_ended_at is not None:
                load_step_gap = batch_load_ended_at - train_step_ended_at
                if epoch > 0:
                    load_gap_durations.append(load_step_gap)

            if include_model:
                started_train_step_at = time.perf_counter()
                optimizer.zero_grad()

                predictions = model(images)
                targets = torch.nn.functional.one_hot(labels, num_classes=3).to(
                    torch.float32
                )
                loss_value = loss(predictions, targets)
                loss_value.backward()
                optimizer.step()
                train_step_ended_at = time.perf_counter()
                train_step_duration = train_step_ended_at - started_train_step_at

                if epoch > 0:
                    step_durations.append(train_step_duration)

            if profiler:
                profiler.step()

    ended_at = time.perf_counter()

    print(f"Total time taken: {ended_at - started_at:.3f} s")
    print(f"Images loaded: {images_loaded}")
    throughput = images_loaded / (ended_at - started_at)
    average_load_step_gap = sum(load_gap_durations) / len(load_gap_durations)
    average_train_step_duration = sum(step_durations) / len(step_durations)
    print(f"Average load step gap: {average_load_step_gap:.3f} s")
    print(f"Average train step duration: {average_train_step_duration:.3f} s")
    model_only_speedup_factor = (
        average_load_step_gap + average_train_step_duration
    ) / average_train_step_duration
    print(f"Model only speedup factor: {model_only_speedup_factor:.3f}")
    print(f"Measured combined throughput: {throughput:.0f} samples/s")
    estimated_model_only_throughput = model_only_speedup_factor * throughput
    print(
        f"Estimated model only throughput: {estimated_model_only_throughput:.0f} samples/s"
    )

    # TODO: use the same name as the profiler
    return BenchmarkResult(
        description=f"decode_image={decode_image}, decode_image_on_gpu={decode_image_on_gpu}, num_workers={num_workers}, batch_size={batch_size}, include_model={include_model}",
        time_taken=ended_at - started_at,
        images_loaded=images_loaded,
        throughput=images_loaded / (ended_at - started_at),
    )


if __name__ == "__main__":
    profile = False
    console = Console()
    writer = SummaryWriter("runs/log_dir")

    results = []
    # CPU optimal: 4 workers with bs 64 @ 398 samples/s
    for num_workers in [4]:
        for batch_size in [64]:
            console.print(
                f"Benchmarking with image decoding on CPU. Batch size: {batch_size}, num_workers: {num_workers}."
            )
            baseline = benchmark(
                num_workers=num_workers,
                batch_size=batch_size,
                include_model=True,
                decode_image=True,
                decode_image_on_gpu=False,
                profile=profile,
            )
            results.append(baseline)

    for num_workers in [2]:
        for batch_size in [128]:
            console.print(
                f"Benchmarking with image decoding on GPU. Batch size: {batch_size}, num_workers: {num_workers}."
            )
            decode_on_gpu = benchmark(
                batch_size=batch_size,
                num_workers=num_workers,
                include_model=True,
                decode_image=False,
                decode_image_on_gpu=True,
                profile=profile,
            )
            results.append(decode_on_gpu)

    table = summarize_benchmark_results(results[0], results[1:])
    console.print(table)
    writer.flush()
    writer.close()
