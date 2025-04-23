import logging
import time
from typing import Callable, List
import torch
from torch.utils.data import RandomSampler, SequentialSampler, default_collate
import torchdata.nodes as tn


from gradient_mechanics.data.transforms import Transform


logger = logging.getLogger(__name__)


class ApplyGPUTransforms:
    def __init__(self, transforms: List[Transform], gpu_device: int):
        self.transforms = transforms
        self.gpu_device = gpu_device

    def __call__(self, batch: List[torch.Tensor]):
        measured = []
        logger.debug("start transforms ---")
        started_transforms_at = time.perf_counter()
        for transform in self.transforms:
            started_at = time.perf_counter()
            batch = transform(batch)
            elapsed_time = time.perf_counter() - started_at
            measured.append((elapsed_time, transform.__class__.__name__))
        elapsed_time = time.perf_counter() - started_transforms_at

        logger.debug("Transforms took %.3f s total", elapsed_time)
        logger.debug("breakdown:")
        for duration, name in measured:
            logger.debug("  %.3f s - %s", duration, name)
        logger.debug("end transforms ---")

        return batch


class MapAndCollate:
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __call__(self, batch_of_indices: List[int]):
        batch = [self.dataset[i] for i in batch_of_indices]
        return self.collate_fn(batch)


def GPUDataLoader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler = None,
    batch_sampler: torch.utils.data.BatchSampler = None,
    num_workers: int = None,
    collate_fn: Callable = None,
    pin_memory: bool = None,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable = None,
    *,
    prefetch_factor: int = None,
    persistent_workers: bool = False,
    gpu_device: int = 0,
    gpu_prefetch_factor: int = 1,
    gpu_num_threads: int = 1,
    gpu_transforms: List[Transform] = None,
):
    """Initialize a DataLoader.

    Args:
        dataset: Dataset to load.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        sampler: Sampler to use for the dataset.
        batch_sampler: Batch sampler to use for the dataset.
        num_workers: Number of workers to use for data loading.
        collate_fn: Function to use for collating batches.
        pin_memory: Whether to pin memory.
        drop_last: Whether to drop the last batch if it's not full.
        timeout: Timeout for the data loader.
        worker_init_fn: Function to use for initializing workers.
        prefetch_factor: Number of batches to prefetch.
        persistent_workers: Whether to use persistent workers.
        gpu_device: GPU device to use.
        gpu_prefetch_factor: Number of batches to prefetch on the GPU.
        gpu_num_threads: Number of threads to use for prefetching on the GPU.
        gpu_transforms: Transforms to apply on the GPU.
    """

    if not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
        raise ValueError("Dataset must have __getitem__ and __len__ methods")

    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    node = tn.SamplerWrapper(sampler)

    node = tn.Batcher(node, batch_size=batch_size, drop_last=drop_last)

    map_and_collate = MapAndCollate(dataset, collate_fn or default_collate)

    node = tn.ParallelMapper(
        node,
        map_fn=map_and_collate,
        num_workers=num_workers,
        method="process",
        multiprocessing_context="spawn",
        in_order=True,
    )

    if pin_memory:
        node = tn.PinMemory(node, pin_memory_device=f"cuda:{gpu_device}")

    if prefetch_factor is not None:
        node = tn.Prefetcher(node, prefetch_factor=prefetch_factor)

    node = tn.ParallelMapper(
        node,
        map_fn=ApplyGPUTransforms(gpu_transforms, gpu_device),
        num_workers=gpu_num_threads,
        method="thread",
        in_order=True,
    )

    if gpu_prefetch_factor is not None:
        node = tn.Prefetcher(node, prefetch_factor=gpu_prefetch_factor)

    loader = tn.Loader(node)

    return loader
