import logging
from typing import Callable, List
import torch

from gradient_mechanics.data.gpu_iterator import GPUIterator
from gradient_mechanics.data.transforms import Transform


logger = logging.getLogger(__name__)


class GPUDataLoader:
    def __init__(
        self,
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
        """Initialize a GPUDataLoader.

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
        self.gpu_device = gpu_device
        self.gpu_prefetch_factor = gpu_prefetch_factor
        self.gpu_num_threads = gpu_num_threads
        self.gpu_transforms = gpu_transforms or []

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def __iter__(self):
        """Return an iterator over the data loader."""
        return GPUIterator(
            iterator=iter(self.data_loader),
            gpu_device=self.gpu_device,
            gpu_prefetch_factor=self.gpu_prefetch_factor,
            num_threads=self.gpu_num_threads,
            transforms=self.gpu_transforms,
        )

    def __len__(self):
        """Return the number of batches in the data loader."""
        return len(self.data_loader)
