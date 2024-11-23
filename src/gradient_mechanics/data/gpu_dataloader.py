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
        num_workers: int = None,
        collate_fn: Callable = None,
        prefetch_factor: int = None,
        pin_memory: bool = None,
        gpu_device: int = 0,
        gpu_prefetch_factor: int = 1,
        gpu_num_threads: int = 1,
        gpu_transforms: List[Transform] = None,
    ):
        """Initialize a GPUDataLoader.

        Arguments:
            dataset: Dataset to load.
            batch_size: Batch size.
            num_workers: Number of workers to use for data loading.
            collate_fn: Function to use for collating batches.
            prefetch_factor: Number of batches to prefetch.
            pin_memory: Whether to pin memory.
            gpu_device: GPU device to use.
            gpu_prefetch_factor: Number of batches to prefetch on the GPU.
            gpu_num_threads: Number of threads to use for prefetching on the GPU.
            gpu_transforms: Transforms to apply on the GPU.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor

        self.gpu_device = gpu_device
        self.gpu_prefetch_factor = gpu_prefetch_factor
        self.gpu_num_threads = gpu_num_threads
        self.gpu_transforms = gpu_transforms or []
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            pin_memory=pin_memory,
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
