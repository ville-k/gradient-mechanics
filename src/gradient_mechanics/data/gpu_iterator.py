import logging
import time


import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import List

from gradient_mechanics.data.transforms import Transform

logger = logging.getLogger(__name__)


class GPUIterator:
    def __init__(
        self,
        iterator,
        gpu_device: int,
        gpu_prefetch_factor: int = 1,
        num_threads: int = 1,
        transforms: List[Transform] = None,
    ):
        """Initialize a GPUIterator.

        Arguments:
            iterator: Iterator to wrap.
            gpu_device: GPU device to use.
            gpu_prefetch_factor: Number of batches to prefetch on the GPU.
            num_threads: Number of threads to use for prefetching on the GPU.
            transforms: Transforms to apply on the GPU.
        """
        self.iterator = iter(iterator)
        self.gpu_device = gpu_device
        self.gpu_prefetch_factor = gpu_prefetch_factor
        self.num_threads = num_threads
        self.transforms = transforms or []

        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.futures = deque()
        self.stop_iteration = False

        for _ in range(self.gpu_prefetch_factor):
            self._submit_next()

    def __del__(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)

    def __iter__(self):
        """Return an iterator over the GPUIterator."""
        return self

    def __next__(self):
        """Return the next item from the iterator."""
        while True:
            stop_iteration = False
            with self.lock:
                if self.stop_iteration:
                    stop_iteration = self.stop_iteration

            if not self.futures:
                if stop_iteration:
                    raise StopIteration
                else:
                    self.submit_next()

            future = self.futures.popleft()
            try:
                result = future.result()
                self._submit_next()
                return result
            except StopIteration:
                with self.lock:
                    self.stop_iteration = True
                if not self.futures:
                    raise StopIteration
            except Exception as e:
                raise e

    def _submit_next(self):
        """Submit the next item to the executor."""
        with self.lock:
            if not self.stop_iteration:
                future = self.executor.submit(self._fetch_next_item)
                self.futures.append(future)

    def _fetch_next_item(self):
        """Fetch the next item from the iterator."""
        try:
            with self.lock:
                item = next(self.iterator)

            if item is not None:
                measured = []
                logger.debug("start transforms ---")
                started_transforms_at = time.perf_counter()
                for transform in self.transforms:
                    started_at = time.perf_counter()
                    item = transform(item)
                    elapsed_time = time.perf_counter() - started_at
                    measured.append((elapsed_time, transform.__class__.__name__))
                elapsed_time = time.perf_counter() - started_transforms_at

                logger.debug("Transforms took %.3f s total", elapsed_time)
                logger.debug("breakdown:")
                for duration, name in measured:
                    logger.debug("  %.3f s - %s", duration, name)
                logger.debug("end transforms ---")

            return item
        except StopIteration:
            with self.lock:
                self.stop_iteration = True
            raise StopIteration
