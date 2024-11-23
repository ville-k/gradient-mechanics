import logging
import time

import torch

from gradient_mechanics.data import episodes
from gradient_mechanics.data import video_demuxing
from gradient_mechanics.data import video_transforms


logger = logging.getLogger(__name__)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_file_path, episode_length=1, episode_stride=1):
        """
        Initialize the VideoDataset.

        Args:
            video_file_path: Path to the video file.
            episode_length: Number of frames in an episode.
            episode_stride: Stride between episode frames.
        """
        self.video_file_path = video_file_path
        self._episode_length = episode_length
        self._episode_stride = episode_stride
        # Initialized in lazy_init
        self._is_initialized = False
        self._sample_index_generator = None
        self._indexing_demuxer = None
        # Performance metrics
        self._initialization_duration = None
        self._sample_load_times = []

    def _lazy_init(self):
        """Initialize the VideoDataset lazily."""
        if self._is_initialized:
            return

        started_at = time.perf_counter()
        self._indexing_demuxer = video_demuxing.IndexingDemuxer(self.video_file_path)
        self._sample_index_generator = episodes.EpisodeGenerator(
            sample_count=len(self._indexing_demuxer),
            episode_length=self._episode_length,
            episode_stride=self._episode_stride,
        )
        self._is_initialized = True
        ended_at = time.perf_counter()
        self._initialization_duration = ended_at - started_at

    def report_stats(self):
        """Print performance metrics."""
        print(f"Total frames: {len(self)}")
        print(f"Initialization duration: {self._initialization_duration:.4f} seconds")
        print("Sample load times:")
        print(f"Total samples: {len(self._sample_load_times)}")
        if len(self._sample_load_times) == 0:
            return
        print(
            f"Mean sample load time: {sum(self._sample_load_times) / len(self._sample_load_times):.4f}"
        )
        print(f"Max sample load time: {max(self._sample_load_times):.4f}")
        print(f"Min sample load time: {min(self._sample_load_times):.4f}")

    def __len__(self):
        """Return the total number of episodes."""
        self._lazy_init()
        return self._sample_index_generator.episode_count()

    def __getitem__(self, idx: int) -> video_transforms.PacketBuffers:
        """Return the episode at the given index."""
        self._lazy_init()

        started_at = time.perf_counter()
        indices = self._sample_index_generator.episode_samples_indices(idx)
        episode_buffers = self._indexing_demuxer.packet_buffers_for_frame_indices(
            indices
        )
        ended_at = time.perf_counter()
        self._sample_load_times.append(ended_at - started_at)
        return episode_buffers
