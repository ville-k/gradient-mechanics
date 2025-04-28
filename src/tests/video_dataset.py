import logging
import os

import torch

from gradient_mechanics.data import episodes, video_indexing
from gradient_mechanics.data import video_demuxing
from gradient_mechanics.data import video_transforms


logger = logging.getLogger(__name__)


class VideoDataset(torch.utils.data.Dataset):
    _INDEX_EXTENSION = ".index.json"

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
        self._sample_index_generator = None
        self._indexing_demuxer = None
        self._is_initialized = False

    def _lazy_init(self):
        """Initialize the VideoDataset lazily."""
        if self._is_initialized:
            return

        index_file_path = self.video_file_path + self._INDEX_EXTENSION
        if not os.path.exists(index_file_path):
            logger.info(
                f"Index file {index_file_path} does not exist, generating index"
            )
            video_index = video_indexing.VideoIndex.generate(self.video_file_path)
            video_index.save(index_file_path)
        else:
            logger.info(f"Index file {index_file_path} exists, loading index")
            video_index = video_indexing.VideoIndex.load(index_file_path)

        self._indexing_demuxer = video_demuxing.IndexingDemuxer(
            self.video_file_path, video_index
        )
        self._sample_index_generator = episodes.EpisodeGenerator(
            sample_count=len(self._indexing_demuxer),
            episode_length=self._episode_length,
            episode_stride=self._episode_stride,
        )
        self._is_initialized = True

    def __len__(self):
        """Return the total number of episodes."""
        self._lazy_init()
        return self._sample_index_generator.episode_count()

    def __getitem__(self, idx: int) -> video_transforms.PacketBuffers:
        """Return the episode at the given index."""
        self._lazy_init()

        indices = self._sample_index_generator.episode_samples_indices(idx)
        episode_buffers = self._indexing_demuxer.packet_buffers_for_frame_indices(
            indices
        )

        return episode_buffers
