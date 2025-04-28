import logging
import os
from typing import List, Optional

import PyNvVideoCodec as nvc
import torch
from gradient_mechanics.data import video_indexing, video_transforms

logger = logging.getLogger(__name__)


class IndexingDemuxer:
    def __init__(
        self, video_file_path: str, video_index: video_indexing.VideoIndex
    ) -> None:
        """
        Initialize the IndexingDemuxer.

        Args:
            video_file_path: Path to the video file to demux.
            video_index: Video index to use for demuxing.
        """
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file {video_file_path} does not exist")
        self._video_file_path = video_file_path
        self._video_index = video_index

    def __len__(self) -> int:
        return len(self._video_index)

    def packet_buffers_for_frame_indices(
        self, frame_indices: List[int]
    ) -> video_transforms.PacketBuffers:
        """
        Fetch packets and dependencies for the given frame indices.

        Args:
            frame_indices: List of frame indices to fetch packets for.

        Returns:
            PacketBuffers object containing the target frames, packet frames, and packets.
        """
        if not frame_indices:
            raise ValueError("frame_indices must not be empty")

        if not all(
            0 <= frame_index < len(self._video_index) for frame_index in frame_indices
        ):
            raise ValueError("frame_indices must be within the range of the video")

        packet_indices = self._video_index.packet_indices_for_frames(frame_indices)
        packets = self._fetch_packets(packet_indices)

        return video_transforms.PacketBuffers(
            target_frames=frame_indices,
            packet_frames=packet_indices,
            packets=packets,
        )

    def _fetch_packets(self, packet_indices: List[int]) -> List[torch.Tensor]:
        """
        Fetch packets for the given packet indices.

        Args:
            packet_indices: List of packet indices to fetch.
        """
        demuxer = nvc.CreateDemuxer(filename=self._video_file_path)
        packet_index_to_packet: dict[int, Optional[torch.ByteTensor]] = {
            packet_index: None for packet_index in packet_indices
        }
        packets_fetched = 0

        for packet_index, packet_iter in enumerate(demuxer):
            # bsl_data is a private member of the PacketData class that points to the underlying data
            # as a workaround we need to copy it to a separate buffer for safe keeping:
            if packet_iter.bsl_data is None or packet_iter.bsl == 0:
                logger.info("Skipping packet with no bsl_data")
                continue
            copied_bsl = video_transforms.buffer_from_packet(packet_iter)

            if packet_index in packet_index_to_packet:
                packet_index_to_packet[packet_index] = copied_bsl
                packets_fetched += 1
                if packets_fetched == len(packet_index_to_packet):
                    break

        assert packets_fetched == len(packet_index_to_packet)
        return [packet for packet in packet_index_to_packet.values()]
