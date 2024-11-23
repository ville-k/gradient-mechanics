from gradient_mechanics.data import video_demuxing

import pytest


def test_raises_on_invalid_video_file_path():
    with pytest.raises(FileNotFoundError):
        video_demuxing.IndexingDemuxer("invalid_file_path.mp4")


def test_returns_correct_length(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    assert len(demuxer) == 30


def test_raises_on_empty_frame_indices(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    with pytest.raises(ValueError, match="frame_indices must not be empty"):
        demuxer.packet_buffers_for_frame_indices([])


def test_raises_on_below_bounds_frame_indices(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    with pytest.raises(
        ValueError, match="frame_indices must be within the range of the video"
    ):
        demuxer.packet_buffers_for_frame_indices([-1])


def test_raises_on_above_bounds_frame_indices(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    above_bounds_frame = len(demuxer) + 1
    with pytest.raises(
        ValueError, match="frame_indices must be within the range of the video"
    ):
        demuxer.packet_buffers_for_frame_indices([above_bounds_frame])


def test_returns_correct_packet_buffers_for_a_keyframe(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    packet_buffers = demuxer.packet_buffers_for_frame_indices([0])
    assert packet_buffers.target_frames == [0]
    assert packet_buffers.packet_frames == [0]
    assert len(packet_buffers.packets) == 1


def test_returns_correct_packet_buffers_for_a_b_frame(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    packet_buffers = demuxer.packet_buffers_for_frame_indices([1])
    assert packet_buffers.target_frames == [1]
    assert packet_buffers.packet_frames == [0, 1, 4]
    assert len(packet_buffers.packets) == 3


def test_returns_correct_packet_buffers_for_a_p_frame(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    packet_buffers = demuxer.packet_buffers_for_frame_indices([4])
    assert packet_buffers.target_frames == [4]
    assert packet_buffers.packet_frames == [0, 4]
    assert len(packet_buffers.packets) == 2


def test_returns_correct_packet_buffers_for_multiple_frames(short_mp4_file_path):
    demuxer = video_demuxing.IndexingDemuxer(str(short_mp4_file_path))
    packet_buffers = demuxer.packet_buffers_for_frame_indices([0, 1, 4])
    assert packet_buffers.target_frames == [0, 1, 4]
    assert packet_buffers.packet_frames == [0, 1, 4]
    assert len(packet_buffers.packets) == 3
