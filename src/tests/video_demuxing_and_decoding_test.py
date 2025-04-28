import pytest
import torch
from torch.utils.data import dataloader
from gradient_mechanics.data import video_demuxing, video_indexing
from gradient_mechanics.data import video_transforms

import video_fixture_generation


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_collates_packet_buffers(batch_size, ten_key_frame_video):
    index = video_indexing.VideoIndex.generate(ten_key_frame_video)
    demuxer = video_demuxing.IndexingDemuxer(str(ten_key_frame_video), index)

    samples = []
    for sample_index in range(batch_size):
        sample: video_transforms.PacketBuffers = (
            demuxer.packet_buffers_for_frame_indices([sample_index])
        )
        samples.append(sample)

    collated: video_transforms.PacketBuffersBatch = dataloader.default_collate(samples)
    assert len(collated) == 1
    assert len(collated.samples) == batch_size


@pytest.mark.parametrize(
    "video_path_fixture", ["ten_key_frame_video", "ten_key_and_predicted_frame_video"]
)
def test_returns_all_frames_in_order_ony_by_one(video_path_fixture, request):
    video_path = request.getfixturevalue(video_path_fixture)
    index = video_indexing.VideoIndex.generate(video_path)
    demuxer = video_demuxing.IndexingDemuxer(str(video_path), index)
    assert len(demuxer) == 10

    decoder = video_transforms.DecodeVideo(device_id=0)
    for frame_id in range(len(demuxer)):
        packet_buffers = demuxer.packet_buffers_for_frame_indices([frame_id])
        collated = dataloader.default_collate([packet_buffers])
        decoded = decoder.decode_batch(collated)
        assert len(decoded) == 1
        for frame in decoded:
            array = torch.as_tensor(frame.cuda(), dtype=torch.uint8).cpu().numpy()
            assert array.shape == (1, 60, 120, 3)
            value = video_fixture_generation.read_frame_value(array[0])
            assert value == frame_id


@pytest.mark.parametrize(
    "video_path_fixture", ["ten_key_frame_video", "ten_key_and_predicted_frame_video"]
)
def test_returns_all_frames_in_order_ony_by_one_batch_size_two(
    video_path_fixture, request
):
    video_path = request.getfixturevalue(video_path_fixture)
    index = video_indexing.VideoIndex.generate(video_path)
    demuxer = video_demuxing.IndexingDemuxer(str(video_path), index)
    assert len(demuxer) == 10

    decoder = video_transforms.DecodeVideo(device_id=0)
    for frame_id in range(len(demuxer)):
        sample_1 = demuxer.packet_buffers_for_frame_indices([frame_id])
        sample_2 = demuxer.packet_buffers_for_frame_indices([frame_id])

        collated = dataloader.default_collate([sample_1, sample_2])
        decoded: list[torch.Tensor] = decoder.decode_batch(collated)
        assert len(decoded) == 2

        for frame in decoded:
            array = torch.as_tensor(frame.cuda(), dtype=torch.uint8).cpu().numpy()
            assert array.shape == (1, 60, 120, 3)
            value = video_fixture_generation.read_frame_value(array[0])
            assert value == frame_id


@pytest.mark.parametrize(
    "video_path_fixture", ["ten_key_frame_video", "ten_key_and_predicted_frame_video"]
)
def test_returns_all_frames_in_order_all_at_once(video_path_fixture, request):
    video_path = request.getfixturevalue(video_path_fixture)
    index = video_indexing.VideoIndex.generate(video_path)
    demuxer = video_demuxing.IndexingDemuxer(str(video_path), index)
    assert len(demuxer) == 10
    indices = list(range(len(demuxer)))
    packet_buffers = demuxer.packet_buffers_for_frame_indices(indices)
    decoder = video_transforms.DecodeVideo(device_id=0)
    collated = dataloader.default_collate([packet_buffers])
    decoded = decoder.decode_batch(collated)
    assert len(decoded) == 1
    assert decoded[0].shape[0] == len(demuxer)
    array = torch.as_tensor(decoded[0].cuda(), dtype=torch.uint8).cpu().numpy()
    assert array.shape == (10, 60, 120, 3)
    for frame_offset in range(len(demuxer)):
        value = video_fixture_generation.read_frame_value(array[frame_offset])
        assert value == frame_offset


@pytest.mark.parametrize(
    "video_path_fixture", ["ten_key_frame_video", "ten_key_and_predicted_frame_video"]
)
def test_returns_all_frames_in_order_two_by_two_overlapping(
    video_path_fixture, request
):
    video_path = request.getfixturevalue(video_path_fixture)
    index = video_indexing.VideoIndex.generate(video_path)
    demuxer = video_demuxing.IndexingDemuxer(str(video_path), index)
    assert len(demuxer) == 10
    decoder = video_transforms.DecodeVideo(device_id=0)
    for frame_id in range(len(demuxer) - 1):
        packet_buffers = demuxer.packet_buffers_for_frame_indices(
            [frame_id, frame_id + 1]
        )
        assert len(packet_buffers.target_frames) == 2, packet_buffers.target_frames

        collated = dataloader.default_collate([packet_buffers])
        decoded = decoder.decode_batch(collated)
        assert len(decoded) == 1
        array = torch.as_tensor(decoded[0].cuda(), dtype=torch.uint8).cpu().numpy()

        assert array.shape == (2, 60, 120, 3)
        for frame_offset in range(2):
            value = video_fixture_generation.read_frame_value(array[frame_offset])
            assert value == (frame_id + frame_offset)


@pytest.mark.parametrize(
    "video_path_fixture", ["ten_key_frame_video", "ten_key_and_predicted_frame_video"]
)
def test_returns_all_frames_in_order_two_by_two_non_overlapping(
    video_path_fixture, request
):
    video_path = request.getfixturevalue(video_path_fixture)
    index = video_indexing.VideoIndex.generate(video_path)
    demuxer = video_demuxing.IndexingDemuxer(str(video_path), index)
    assert len(demuxer) == 10
    decoder = video_transforms.DecodeVideo(device_id=0)

    sequence_length = 2
    for frame_id in range(0, len(demuxer), sequence_length):
        indices = [frame_id + offset for offset in range(sequence_length)]

        packet_buffers = demuxer.packet_buffers_for_frame_indices(indices)
        assert len(packet_buffers.target_frames) == sequence_length, (
            packet_buffers.target_frames
        )

        collated = dataloader.default_collate([packet_buffers])
        decoded = decoder.decode_batch(collated)
        assert len(decoded) == 1

        array = torch.as_tensor(decoded[0].cuda(), dtype=torch.uint8).cpu().numpy()
        assert array.shape == (sequence_length, 60, 120, 3)

        for frame_offset in range(sequence_length):
            value = video_fixture_generation.read_frame_value(array[frame_offset])
            expected_frame_id = frame_id + frame_offset
            assert value == expected_frame_id


@pytest.mark.parametrize(
    "video_path_fixture", ["ten_key_frame_video", "ten_key_and_predicted_frame_video"]
)
def test_returns_all_frames_in_order_two_by_two_non_overlapping_skip_two(
    video_path_fixture, request
):
    video_path = request.getfixturevalue(video_path_fixture)
    index = video_indexing.VideoIndex.generate(video_path)
    demuxer = video_demuxing.IndexingDemuxer(str(video_path), index)
    assert len(demuxer) == 10
    decoder = video_transforms.DecodeVideo(device_id=0)

    sequence_length = 1
    skipped_frames = 2
    for frame_id in range(0, len(demuxer), (sequence_length + skipped_frames)):
        indices = [frame_id + offset for offset in range(sequence_length)]

        packet_buffers = demuxer.packet_buffers_for_frame_indices(indices)
        assert len(packet_buffers.target_frames) == sequence_length, (
            packet_buffers.target_frames
        )

        collated = dataloader.default_collate([packet_buffers])
        decoded = decoder.decode_batch(collated)
        assert len(decoded) == 1

        array = torch.as_tensor(decoded[0].cuda(), dtype=torch.uint8).cpu().numpy()
        assert array.shape == (sequence_length, 60, 120, 3)

        for frame_offset in range(sequence_length):
            value = video_fixture_generation.read_frame_value(array[frame_offset])
            expected_frame_id = frame_id + frame_offset
            assert value == expected_frame_id
