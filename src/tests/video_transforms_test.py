import cvcuda
import numpy as np
from gradient_mechanics.data import video_indexing, video_transforms
from gradient_mechanics.data import video_demuxing


def test_decode_single_sample(ten_key_frame_video):
    index = video_indexing.VideoIndex.generate(ten_key_frame_video)
    demuxer = video_demuxing.IndexingDemuxer(str(ten_key_frame_video), index)
    packet_buffers: video_transforms.PacketBuffers = (
        demuxer.packet_buffers_for_frame_indices([0])
    )

    decode = video_transforms.DecodeVideo(device_id=0)
    decoded = decode.decode_sample(packet_buffers)
    assert len(decoded) == 1
    assert decoded[0].shape == (1, 60, 120, 3)
    assert type(decoded[0]) is cvcuda.Tensor
    assert decoded[0].dtype == np.uint8


def test_decode_batch(ten_key_frame_video):
    index = video_indexing.VideoIndex.generate(ten_key_frame_video)
    demuxer = video_demuxing.IndexingDemuxer(str(ten_key_frame_video), index)
    packet_buffers: video_transforms.PacketBuffers = (
        demuxer.packet_buffers_for_frame_indices([0])
    )
    collated = video_transforms.PacketBuffersBatch(samples=[packet_buffers])

    decode = video_transforms.DecodeVideo(device_id=0)
    decoded = decode.decode_batch(collated)
    assert len(decoded) == 1
    for frame in decoded:
        assert frame.shape == (1, 60, 120, 3)
        assert type(frame) is cvcuda.Tensor
        assert frame.dtype == np.uint8


def test_decode_transform_interface(ten_key_frame_video):
    index = video_indexing.VideoIndex.generate(ten_key_frame_video)
    demuxer = video_demuxing.IndexingDemuxer(str(ten_key_frame_video), index)
    packet_buffers: video_transforms.PacketBuffers = (
        demuxer.packet_buffers_for_frame_indices([0])
    )
    collated = video_transforms.PacketBuffersBatch(samples=[packet_buffers])

    decode = video_transforms.DecodeVideo(device_id=0)
    decoded = decode(collated)
    assert len(decoded) == 1
    for frame in decoded:
        assert frame.shape == (1, 60, 120, 3)
        assert type(frame) is cvcuda.Tensor
        assert frame.dtype == np.uint8
