from gradient_mechanics.data import video_indexing

import pytest


def test_generate_index_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        video_indexing.VideoIndex.generate("data/does_not_exist.mp4")


def test_video_indexing(video_all_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    assert index.duration_in_seconds == 1
    assert index.number_of_frames == 30


def test_video_indexing_on_some_keyframes(video_some_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_some_keyframes_path)
    assert index.duration_in_seconds == 1
    assert len(index) == 30
    assert index.number_of_frames == 30


def test_frame_at_index(video_all_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    frame_info = index.frame_at_index(0)
    assert frame_info.frame_number == 0
    assert frame_info.timestamp == 0.0
    assert frame_info.packet_index == 0
    assert frame_info.picture_type == "I"

    frame_info = index.frame_at_index(1)
    assert frame_info.frame_number == 1
    assert frame_info.timestamp == 0.03333333333333333
    assert frame_info.packet_index == 1
    assert frame_info.picture_type == "I"

    frame_info = index.frame_at_index(29)
    assert frame_info.frame_number == 29
    assert frame_info.timestamp == 0.9666666666666667
    assert frame_info.packet_index == 29
    assert frame_info.picture_type == "I"


def test_frame_at_index_raises_out_of_range(video_all_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    with pytest.raises(IndexError):
        index.frame_at_index(-1)
    with pytest.raises(IndexError):
        index.frame_at_index(30)


def test_all_key_frames(video_all_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    assert all(
        [info.picture_type == "I" for info in index.frame_index_to_info.values()]
    )


def test_all_key_frames_packet_indices_for_frame(video_all_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    for frame_index in range(30):
        packet_indices_for_frame = index.packet_indices_for_frame(frame_index)
        assert packet_indices_for_frame == [frame_index]


def test_all_key_frames_packet_indices_for_frames(video_all_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    for frame_index in range(1, 30):
        frames = [frame_index - 1, frame_index]
        packet_indices_for_frames = index.packet_indices_for_frames(frames)
        assert packet_indices_for_frames == [frame_index - 1, frame_index]


def test_some_key_frames(video_some_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_some_keyframes_path)
    assert len(index.frame_index_to_info) == 30

    info = index.frame_at_index(0)
    assert info.picture_type == "I"
    for frame_index in range(1, 30):
        info = index.frame_at_index(frame_index)
        assert info.picture_type in ["P", "B"]


def test_some_key_frames_packet_indices_for_frame(video_some_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_some_keyframes_path)

    packet_indices_for_frame = index.packet_indices_for_frame(0)
    assert packet_indices_for_frame == [0]

    """IBBBPBBBPBBBPBBBPBBBPBBBPBBBPP"""
    packet_indices_for_frame = index.packet_indices_for_frame(1)
    assert packet_indices_for_frame == [0, 1, 4]

    packet_indices_for_frame = index.packet_indices_for_frame(2)
    assert packet_indices_for_frame == [0, 2, 4]

    packet_indices_for_frame = index.packet_indices_for_frame(3)
    assert packet_indices_for_frame == [0, 3, 4]

    packet_indices_for_frame = index.packet_indices_for_frame(4)
    assert packet_indices_for_frame == [0, 4]

    packet_indices_for_frame = index.packet_indices_for_frame(5)
    assert packet_indices_for_frame == [0, 4, 5, 8]

    packet_indices_for_frame = index.packet_indices_for_frame(6)
    assert packet_indices_for_frame == [0, 4, 6, 8]

    packet_indices_for_frame = index.packet_indices_for_frame(7)
    assert packet_indices_for_frame == [0, 4, 7, 8]

    packet_indices_for_frame = index.packet_indices_for_frame(8)
    assert packet_indices_for_frame == [0, 4, 8]

    packet_indices_for_frame = index.packet_indices_for_frame(9)
    assert packet_indices_for_frame == [0, 4, 8, 9, 12]

    packet_indices_for_frame = index.packet_indices_for_frame(10)
    assert packet_indices_for_frame == [0, 4, 8, 10, 12]

    packet_indices_for_frame = index.packet_indices_for_frame(11)
    assert packet_indices_for_frame == [0, 4, 8, 11, 12]

    packet_indices_for_frame = index.packet_indices_for_frame(12)
    assert packet_indices_for_frame == [0, 4, 8, 12]


def test_some_key_frames_packet_indices_for_frames(video_some_keyframes_path):
    index = video_indexing.VideoIndex.generate(video_some_keyframes_path)

    packet_indices_for_frame = index.packet_indices_for_frames([0, 1])
    assert packet_indices_for_frame == [0, 1, 4]

    packet_indices_for_frames = index.packet_indices_for_frames([3, 9, 12])
    assert packet_indices_for_frames == [0, 3, 4, 8, 9, 12]


def test_save_and_load_video_index(video_all_keyframes_path, tmp_path):
    index = video_indexing.VideoIndex.generate(video_all_keyframes_path)
    index.save(tmp_path / "video_index.json")
    index_loaded = video_indexing.VideoIndex.load(tmp_path / "video_index.json")
    assert index == index_loaded
