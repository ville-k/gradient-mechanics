import av
import numpy as np
import pytest


from video_fixture_generation import (
    value_to_rgb,
    rgb_to_value,
    create_test_video,
    read_frame_value,
    create_frame_with_value,
)


def test_value_to_rgb_value_range_roundtrip_succceeds():
    for value in range(0, 999):
        rgb = value_to_rgb(value)
        decoded = rgb_to_value(rgb)
        assert decoded == value, f"Failed to roundtrip value {value}"


def test_value_to_rgb_with_invalid_inputs():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError, match="Value must be between 0 and 999, got -1"):
        value_to_rgb(-1)

    with pytest.raises(ValueError, match="Value must be between 0 and 999, got 1000"):
        value_to_rgb(1000)


def test_create_frame_with_value_with_invalid_inputs():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError, match="Value must be between 0 and 999, got -1"):
        create_frame_with_value(-1)

    with pytest.raises(ValueError, match="Value must be between 0 and 999, got 1000"):
        create_frame_with_value(1000)


@pytest.mark.parametrize("width, height", [(32, 32), (32, 64), (64, 32)])
def test_create_frame_with_value_frame_shape(width, height):
    """Test that the frame has the correct shape."""
    frame = create_frame_with_value(0, width=width, height=height)
    assert frame.shape == (height, width, 3), "Unexpected frame shape"


def test_read_frame_value_with_valid_inputs():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    value = read_frame_value(frame)
    assert value == 0, f"Unexpected frame value {value}"


def test_read_frame_value_with_invalid_inputs():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError, match="Frame must be 3D with 3 channels"):
        read_frame_value(np.zeros((32, 32, 1), dtype=np.uint8))

    with pytest.raises(ValueError, match="Frame must be 3D with 3 channels"):
        read_frame_value(np.zeros((32, 32, 4), dtype=np.uint8))


def test_create_frame_with_value_roundtrip():
    """Test that the value can be read back from the frame."""
    for value in range(0, 999):
        frame = create_frame_with_value(value)
        decoded_value = read_frame_value(frame)
        assert decoded_value == value, f"Failed to roundtrip value {value}"


def test_create_test_video_with_key_frames_only(tmp_path):
    """Test that key frames are properly generated."""
    video_path = create_test_video(
        tmp_path / "key_frames_only.mp4",
        frame_count=999,
        frame_width=120,
        frame_height=60,
        frame_pattern="I",
    )
    assert video_path.exists(), "Video file not created"
    assert video_path.name == "key_frames_only.mp4", "Unexpected video file name"

    container = av.open(str(video_path))
    for frame_index, frame in enumerate(container.decode(video=0)):
        assert frame.pict_type == "I", "Not all frames are key frames"
        frame_array = frame.to_ndarray(format="rgb24")
        value = read_frame_value(frame_array)
        assert (
            value == frame_index
        ), f"Unexpected frame value {value} at index {frame_index}"


def test_create_test_video_with_key_and_predicted_frames(tmp_path):
    """Test that key frames are properly generated."""
    video_path = create_test_video(
        tmp_path / "key_and_predicted_frames.mp4",
        frame_count=999,
        frame_width=120,
        frame_height=60,
        frame_pattern="IP",
    )
    assert video_path.exists(), "Video file not created"
    assert (
        video_path.name == "key_and_predicted_frames.mp4"
    ), "Unexpected video file name"

    container = av.open(str(video_path))
    for frame_index, frame in enumerate(container.decode(video=0)):
        expected_frame_type = "I" if frame_index % 2 == 0 else "P"
        assert (
            frame.pict_type == expected_frame_type
        ), f"Unexpected frame type {frame.pict_type} at index {frame_index}"
        frame_array = frame.to_ndarray(format="rgb24")
        value = read_frame_value(frame_array)
        assert (
            value == frame_index
        ), f"Unexpected frame value {value} at index {frame_index}"
