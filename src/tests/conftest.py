import pathlib

import av
import pytest

import video_fixture_generation


@pytest.fixture
def data_root():
    root = pathlib.Path(__file__).parent.parent.parent / "data"
    return root.absolute()


@pytest.fixture
def short_mp4_file_path(data_root):
    return data_root / "humming_bird_1s.mp4"


@pytest.fixture
def video_all_keyframes_path(data_root):
    return data_root / "humming_bird_1s_keyframes_only.mp4"


@pytest.fixture
def video_some_keyframes_path(data_root):
    """IBBBPBBBPBBBPBBBPBBBPBBBPBBBPP"""
    return data_root / "humming_bird_1s.mp4"


@pytest.fixture
def thirty_jpegs_dir_path(short_mp4_file_path, tmp_path):
    container = av.open(short_mp4_file_path)

    for index, frame in enumerate(container.decode(video=0)):
        frame.to_image().save(tmp_path / f"short_{index:02}.jpeg")

    return tmp_path


@pytest.fixture
def single_jpeg_path(thirty_jpegs_dir_path):
    return thirty_jpegs_dir_path / "short_00.jpeg"


@pytest.fixture(scope="session")
def thirty_key_frame_video(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    video_path = data_dir / "thirty_key_frames.mp4"
    video_fixture_generation.create_test_video(str(video_path), frame_pattern="I")
    return video_path


@pytest.fixture(scope="session")
def ten_key_frame_video(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    video_path = data_dir / "ten_key_frames.mp4"
    video_fixture_generation.create_test_video(
        str(video_path),
        frame_width=120,
        frame_height=60,
        frame_count=10,
        frame_pattern="I",
    )
    return video_path


@pytest.fixture(scope="session")
def ten_key_and_predicted_frame_video(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    video_path = data_dir / "ten_key_and_predicted_frames.mp4"
    video_fixture_generation.create_test_video(
        str(video_path),
        frame_width=120,
        frame_height=60,
        frame_count=10,
        frame_pattern="IP",
    )
    return video_path
