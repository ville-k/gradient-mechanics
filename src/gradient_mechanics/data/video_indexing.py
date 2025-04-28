import dataclasses
import json
import pathlib

import av
import av.container
from av.video.frame import PictureType


@dataclasses.dataclass(frozen=True)
class FrameInfo:
    frame_number: int
    timestamp: float
    packet_index: int
    picture_type: str = None


@dataclasses.dataclass(frozen=True)
class VideoIndex:
    video_file_path: str
    number_of_frames: int
    duration_in_seconds: float
    frame_index_to_info: dict[int, FrameInfo] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return self.number_of_frames

    def frame_at_index(self, frame_index: int) -> FrameInfo:
        if frame_index < 0 or frame_index >= self.number_of_frames:
            raise IndexError(f"Frame index {frame_index} out of range")

        frame_info = self.frame_index_to_info[frame_index]
        return frame_info

    def packet_indices_for_frames(self, frame_indices: list[int]) -> list[int]:
        # Use dict as an ordered set to form the union of the packet indices
        packet_indices: dict[int, None] = dict()
        for frame_index in frame_indices:
            indices = self.packet_indices_for_frame(frame_index)
            for index in indices:
                packet_indices[index] = None

        return list(packet_indices.keys())

    def packet_indices_for_frame(self, frame_index: int) -> list[int]:
        if frame_index < 0 or frame_index >= self.number_of_frames:
            raise IndexError(f"Frame index {frame_index} out of range")

        frame_info = self.frame_index_to_info[frame_index]
        if frame_info.picture_type == "I":
            return [frame_index]

        if frame_info.picture_type == "P":
            preceding_dependencies = self._find_preceding_dependencies(frame_index)
            return preceding_dependencies + [frame_index]

        if frame_info.picture_type == "B":
            preceding_dependencies = self._find_preceding_dependencies(frame_index)
            following_index = self._find_following_dependency(frame_index)
            return preceding_dependencies + [frame_index, following_index]

        raise ValueError(f"Unknown picture type {frame_info.picture_type}")

    def _find_preceding_dependencies(self, frame_index: int) -> list[int]:
        reference_frames = []
        for i in range(frame_index - 1, -1, -1):
            frame_info = self.frame_index_to_info[i]
            if frame_info.picture_type == "I":
                reference_frames.insert(0, i)
                break
            if frame_info.picture_type == "P":
                reference_frames.insert(0, i)

        return reference_frames

    def _find_following_dependency(self, frame_index: int) -> int:
        for i in range(frame_index + 1, self.number_of_frames):
            frame_info = self.frame_index_to_info[i]
            if frame_info.picture_type in ["P", "I"]:
                return i
        raise ValueError(f"No following P or I frame found for frame {frame_index}")

    def save(self, file_path: pathlib.Path):
        """Save the video index to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, file_path: pathlib.Path) -> "VideoIndex":
        """Load the video index from a JSON file."""
        with open(file_path, "r") as f:
            content = json.load(f)
            video_file_path = content["video_file_path"]
            number_of_frames = content["number_of_frames"]
            duration_seconds = content["duration_in_seconds"]
            frame_index_to_info = {
                int(frame_index): FrameInfo(**frame_info)
                for frame_index, frame_info in content["frame_index_to_info"].items()
            }
            return VideoIndex(
                video_file_path=video_file_path,
                number_of_frames=number_of_frames,
                duration_in_seconds=duration_seconds,
                frame_index_to_info=frame_index_to_info,
            )

    @classmethod
    def generate(cls, video_file_path: pathlib.Path) -> "VideoIndex":
        container: av.container.InputContainer = av.open(str(video_file_path))
        number_of_frames = 0
        duration_time_base = 0
        time_base = None

        frame_index_to_info: dict[int, FrameInfo] = dict()

        for packet in container.demux(video=0):
            duration_time_base += packet.duration
            time_base = packet.time_base

            frames = packet.decode()
            # frames are queued within the decoder and None indicates that the queue is empty
            for frame in frames:
                assert frame_index_to_info.get(number_of_frames) is None
                frame_index_to_info[number_of_frames] = FrameInfo(
                    frame_number=number_of_frames,
                    timestamp=float(frame.pts * frame.time_base),
                    packet_index=number_of_frames,
                    picture_type=PictureType(frame.pict_type).name,
                )

                number_of_frames += 1

        duration_in_seconds = float((duration_time_base * time_base))
        assert number_of_frames == len(frame_index_to_info)
        return VideoIndex(
            video_file_path=str(video_file_path),
            number_of_frames=len(frame_index_to_info),
            duration_in_seconds=duration_in_seconds,
            frame_index_to_info=frame_index_to_info,
        )
