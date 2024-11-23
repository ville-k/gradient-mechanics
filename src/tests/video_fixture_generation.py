import av
import numpy as np


def value_to_rgb(value: int) -> tuple[int, int, int]:
    """
    Convert a value (0-999) to RGB components where:
    - R stores hundreds (0-9) * 25
    - G stores tens (0-9) * 25
    - B stores ones (0-9) * 25

    Args:
        value: Integer between 0 and 999

    Returns:
        Tuple of (r, g, b) values as uint8

    Example:
        432 -> (100, 75, 50)  # 4*25=100 for hundreds, 3*25=75 for tens, 2*25=50 for ones
    """
    if not isinstance(value, int):
        raise ValueError(f"Value must be an integer, got {type(value)}")
    if value < 0 or value > 999:
        raise ValueError(f"Value must be between 0 and 999, got {value}")

    hundreds = value // 100  # 0-9
    tens = (value % 100) // 10  # 0-9
    ones = value % 10  # 0-9

    # Scale to make values more distinct (multiply by 25 to spread across 0-225 range)
    r = hundreds * 25
    g = tens * 25
    b = ones * 25

    return (r, g, b)


def rgb_to_value(rgb: tuple[int, int, int]) -> int:
    """
    Convert RGB components back to original value where:
    - R represents hundreds
    - G represents tens
    - B represents ones

    Args:
        rgb: Tuple of (r, g, b) values as uint8

    Returns:
        Integer value between 0 and 999

    Example:
        (100, 75, 50) -> 432  # 100/25=4 hundreds, 75/25=3 tens, 50/25=2 ones
    """
    if len(rgb) != 3:
        raise ValueError(f"RGB must have 3 components, got {len(rgb)}")

    # Convert back from scaled values
    hundreds = round(rgb[0] / 25)  # 0-9
    tens = round(rgb[1] / 25)  # 0-9
    ones = round(rgb[2] / 25)  # 0-9

    value = hundreds * 100 + tens * 10 + ones
    return value


def create_frame_with_value(value, width=32, height=32):
    """
    Creates a frame with RGB values.
    """
    rgb_values = value_to_rgb(value)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = rgb_values[0]
    frame[:, :, 1] = rgb_values[1]
    frame[:, :, 2] = rgb_values[2]

    return frame


def read_frame_value(frame_array):
    """
    Reads the value from a decoded RGB frame.
    """
    if len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
        raise ValueError(
            f"Frame must be 3D with 3 channels, got shape {frame_array.shape}"
        )

    median_color = np.median(frame_array, axis=(0, 1))
    value = rgb_to_value(median_color)
    return value


def create_test_video(
    output_filename,
    frame_width=320,
    frame_height=240,
    frame_count=30,
    fps=30,
    frame_pattern="I",
) -> str:
    """
    Creates a test video with RGB frames.
    """
    container = av.open(output_filename, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frame_width
    stream.height = frame_height
    stream.pix_fmt = "yuv420p"
    stream.options = {
        "crf": "18",  # High quality (lower value = higher quality)
        "preset": "veryslow",  # Better compression
        "qmin": "1",  # Minimum quantizer scale
        "qmax": "1",  # Maximum quantizer scale (same as qmin for consistent quality)
        "sc_threshold": "0",  # Disable scene change detection
        "refs": "1",  # Minimum reference frames
    }

    if frame_pattern == "I":
        # Force all frames to be I-frames
        stream.options.update(
            {
                "g": "1",  # GOP size of 1 (every frame is an I-frame)
                "keyint_min": "1",  # Minimum GOP size
                "b-frames": "0",  # Disable B-frames
                "force-key-frames": "expr:1",  # Force keyframe every frame
            }
        )

    elif frame_pattern == "IP":
        # Only I and P frames
        stream.options.update(
            {
                "g": "2",  # GOP size (I-frame every 30 frames)
                "keyint_min": "2",  # Minimum GOP size
                "b-frames": "0",  # Disable B-frames
            }
        )
    else:
        raise ValueError(f"Invalid frame pattern: {frame_pattern}")

    for value in range(frame_count):
        frame_data = create_frame_with_value(value)
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    packets = stream.encode()
    for packet in packets:
        container.mux(packet)
    container.close()

    return output_filename


def analyze_frame_pattern(filename):
    """
    Analyze and print the frame pattern, now handling RGB frames.
    """
    container = av.open(filename)
    frame_types = []
    values = []

    # Decode frames to get values
    for frame in container.decode(video=0):
        frame_data = frame.to_ndarray(format="rgb24")
        frame_types.append(frame.pict_type.name)
        value = read_frame_value(frame_data)
        values.append(value)

    print("Frame pattern analysis:")
    print(f"Total frames: {len(frame_types)}")
    print(f"I-frames: {frame_types.count('I')}")
    print(f"B-frames: {frame_types.count('B')}")
    print(f"P-frames: {frame_types.count('P')}")
    print(f"Frame values: {values}")
    print(f"Fame types: {frame_types}")


if __name__ == "__main__":
    output_filename = "test_rgb.mp4"
    create_test_video(output_filename, fps=30, frame_pattern="IP")
    analyze_frame_pattern(output_filename)
    print("Done")
