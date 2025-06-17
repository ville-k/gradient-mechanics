# Gradient Mechanics

Gradient Mechanics provides utilities for GPU accelerated loading and
preprocessing of images and videos. It combines NVIDIA's NVImageCodec,
PyNvVideoCodec and CV-CUDA libraries so that decoding and transforms happen
directly on the GPU. See the documentation in the `docs/` directory for more
details on how the project is structured and how to get started.

## Quickstart

Create and sync the development environment using
[uv](https://docs.astral.sh/uv/):

```sh
uv venv
uv sync
```

Run the tests to make sure everything is set up correctly:

```sh
uv run pytest
```

## Benchmarks

### HuggingFace Image Dataset

```sh
uv run python src/tests/benchmark_huggingface_images.py
```

### Video Dataset

```sh
uv run python src/tests/benchmark_video_dataset.py data/humming_bird_1s.mp4
```

## Transcoding

### Keyframes Only

```sh
ffmpeg -i data/humming_bird_1s.mp4 -c:v libx264 -x264-params keyint=1:scenecut=0 -crf 18 data/humming_bird_1s_keyframes_only.mp4
```

### Inspect Encoding

```sh
ffprobe -loglevel error -select_streams v:0 -show_entries frame=pict_type -of default=noprint_wrappers=1 data/humming_bird_1s_keyframes_only.mp4
```

Terminology:

* I-frame is a key frame. Decoding a keyframe is not dependent on other frames.
* P-frame is a predicted frame. It is dependent on an preceding frame.
* B-frame is a bi-directional predicted frame. It depends on a preceding and a succeeding frame.
