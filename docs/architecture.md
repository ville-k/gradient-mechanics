# Architecture

Gradient Mechanics wraps several NVIDIA GPU libraries and exposes a
PyTorch‑friendly interface for building image and video pipelines that execute
entirely on the GPU.

## Libraries Used

- [NVImageCodec](https://docs.nvidia.com/cuda/nvimagecodec/index.html) for
  decoding JPEG image batches.
- [PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html)
  for efficient GPU video decoding.
- [CV-CUDA](https://cvcuda.github.io/CV-CUDA/index.html) for performing common
  image transforms such as resizing and cropping.

These components are wrapped by the Python modules in `gradient_mechanics` to
provide `torch` and `torchdata` based data loaders that run entirely on the GPU.

## Overview

All source code lives under `src/gradient_mechanics`. The `data` package hosts
the utilities for data loading and GPU preprocessing. A typical workflow is to
create a dataset that yields either encoded images or video frame indices and
wrap it with one of the provided GPU data loaders. The loader performs
host‑side batching, launches GPU transforms on worker threads and returns fully
processed tensors ready for a model.

### Transforms

`Transform` is the base class for GPU transforms. Each transform registers the
input types it can handle and the base class recursively walks nested data
structures, applying the transform where appropriate. Notable built‑in
transforms include:

- `JPEGDecode` – uses NVImageCodec to decode JPEG batches directly on the GPU.
- `Resize`, `Crop` and `ToTensor` – wrappers around CV‑CUDA operations.
- `AsyncH2D` – asynchronously copies tensors from host to device memory.

Transforms are composable and can be passed to the data loaders in a list.

### GPU iteration and DataLoaders

`GPUIterator` wraps an existing iterator and executes a sequence of GPU
transforms using a pool of threads. It also prefetches a configurable number of
batches so that GPU work overlaps with data loading. `GPUDataLoader` extends the
standard `torch.utils.data.DataLoader` and produces a `GPUIterator` for its
output, enabling asynchronous GPU preprocessing after batches are collated.

The `torchdata_loading` module exposes the same concept using `torchdata` nodes
which can be chained together for streaming pipelines.

### Video pipeline

Video decoding is handled through a set of helpers that minimise unnecessary
work:

- `VideoIndex` stores metadata about each frame and its packet dependencies.
- `IndexingDemuxer` fetches only the packets required for a given set of frame
  indices.
- `DecodeVideo` turns the packets into tensors with PyNvVideoCodec and CV‑CUDA.

Together these classes enable targeted decoding of just the frames used in each
training episode.

### Episodes

`EpisodeGenerator` provides utilities for breaking a long video into fixed
length episodes with a configurable stride. The tests and benchmarks use it to
generate small clips for model input.

## Putting it together

1. Create a dataset that yields encoded images or frame indices.
2. Instantiate a `GPUDataLoader` (or the `torchdata` variant) with a list of GPU
   transforms.
3. Iterate over the loader to receive fully processed tensors on the GPU.

The design keeps the data on the device once decoding occurs, avoiding costly
host/device transfers and making it straightforward to build high‑throughput
training pipelines.
