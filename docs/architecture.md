# Architecture

Gradient Mechanics is built around a few core NVIDIA libraries that handle
decoding and image processing on the GPU.

## Libraries Used

- [NVImageCodec](https://docs.nvidia.com/cuda/nvimagecodec/index.html) for
  decoding JPEG image batches.
- [PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html)
  for efficient GPU video decoding.
- [CV-CUDA](https://cvcuda.github.io/CV-CUDA/index.html) for performing common
  image transforms such as resizing and cropping.

These components are wrapped by the Python modules in `gradient_mechanics` to
provide torch and torchdata based data loaders that run entirely on the GPU.
