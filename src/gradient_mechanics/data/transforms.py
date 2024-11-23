import logging
import sys
import time
from typing import Any, List, Optional, Set
import typing
import cvcuda
import numpy as np
from nvidia import nvimgcodec
import torch
from torch.utils.data._utils import collate
from torchvision import io

logger = logging.getLogger(__name__)


class Transform:
    """Base class for image transforms."""

    def __init__(self, device_id, stream=None) -> None:
        """Initialize the transform.

        Arguments:
            device_id: GPU device ID.
            stream: CUDA stream to use. Defaults to 0.
        """
        self.device_id = device_id
        self.device = torch.device("cuda", device_id)
        self.stream = stream or 0
        self.input_types: Set[Any] = set()

    def register_input_type(self, input_type: type) -> None:
        """Register the input type that the transform can process."""
        self.input_types.add(input_type)

    def __call__(self, batch: List[Any]) -> List[cvcuda.Tensor]:
        """Apply the transform to a list of registered data types."""
        try:
            transformed = _apply_recursive(self.transform, batch, self.input_types)
        except Exception as e:
            print(
                f"Exception was raised during the processing of transform '{self.__class__.__name__}'. This could be caused by input data not conforming to what is expected of container types.",
                file=sys.stderr,
            )
            raise e

        return transformed

    def transform(self, batch: List[Any]) -> List[cvcuda.Tensor]:
        """Apply the transform to a list of registered data types. This method should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


def _apply_recursive(
    transform: Transform, data: List[Any], registered_types: Set[type]
) -> List[Any]:
    """Apply the transform to the data recursively. This function is used to apply the transform to collections of registered types within nested data structures."""
    if type(data) in registered_types:
        return transform([data])[0]
    if isinstance(data, list):
        if len(data) > 0 and type(data[0]) in registered_types:
            return transform(data)
        else:
            return [
                _apply_recursive(transform, item, registered_types) for item in data
            ]
    elif isinstance(data, set):
        return {_apply_recursive(transform, item, registered_types) for item in data}
    elif isinstance(data, dict):
        return {
            key: _apply_recursive(transform, value, registered_types)
            for key, value in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(
            *[_apply_recursive(transform, item, registered_types) for item in data]
        )
    elif isinstance(data, tuple):
        return tuple(
            [_apply_recursive(transform, item, registered_types) for item in data]
        )

    return data


class EncodedImage(typing.NamedTuple):
    """A single encoded image."""

    buffer: torch.Tensor


class EncodedImageBatch(typing.NamedTuple):
    """A batch of encoded images."""

    samples: List[EncodedImage]


def collate_encoded_images(
    samples: List[EncodedImage], *, collate_fn_map=None
) -> EncodedImageBatch:
    return EncodedImageBatch(samples=samples)


# Register the collate function for EncodedImageBatch
collate.default_collate_fn_map[EncodedImage] = collate_encoded_images


class JPEGDecode(Transform):
    def __init__(self, max_num_cpu_threads=3, **kwargs) -> None:
        """Decode a batch of JPEG images.

        Uses https://docs.nvidia.com/cuda/nvimagecodec/index.html for decoding with the
        Hybrid CPU-GPU backend.

        Arguments:
            max_num_cpu_threads: Maximum number of CPU threads to use for decoding.
        """
        super().__init__(**kwargs)
        self.register_input_type(EncodedImageBatch)

        hybrid_backend = nvimgcodec.Backend(
            backend_kind=nvimgcodec.BackendKind.HYBRID_CPU_GPU,
        )
        self.decoder = nvimgcodec.Decoder(
            device_id=self.device_id,
            max_num_cpu_threads=max_num_cpu_threads,
            backends=[hybrid_backend],
        )
        self.cvcuda_stream = cvcuda.cuda.as_stream(self.stream)
        self.torch_stream = torch.cuda.ExternalStream(self.stream)

    def transform(self, batch: List[EncodedImageBatch]) -> List[cvcuda.Tensor]:
        """Decode a batch of JPEG images.

        Arguments:
            batch: List of EncodedImageBatch objects.

        Returns:
            List of cvcuda.Tensor objects.
        """
        transformed: List[cvcuda.Tensor] = []

        with self.cvcuda_stream:
            with torch.cuda.stream(self.torch_stream):
                for i, item in enumerate(batch):
                    encoded_batch: EncodedImageBatch = item
                    started_at = time.perf_counter()
                    buffers: list[np.ndarray] = [
                        sample.buffer.numpy() for sample in encoded_batch.samples
                    ]
                    to_numpy_duration = time.perf_counter() - started_at

                    started_at = time.perf_counter()
                    images: list[nvimgcodec.Image] = self.decoder.decode(
                        buffers, cuda_stream=self.cvcuda_stream.handle
                    )
                    decode_duration = time.perf_counter() - started_at

                    started_at = time.perf_counter()
                    image_tensors: list[cvcuda.Tensor] = cvcuda.as_tensors(
                        images, "HWC"
                    )
                    image_tensors: list[cvcuda.Tensor] = [
                        tensor for tensor in image_tensors
                    ]

                    to_cvcuda_duration = time.perf_counter() - started_at
                    logger.debug("decode breakdown:")
                    logger.debug("  %s.3f s - to numpy", to_numpy_duration)
                    logger.debug(
                        "  %s.3f s - decode %d images", decode_duration, len(images)
                    )
                    logger.debug(
                        "  %s.3f s - to cvcuda %d image_tensors",
                        to_cvcuda_duration,
                        len(image_tensors),
                    )

                    transformed.append(image_tensors)

        return transformed


class Resize(Transform):
    def __init__(
        self,
        height: int,
        width: int,
        interpolation: Optional[cvcuda.Interp] = None,
        **kwargs,
    ) -> None:
        """Resize the image to the provided dimensions.

        Arguments:
            height: Height of the resized image.
            width: Width of the resized image.
            interpolation: Interpolation method to use. Defaults to cvcuda.Interp.LINEAR.
        """
        super().__init__(**kwargs)
        self.register_input_type(cvcuda.Tensor)
        self.height = height
        self.width = width
        self.interpolation = interpolation or cvcuda.Interp.LINEAR
        self.cvcuda_stream = cvcuda.cuda.as_stream(self.stream)

    def transform(self, batch: List[cvcuda.Tensor]) -> List[cvcuda.Tensor]:
        """Resize the image to the provided dimensions.

        Arguments:
            batch: List of images in HWC format.

        Returns:
            List of resized images in HWC format.
        """

        with self.cvcuda_stream:
            batch_started_at = time.perf_counter()
            image_batch: cvcuda.ImageBatchVarShape = cvcuda.as_images(
                [image_tensor.cuda() for image_tensor in batch]
            )
            batch_duration = time.perf_counter() - batch_started_at

            resize_started_at = time.perf_counter()
            resized: cvcuda.ImageBatchVarShape = cvcuda.resize(
                image_batch,
                [(self.width, self.height)] * len(batch),
                self.interpolation,
                stream=self.cvcuda_stream,
            )
            resize_duration = time.perf_counter() - resize_started_at

            tensor_list_started_at = time.perf_counter()
            tensor_list = [cvcuda.as_tensor(image) for image in resized]

            tensor_list_duration = time.perf_counter() - tensor_list_started_at

            logger.debug("resize breakdown:")
            logger.debug("  %.3f s - batch", batch_duration)
            logger.debug("  %.3f s - resize", resize_duration)
            logger.debug("  %.3f s - tensor_list", tensor_list_duration)

            return tensor_list


class Crop(Transform):
    def __init__(self, x: int, y: int, width: int, height: int, **kwargs) -> None:
        """Crop the image to provided dimensions.

        Arguments:
            x: X-coordinate of the top-left corner of the crop.
            y: Y-coordinate of the top-left corner of the crop.
            width: Width of the crop.
            height: Height of the crop.
        """
        super().__init__(**kwargs)
        self.register_input_type(cvcuda.Tensor)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cvcuda_stream = cvcuda.cuda.as_stream(self.stream)

    def transform(self, batch: List[cvcuda.Tensor]) -> List[cvcuda.Tensor]:
        """Crop the image to provided dimensions.

        Arguments:
            batch: List of image tensors in HWC format.

        Returns:
            List of cropped image tensors in HWC format.
        """
        cropped_batch: List[cvcuda.Tensor] = []

        for image in batch:
            cropped = cvcuda.customcrop(
                image,
                cvcuda.RectI(x=self.x, y=self.y, width=self.width, height=self.height),
                stream=self.cvcuda_stream,
            )
            cropped_batch.append(cropped)

        return cropped_batch


class ToTensor(Transform):
    def __init__(self, **kwargs) -> None:
        """Convert list of `cvcuda.Tensor`s to `torch.Tensor`s and stack them."""
        super().__init__(**kwargs)
        self.register_input_type(cvcuda.Tensor)
        self.cvcuda_stream: cvcuda.Stream = cvcuda.cuda.as_stream(self.stream)
        self.torch_stream = torch.cuda.ExternalStream(self.stream)

    def transform(self, batch: List[cvcuda.Tensor]) -> torch.Tensor:
        """Convert list of `cvcuda.Tensor`s to `torch.Tensor`s and stack them."""
        with torch.cuda.stream(self.torch_stream):
            with self.cvcuda_stream:
                stacked = cvcuda.stack(batch, stream=self.cvcuda_stream)

                floated = cvcuda.convertto(
                    stacked, cvcuda.Type.F32, stream=self.cvcuda_stream
                )
                permuted = cvcuda.reformat(
                    floated, cvcuda.TensorLayout.NCHW, stream=self.cvcuda_stream
                )

                cuda_tensor = permuted.cuda()
                torched = torch.as_tensor(cuda_tensor, device=self.device)

        return torched


class AsyncH2D(Transform):
    def __init__(self, **kwargs) -> None:
        """Asynchronously copy data from host to device."""
        super().__init__(**kwargs)
        self.register_input_type(torch.Tensor)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.torch_stream = torch.cuda.ExternalStream(self.stream)

    def transform(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Asynchronously copy data from host to device.

        Arguments:
            batch: List of tensors to copy.

        Returns:
            List of tensors copied to the device.
        """

        with torch.cuda.stream(self.copy_stream):
            out_tensors: List[torch.Tensor] = []
            for tensor in batch:
                tensor = tensor.to(self.device, non_blocking=True)
                out_tensors.append(tensor)

        self.copy_stream.synchronize()

        return out_tensors
