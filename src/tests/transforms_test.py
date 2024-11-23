from typing import Any, List
import typing

import cvcuda
import numpy as np
import pytest
import torch
from gradient_mechanics.data import transforms


@pytest.fixture
def single_jpeg(single_jpeg_path) -> cvcuda.Tensor:
    data = bytearray(open(single_jpeg_path, "rb").read())
    encoded_image = transforms.EncodedImage(
        torch.from_numpy(np.frombuffer(data, dtype=np.uint8))
    )
    encoded_image_batch = transforms.EncodedImageBatch([encoded_image])
    decode = transforms.JPEGDecode(device_id=0)
    decoded_image = decode(encoded_image_batch)
    return decoded_image[0]


class MockTransformableType(typing.NamedTuple):
    data: int


class MockTransformedType(typing.NamedTuple):
    data: str


class MockNamedTupleType(typing.NamedTuple):
    items: List[Any]


class MockTransform(transforms.Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_input_type(MockTransformableType)

    def transform(
        self, batch: List[MockTransformableType]
    ) -> List[MockTransformedType]:
        return [MockTransformedType(data=str(sample.data)) for sample in batch]


def test_transform_does_not_receive_unregistered_input_type():
    transform = MockTransform(device_id=0)
    data = [1, 2, 3]
    transformed = transform(data)

    assert transformed == data


@pytest.mark.parametrize(
    "data, expected_transformed",
    [
        # Nothing to transform
        (tuple(), tuple()),
        ((1,), (1,)),
        (list(), list()),
        ([1], [1]),
        (set(), set()),
        ({1}, {1}),
        (dict(), dict()),
        ({"key": 42}, {"key": 42}),
        # Single item
        (MockTransformableType(1), MockTransformedType("1")),
        ((MockTransformableType(1),), (MockTransformedType("1"),)),
        ([MockTransformableType(1)], [MockTransformedType("1")]),
        ({MockTransformableType(1)}, {MockTransformedType("1")}),
        ({"key": MockTransformableType(1)}, {"key": MockTransformedType("1")}),
        (
            MockNamedTupleType([MockTransformableType(1)]),
            MockNamedTupleType([MockTransformedType("1")]),
        ),
        # Multiple items
        (
            (MockTransformableType(1), MockTransformableType(2)),
            (MockTransformedType("1"), MockTransformedType("2")),
        ),
        (
            [MockTransformableType(1), MockTransformableType(2)],
            [MockTransformedType("1"), MockTransformedType("2")],
        ),
        (
            {MockTransformableType(1), MockTransformableType(2)},
            {MockTransformedType("1"), MockTransformedType("2")},
        ),
        (
            {"key1": MockTransformableType(1), "key2": MockTransformableType(2)},
            {"key1": MockTransformedType("1"), "key2": MockTransformedType("2")},
        ),
        # Mixed items
        ((MockTransformableType(1), 42), (MockTransformedType("1"), 42)),
        ({MockTransformableType(1), 42}, {MockTransformedType("1"), 42}),
        (
            {"key": MockTransformableType(1), "key2": 2},
            {"key": MockTransformedType("1"), "key2": 2},
        ),
    ],
)
def test_receives_registered_input_type(data, expected_transformed):
    transform = MockTransform(device_id=0)
    transformed = transform(data)

    assert transformed == expected_transformed


def test_transform_prints_out_helpful_debug_message_when_magic_breaks(capsys):
    transform = MockTransform(device_id=0)
    data = [MockTransformableType(1), 1]
    with pytest.raises(Exception):
        transform(data)

    captured = capsys.readouterr()
    assert (
        "Exception was raised during the processing of transform 'MockTransform'. This could be caused by input data not conforming to what is expected of container types."
        in captured.err
    )


def test_jpeg_decode(single_jpeg_path):
    data = bytearray(open(single_jpeg_path, "rb").read())
    encoded_image = transforms.EncodedImage(
        torch.from_numpy(np.frombuffer(data, dtype=np.uint8))
    )
    encoded_image_batch = transforms.EncodedImageBatch([encoded_image])
    transform = transforms.JPEGDecode(device_id=0)
    decoded_image = transform(encoded_image_batch)

    assert len(decoded_image) == 1
    assert decoded_image[0].shape == (720, 1280, 3)


def test_jpeg_decode_accepts_stream(single_jpeg_path):
    torch_stream = torch.cuda.Stream(device=0)

    data = bytearray(open(single_jpeg_path, "rb").read())
    encoded_image = transforms.EncodedImage(
        torch.from_numpy(np.frombuffer(data, dtype=np.uint8))
    )
    encoded_image_batch = transforms.EncodedImageBatch([encoded_image])
    transform = transforms.JPEGDecode(device_id=0, stream=torch_stream.cuda_stream)
    decoded_image = transform(encoded_image_batch)

    assert len(decoded_image) == 1
    assert decoded_image[0].shape == (720, 1280, 3)


def test_crop(single_jpeg):
    crop = transforms.Crop(device_id=0, x=72, y=128, width=72, height=128)
    cropped_image = crop([single_jpeg])

    assert len(cropped_image) == 1
    assert cropped_image[0].shape == (128, 72, 3)


def test_crop_accepts_stream(single_jpeg):
    torch_stream = torch.cuda.Stream(device=0)

    crop = transforms.Crop(
        device_id=0, x=72, y=128, width=72, height=128, stream=torch_stream.cuda_stream
    )
    cropped_image = crop([single_jpeg])

    assert len(cropped_image) == 1
    assert cropped_image[0].shape == (128, 72, 3)


def test_resize(single_jpeg):
    resize = transforms.Resize(device_id=0, width=128, height=64)
    resized_image = resize([single_jpeg])
    assert len(resized_image) == 1
    assert resized_image[0].shape == (1, 64, 128, 3)


def test_resize_accepts_stream(single_jpeg):
    torch_stream = torch.cuda.Stream(device=0)

    resize = transforms.Resize(
        device_id=0, width=128, height=64, stream=torch_stream.cuda_stream
    )
    resized_image = resize([single_jpeg])

    assert len(resized_image) == 1
    assert resized_image[0].shape == (1, 64, 128, 3)


def test_to_tensor_single(single_jpeg):
    torch_stream = torch.cuda.Stream(device=0)
    to_tensor = transforms.ToTensor(device_id=0, stream=torch_stream.cuda_stream)
    tensor_image = to_tensor([single_jpeg])

    assert tensor_image.shape == (1, 3, 720, 1280)


def test_to_tensor_multiple(single_jpeg):
    to_tensor = transforms.ToTensor(device_id=0)
    tensor_image = to_tensor([single_jpeg, single_jpeg])

    assert tensor_image.shape == (2, 3, 720, 1280)


def test_to_tensor_accepts_stream(single_jpeg):
    torch_stream = torch.cuda.Stream(device=0)

    to_tensor = transforms.ToTensor(device_id=0, stream=torch_stream.cuda_stream)
    tensor_image = to_tensor([single_jpeg])

    assert tensor_image.shape == (1, 3, 720, 1280)
