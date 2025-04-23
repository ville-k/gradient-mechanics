import torch
import cvcuda
import numpy as np
import pytest

from gradient_mechanics.data import torch_loading
from gradient_mechanics.data import torchdata_loading
from tests.image_dataset import EncodedImageDataset

from gradient_mechanics.data import transforms


@pytest.mark.parametrize("dataloader_cls", [torch_loading.GPUDataLoader, torchdata_loading.GPUDataLoader])
def test_image_decoding(thirty_jpegs_dir_path, dataloader_cls):
    dataset = EncodedImageDataset(image_dir=thirty_jpegs_dir_path)
    gpu_transforms = [transforms.JPEGDecode(device_id=0)]
    dataloader = dataloader_cls(
        dataset,
        batch_size=5,
        num_workers=0,
        gpu_device=0,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )

    batch_count = 0
    for batch in dataloader:
        assert len(batch) == 5
        assert batch[0].shape == (720, 1280, 3)
        assert batch[0].dtype == np.uint8
        batch_count += 1
    assert batch_count == 6

@pytest.mark.parametrize("dataloader_cls", [torch_loading.GPUDataLoader, torchdata_loading.GPUDataLoader])
def test_image_resizing(thirty_jpegs_dir_path, dataloader_cls):
    dataset = EncodedImageDataset(image_dir=thirty_jpegs_dir_path)
    gpu_transforms = [
        transforms.JPEGDecode(device_id=0),
        transforms.Resize(device_id=0, height=360, width=640),
    ]
    dataloader = dataloader_cls(
        dataset,
        batch_size=5,
        num_workers=0,
        gpu_device=0,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )

    batch_count = 0
    for batch in dataloader:
        assert len(batch) == 5
        assert batch[0].shape == (1, 360, 640, 3)
        assert batch[0].dtype == np.uint8
        batch_count += 1
    assert batch_count == 6


@pytest.mark.parametrize("dataloader_cls", [torch_loading.GPUDataLoader, torchdata_loading.GPUDataLoader])
def test_image_cropping(thirty_jpegs_dir_path, dataloader_cls):
    dataset = EncodedImageDataset(image_dir=thirty_jpegs_dir_path)
    gpu_transforms = [
        transforms.JPEGDecode(device_id=0),
        transforms.Crop(device_id=0, x=180, y=320, height=180, width=320),
    ]
    dataloader = dataloader_cls(
        dataset,
        batch_size=5,
        num_workers=0,
        gpu_device=0,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )

    batch_count = 0
    for batch in dataloader:
        assert len(batch) == 5
        assert batch[0].shape == (180, 320, 3)
        assert batch[0].dtype == np.uint8
        batch_count += 1
    assert batch_count == 6


@pytest.mark.parametrize("dataloader_cls", [torch_loading.GPUDataLoader, torchdata_loading.GPUDataLoader])
def test_image_resize_and_crop(thirty_jpegs_dir_path, dataloader_cls):
    dataset = EncodedImageDataset(image_dir=thirty_jpegs_dir_path)
    gpu_transforms = [
        transforms.JPEGDecode(device_id=0),
        transforms.Resize(device_id=0, height=360, width=640),
        transforms.Crop(device_id=0, x=90, y=160, height=90, width=160),
    ]
    dataloader = dataloader_cls(
        dataset,
        batch_size=5,
        num_workers=0,
        gpu_device=0,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )


    batch_count = 0
    for batch in dataloader:
        assert len(batch) == 5
        assert batch[0].shape == (1, 90, 160, 3)
        assert batch[0].dtype == np.uint8
        batch_count += 1
    assert batch_count == 6
