from tests.image_dataset import EncodedImageDataset
from gradient_mechanics.data import torch_loading, transforms


def test_encoded_image_dataset_returns_correct_length(thirty_jpegs_dir_path):
    dataset = EncodedImageDataset(image_dir=thirty_jpegs_dir_path)
    assert len(dataset) == 30
    image_count = 0
    for encoded_image in dataset:
        assert isinstance(encoded_image, transforms.EncodedImage)
        image_count += 1

    assert image_count == 30


def test_encoded_image_dataset_collates_correctly(thirty_jpegs_dir_path):
    dataset = EncodedImageDataset(image_dir=thirty_jpegs_dir_path)
    gpu_transforms = [transforms.JPEGDecode(device_id=0)]
    dataloader = torch_loading.GPUDataLoader(
        dataset,
        batch_size=3,
        num_workers=0,
        gpu_device=0,
        gpu_prefetch_factor=1,
        gpu_transforms=gpu_transforms,
    )
    batch_count = 0
    for batch in dataloader:
        assert len(batch) == 3
        assert batch[0].shape == (720, 1280, 3)
        batch_count += 1

    assert batch_count == 10
