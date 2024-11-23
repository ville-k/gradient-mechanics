import pathlib

import torch
from torchvision import io
from gradient_mechanics.data import transforms


class EncodedImageDataset(torch.utils.data.Dataset):
    """A dataset that reads JPEG images from a directory. The images are read as torch tensors, but not decoded."""

    def __init__(self, image_dir: str):
        super().__init__()
        self.image_dir = image_dir
        self._image_paths = None

    def _lazy_init(self):
        if self._image_paths is not None:
            return

        self._image_paths = list(pathlib.Path(self.image_dir).glob("*.[jpeg jpg]*"))

    def __len__(self):
        self._lazy_init()
        return len(self._image_paths)

    def __getitem__(self, index):
        self._lazy_init()
        image_path = self._image_paths[index]
        image = io.read_file(str(image_path))
        return transforms.EncodedImage(buffer=image)
