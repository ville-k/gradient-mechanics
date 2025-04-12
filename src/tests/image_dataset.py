import pathlib
import os
import torch
from torchvision import io
from gradient_mechanics.data import transforms


class EncodedImageDataset(torch.utils.data.Dataset):
    """A dataset that reads JPEG images from a directory. The images are read as torch tensors, but not decoded."""

    def __init__(self, image_dir: str):
        super().__init__()
        self.image_dir = image_dir
        # Do initialization upfront rather than lazily
        self._image_paths = sorted(
            [path for path in pathlib.Path(image_dir).glob("*.jp*g") 
             if os.path.isfile(path)]
        )

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, index):
        try:
            image_path = self._image_paths[index]
            image = io.read_file(str(image_path))
            return transforms.EncodedImage(buffer=image)
        except Exception as e:
            # Provide better error information
            raise RuntimeError(f"Error reading image at index {index}, path {self._image_paths[index]}: {str(e)}")
