"""Dataset loaders.

Two layouts are supported, both documented in ``docs/DATA_FORMAT.md``:

1. ``ClassFolderDataset`` — root with one subfolder per class. Used by the
   class-conditional diffusion scripts and the EfficientNetV2 classifier.

2. ``UnpairedFolderDataset`` — two flat folders (``trainA`` and ``trainB``)
   for unpaired CycleGAN stain normalization.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTS)


class ClassFolderDataset(Dataset):
    """Patches arranged as ``<root>/<class_name>/*.png``.

    The class names are taken from ``list_classes`` (case-sensitive); the order
    of that list defines the integer label (index 0 = first class).

    Args:
        path_root: dataset root directory.
        list_classes: ordered list of class folder names. Must all exist under
            ``path_root``.
        int_image_size: side length the patches are resized to (square).
        bool_augment: enable random horizontal/vertical flips (default True).
        tuple_normalize_mean: per-channel mean for normalization.
        tuple_normalize_std: per-channel std for normalization.
    """

    def __init__(
        self,
        path_root: str | Path,
        list_classes: Sequence[str],
        int_image_size: int,
        bool_augment: bool = True,
        tuple_normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        tuple_normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.path_root = Path(path_root)
        self.list_classes = list(list_classes)
        self.int_image_size = int_image_size
        self.bool_augment = bool_augment

        self.list_samples: list[tuple[Path, int]] = []
        for int_label, str_class in enumerate(self.list_classes):
            path_class_dir = self.path_root / str_class
            if not path_class_dir.is_dir():
                raise FileNotFoundError(
                    f"Class folder missing: {path_class_dir}. "
                    f"Expected layout: {self.path_root}/<class>/*.png"
                )
            list_files = _list_images(path_class_dir)
            if len(list_files) == 0:
                raise RuntimeError(f"No images under {path_class_dir}")
            for path_img in list_files:
                self.list_samples.append((path_img, int_label))

        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple_normalize_mean, tuple_normalize_std),
        ])

    def __len__(self) -> int:
        return len(self.list_samples)

    def _augment(self, tensor_image: torch.Tensor) -> torch.Tensor:
        if not self.bool_augment:
            return tensor_image
        if random.random() < 0.5:
            tensor_image = torch.flip(tensor_image, dims=[2])  # horizontal
        if random.random() < 0.5:
            tensor_image = torch.flip(tensor_image, dims=[1])  # vertical
        return tensor_image

    def __getitem__(self, int_index: int) -> tuple[torch.Tensor, int]:
        path_img, int_label = self.list_samples[int_index]
        image = Image.open(path_img).convert("RGB")
        if image.size != (self.int_image_size, self.int_image_size):
            image = image.resize(
                (self.int_image_size, self.int_image_size), Image.BICUBIC
            )
        tensor_image = self.transform_to_tensor(image)
        tensor_image = self._augment(tensor_image)
        return tensor_image, int_label


class UnpairedFolderDataset(Dataset):
    """Two unpaired image pools for CycleGAN training.

    Layout::

        <path_root>/
        ├── trainA/*.png
        └── trainB/*.png

    Each ``__getitem__`` returns ``(tensor_a, tensor_b)`` where ``tensor_a``
    is the ``int_index``-th sample of pool A and ``tensor_b`` is a *random*
    sample of pool B (so order is decoupled across domains, matching the
    standard CycleGAN training recipe).
    """

    def __init__(
        self,
        path_root: str | Path,
        int_image_size: int = 1024,
        bool_horizontal_flip: bool = False,
        str_subdir_a: str = "trainA",
        str_subdir_b: str = "trainB",
    ) -> None:
        self.path_root = Path(path_root)
        self.int_image_size = int_image_size

        path_a = self.path_root / str_subdir_a
        path_b = self.path_root / str_subdir_b
        if not path_a.is_dir() or not path_b.is_dir():
            raise FileNotFoundError(
                f"Expected '{str_subdir_a}' and '{str_subdir_b}' under {self.path_root}"
            )

        self.list_files_a = _list_images(path_a)
        self.list_files_b = _list_images(path_b)
        if not self.list_files_a or not self.list_files_b:
            raise RuntimeError(
                f"Empty CycleGAN pool: A={len(self.list_files_a)} B={len(self.list_files_b)}"
            )

        list_transforms: list = [
            transforms.Resize(int_image_size, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(int_image_size),
        ]
        if bool_horizontal_flip:
            list_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        list_transforms += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(list_transforms)

    def __len__(self) -> int:
        return max(len(self.list_files_a), len(self.list_files_b))

    def _load(self, path_img: Path) -> torch.Tensor:
        return self.transform(Image.open(path_img).convert("RGB"))

    def __getitem__(self, int_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path_a = self.list_files_a[int_index % len(self.list_files_a)]
        path_b = self.list_files_b[random.randrange(len(self.list_files_b))]
        return self._load(path_a), self._load(path_b)
