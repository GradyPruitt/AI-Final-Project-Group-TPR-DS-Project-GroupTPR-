"""
PyTorch Dataset for DoriaNET building damage classification.

Given the manifest CSV from preprocessing.py, this Dataset loads:
  - the frame image
  - the corresponding building mask
  - crops the frame using the mask's bounding box (with optional padding)
  - applies image transforms (resize, normalize)
  - returns (image_tensor, damage_level)

Why crop instead of using the full frame
------------------------------------------
The reference paper (Cheng et al. 2021) uses a two-stage SPDA architecture:
Model L localizes buildings (Mask R-CNN), Model C classifies the cropped
patches (MobileNet). We simplify to just Model C — using the ground-truth
masks to skip localization. This isolates the classification problem.

For our 3-model comparison, every architecture sees the same crops produced
by this Dataset, so the comparison is about classifier capacity, not
localization quality.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


N_CLASSES = 6   # damage levels 0..5

# Standard ImageNet normalization (every transfer-learning backbone expects this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def default_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    """Standard transforms for ImageNet-pretrained backbones.

    Train transforms include light augmentation (flip, color jitter) which
    helps with our small dataset. Val/test transforms are deterministic.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class DoriaNetBuildingDataset(Dataset):
    """One sample = one cropped building patch + its damage label.

    Parameters
    ----------
    manifest_df : DataFrame from preprocessing.build_manifest(...)
    split : "train", "val", or "test" — filters the manifest.
    transform : torchvision transform applied to the cropped patch.
    pad_frac : pad the bounding box by this fraction of its width/height
        before cropping. 0.10 = 10% pad on each side. Helps the model see
        a little context around the building.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        transform=None,
        pad_frac: float = 0.10,
    ):
        assert split in ("train", "val", "test"), split
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.transform = transform if transform is not None else default_transforms(split == "train")
        self.pad_frac = pad_frac

        if len(self.df) == 0:
            raise ValueError(f"Empty split '{split}' — check the manifest.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["frame_path"]).convert("RGB")

        # Pad the bbox by pad_frac, clamped to image bounds
        x0, y0, x1, y1 = row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]
        w, h = x1 - x0, y1 - y0
        pad_w = int(w * self.pad_frac)
        pad_h = int(h * self.pad_frac)
        W, H = img.size
        x0 = max(0, x0 - pad_w)
        y0 = max(0, y0 - pad_h)
        x1 = min(W, x1 + pad_w)
        y1 = min(H, y1 + pad_h)

        crop = img.crop((x0, y0, x1, y1))
        x = self.transform(crop)
        y = torch.tensor(int(row["damage_level"]), dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Convenience for class imbalance handling
# ---------------------------------------------------------------------------

def class_weights(manifest_df: pd.DataFrame, split: str = "train") -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split.

    Use as `nn.CrossEntropyLoss(weight=class_weights(...))` to upweight rare
    classes. This is a simple way to address the imbalance shown in the EDA.
    """
    sub = manifest_df[manifest_df["split"] == split]
    counts = sub["damage_level"].value_counts().sort_index()
    counts = counts.reindex(range(N_CLASSES), fill_value=0).astype(float)
    # Inverse frequency, then normalize to mean 1.0
    weights = 1.0 / (counts + 1.0)   # +1 smoothing so missing class doesn't div0
    weights = weights * (len(weights) / weights.sum())
    return torch.tensor(weights.values, dtype=torch.float32)
