"""
dataset.py — PyTorch Dataset for SFHQ facial identity classification.
Reads sfhq_dataset.csv, applies transforms, and provides access to image-label pairs.

Splits exposed:
  'retain' : training data (150 identities)
  'test'    : held-out test data (30 identities, never trained on)
  'forget'  : sequential deletion identities (20 identities)
  'retain+forget' : full training set (retain + forget), used to train the original model M
  'forget_step_N' : images belonging to forget step N only (N = 0..19)
"""

import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Transforms 

def get_train_transform(img_size: int = 128) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_val_transform(img_size: int = 128) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


AGE_GROUP_NAMES = {0: "Young", 1: "Adult", 2: "Middle-Aged", 3: "Senior"}
NUM_CLASSES = 4


# Dataset

class VirtualIdentityDataset(Dataset):
    """
    Args:
        csv_path:     Path to sfhq_dataset.csv
        split:        One of 'retain', 'test', 'forget', 'retain+forget',
                      or 'forget_step_N' (N = 0..19)
        transform:    torchvision transform (default: val_transform)
        img_size:     Image resize target (default: 128)
    """

    VALID_SPLITS = {"retain", "test", "forget", "retain+forget"}

    def __init__(
        self,
        csv_path: str,
        split: str = "retain",
        transform=None,
        img_size: int = 128,
    ):
        df_full = pd.read_csv(csv_path)
        # Store dataset parent directory for resolving relative image paths
        self.data_dir = Path(csv_path).parent.parent.parent   
        # Filter by split
        if split == "retain+forget":
            df = df_full[df_full["split"].isin(["retain", "forget"])].copy()
        elif split.startswith("forget_step_"):
            step = int(split.split("_")[-1])
            df = df_full[df_full["forget_step"] == step].copy()
        elif split in self.VALID_SPLITS:
            df = df_full[df_full["split"] == split].copy()
        else:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Choose from {self.VALID_SPLITS} or 'forget_step_N'."
            )

        # Drop unknown labels
        df = df[df["age_group"] >= 0].reset_index(drop=True)

        self.df = df
        self.transform = transform or get_val_transform(img_size)
        self.img_size = img_size
        self.split = split
        self.num_classes = NUM_CLASSES

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        
        if not img_path.is_absolute():
            img_path = self.data_dir / img_path
        img_path = img_path.resolve()
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path} (original: {row['image_path']})")
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["age_group"])
        return img, label

    def get_metadata(self, idx: int) -> dict:
        # Return full metadata row for an image (useful for MIA)
        return self.df.iloc[idx].to_dict()

    def __repr__(self) -> str:
        counts = self.df["age_group"].value_counts().sort_index().to_dict()
        dist = {AGE_GROUP_NAMES.get(k, k): v for k, v in counts.items()}
        return (
            f"VirtualIdentityDataset(split='{self.split}', "
            f"n={len(self)}, classes={dist})"
        )
