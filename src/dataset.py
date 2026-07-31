"""
dataset.py — PyTorch Dataset for SFHQ-InstantID facial-identity classification.

Reads dataset.csv (balanced) or dataset_imbalanced.csv, applies configurable
transforms, and returns (image, identity_label, age_label) for each sample.

Splits
──────
  retain          — training data (450 identities in full config)
  test            — held-out identities, never trained on (90 identities)
  forget          — all identities scheduled for deletion (60 identities)
  retain+forget   — full training set (retain + forget, 510 identities)
  forget_step_N   — images belonging to forget step N (4 identities each, 15 steps)
  forget_variant_N_M — images for variant M within forget step N (1 identity)

Transforms
──────────
  train — 224×224, ImageNet-normalised, with augmentation
  val   — 224×224, ImageNet-normalised, no augmentation

Optional metadata columns (auto-detected from CSV):
  gender, arcface_similarity, laplacian_variance, popularity_bin,
  images_per_identity, detection_confidence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_IDENTITY_CLASSES = 600
NUM_AGE_CLASSES      =   4

AGE_GROUP_NAMES: dict[int, str] = {
    0: "Young", 1: "Adult", 2: "Middle-Aged", 3: "Senior",
}

# ImageNet normalisation (matching pretrained ResNet-18 weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ────────────────────────────────────────────────────────────────


def get_train_transform(img_size: int = 224) -> T.Compose:
    """Training transform: augment + normalise.

    Applies random horizontal flip, mild colour jitter, and ImageNet
    normalisation so the pretrained backbone receives data in its
    expected range.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 224) -> T.Compose:
    """Validation / evaluation transform: resize only + normalise."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────


class VirtualIdentityDataset(Dataset):
    """SFHQ-InstantID dataset with identity-labelled clusters.

    Each sample returns (image, identity_label, age_label).  Identity
    labels are cluster IDs (0–599).  Age labels are proxy model-estimated
    age groups (0–3).

    Args:
        csv_path:  Path to dataset.csv or dataset_imbalanced.csv.
        split:     One of 'retain', 'test', 'forget', 'retain+forget',
                   'forget_step_N', or 'forget_variant_N_M'.
        transform: torchvision transform (default: val_transform at 224).
        img_size:  Target resolution (default 224, matching pretrained
                   ResNet-18 input size).
    """

    VALID_SPLITS: set[str] = {"retain", "test", "forget", "retain+forget"}

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "retain",
        transform: T.Compose | None = None,
        img_size: int = 224,
    ) -> None:
        csv_path = Path(csv_path)
        df_full = pd.read_csv(csv_path)

        # Data root: csv parent's parent (e.g. data/dataset/dataset.csv → data/)
        self.data_dir = csv_path.parent.parent

        # ── Split filter ──────────────────────────────────────────────────
        if split == "retain+forget":
            df = df_full[df_full["split"].isin(["retain", "forget"])].copy()
        elif split.startswith("forget_variant_"):
            # Format: forget_variant_{step}_{variant}
            parts = split.split("_")
            step, variant = int(parts[2]), int(parts[3])
            df = df_full[
                (df_full["forget_step"] == step)
                & (df_full["forget_variant"] == variant)
            ].copy()
        elif split.startswith("forget_step_"):
            step = int(split.split("_")[-1])
            df = df_full[df_full["forget_step"] == step].copy()
        elif split in self.VALID_SPLITS:
            df = df_full[df_full["split"] == split].copy()
        else:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Choose from {self.VALID_SPLITS}, "
                f"'forget_step_N', or 'forget_variant_N_M'."
            )

        # Drop rows with invalid age labels
        if "age_group" in df.columns:
            df = df[df["age_group"] >= 0]
        df = df.reset_index(drop=True)

        self.df = df
        self.transform = transform or get_val_transform(img_size)
        self.img_size = img_size
        self.split = split

        # Detect optional metadata columns
        self._meta_columns = _detect_meta_columns(df)

    # ── length ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    # ── sample access ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]

        # ── Resolve image path ────────────────────────────────────────────
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = (self.data_dir / img_path).resolve()

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path} (original: {row['image_path']})"
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        identity_label = int(row["clusterid"])
        age_label      = int(row.get("age_group", -1))

        return img, identity_label, age_label

    # ── metadata access ───────────────────────────────────────────────────

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Return full row as a dict, including optional columns."""
        return self.df.iloc[idx].to_dict()

    def get_identity_labels(self) -> list[int]:
        """Unique identity labels present in this split's subset."""
        return sorted(self.df["clusterid"].unique().tolist())

    @property
    def meta_columns(self) -> list[str]:
        """Optional metadata columns detected in this CSV."""
        return self._meta_columns

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_ids = self.df["clusterid"].nunique()
        age_dist = self.df["age_group"].value_counts().sort_index().to_dict() \
            if "age_group" in self.df.columns else {}
        age_str = ", ".join(
            f"{AGE_GROUP_NAMES.get(k, k)}:{v}" for k, v in age_dist.items()
        )
        return (
            f"VirtualIdentityDataset(split='{self.split}', "
            f"samples={len(self)}, identities={n_ids}, "
            f"ages=[{age_str}])"
        )


# ── Backwards-compatible alias ────────────────────────────────────────────────

SFHQDataset = VirtualIdentityDataset


# ── Internal helpers ──────────────────────────────────────────────────────────


def _detect_meta_columns(df: pd.DataFrame) -> list[str]:
    """Return optional demographic/quality columns present in the DataFrame."""
    known = {
        "gender", "arcface_similarity", "laplacian_variance",
        "popularity_bin", "images_per_identity", "detection_confidence",
    }
    return sorted(c for c in known if c in df.columns)
