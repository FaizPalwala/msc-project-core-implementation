"""
dataset.py — PyTorch Dataset for SFHQ-InstantID facial-identity classification.

Reads dataset.csv / dataset.parquet (balanced) or dataset_imbalanced.csv /
dataset_imbalanced.parquet, applies configurable transforms, and returns
(image, identity_label, age_label) for each sample.

Standardised schema (v1.1, 750-id redesign)
───────────────────────────────────────────
Balanced (12 columns): image_path, identity_id, age_group (0-3), age,
gender (0/1), split (retain/forget), forget_step (-1 non-forget),
forget_step_poisson (-1 non-forget), image_subset (train/holdout),
arcface_similarity, laplacian_variance, detection_confidence.

Imbalanced (12 columns): image_path, identity_id, age_group (0-3), age,
gender (0/1), split (retain/forget), image_subset (train/holdout),
arcface_similarity, laplacian_variance, detection_confidence,
popularity_bin (str), images_per_identity (int).
— no schedule columns (forget_step*): the imbalanced axis is the
  popularity gradient, not time.

identity_id is the primary key (renamed from clusterid).  File format
(CSV or Parquet) is auto-detected from the file extension.

Splits
──────
  retain              — training data (675 identities in full config)
  forget              — all identities scheduled for deletion (75 identities)
  retain+forget       — full training set (750 identities)
  forget_step_N       — uniform-schedule batch N (5 identities each, 15 steps)
  forget_step_poisson_N — seeded-Poisson-schedule batch N (variable size)
  Per-identity selection: use identity_id filtering (see evaluate.py),
  not the removed forget_variant_N_M split.

The two schedule columns (forget_step, forget_step_poisson) are
balanced-only, complete projections over the same 75-id forget set:
  forget_step         — uniform (5 ids/step × 15), ordinal-safe, multi-seed μ±σ
  forget_step_poisson — seeded-Poisson (λ=5, single fixed schedule),
                        variable batch sizes; GDPR-arrival stress test
A retain row is -1 in both.  The imbalanced CSV has neither column.

Transforms
──────────
  train — 224×224, ImageNet-normalised, with augmentation
  val   — 224×224, ImageNet-normalised, no augmentation

Confound columns (always present in the final schema):
  gender, arcface_similarity, laplacian_variance, detection_confidence
Plus, in the imbalanced variant: popularity_bin, images_per_identity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ── Constants ─────────────────────────────────────────────────────────────────

# These are FALLBACKS only.  Callers that know the CSV path should use
# infer_identity_classes(csv) / infer_forget_schedule(csv) so the counts
# track the dataset (e.g. 600 vs 750 identities) instead of hardcoding.
NUM_IDENTITY_CLASSES = 600
NUM_AGE_CLASSES      =   4

AGE_GROUP_NAMES: dict[int, str] = {
    0: "Young", 1: "Adult", 2: "Middle-Aged", 3: "Senior",
}

# ImageNet normalisation (matching pretrained ResNet-18 weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset metadata inference ────────────────────────────────────────────────


def infer_identity_classes(csv_path: str | Path) -> int:
    """Number of distinct identity classes in the dataset.

    Reads only the CSV header + identity_id column (fast) and returns
    nunique(identity_id).  This replaces the hardcoded 600 so the
    pipeline adapts to dataset iterations (e.g. 750 identities).
    """
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["identity_id"])
    return int(df["identity_id"].nunique())


def infer_forget_schedule(csv_path: str | Path) -> tuple[int, int]:
    """(n_forget_steps, identities_per_step) from the forget split.

    Reads the forget_step column of the forget split and returns the
    number of distinct steps and the (modal) identities per step.
    Falls back to (15, 4) if the column is missing.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["split", "identity_id", "forget_step"])
    fg = df[df["split"] == "forget"]
    if "forget_step" not in fg.columns or fg["forget_step"].isna().all():
        return 15, 4
    steps = sorted(fg["forget_step"].dropna().unique())
    n_steps = len(steps)
    if n_steps == 0:
        return 15, 4
    counts = fg.groupby("forget_step")["identity_id"].nunique()
    per_step = int(counts.mode().iloc[0]) if len(counts) else 4
    return n_steps, per_step


def forget_split_name(forget_step: int, schedule: str = "uniform") -> str:
    """Split string for schedule batch N: 'forget_step_N' or 'forget_step_poisson_N'.

    All unlearning methods build their forget-set split from this helper so
    the schedule (uniform vs seeded-Poisson) flows through consistently.
    """
    if schedule == "poisson":
        return f"forget_step_poisson_{forget_step}"
    return f"forget_step_{forget_step}"


def cumulative_forgotten_count(csv_path: str | Path, step: int,
                               schedule: str = "uniform") -> int:
    """Number of forget identities processed through step (inclusive).

    Analysis axis per Shen et al. (2025): compare schedules by cumulative
    forgotten count, never raw step index.  For the uniform schedule this is
    (step+1) × ids/step; for Poisson it counts identities whose poisson
    batch index ≤ step (variable batch sizes).
    """
    import pandas as pd
    col = "forget_step_poisson" if schedule == "poisson" else "forget_step"
    df = pd.read_csv(csv_path, usecols=["split", "identity_id", col])
    fg = df[df["split"] == "forget"].dropna(subset=[col])
    return int(fg[fg[col] <= step]["identity_id"].nunique())

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
    labels are cluster IDs (0–N−1).  Age labels are proxy model-estimated
    age groups (0–3).

    Args:
        csv_path:  Path to dataset.csv or dataset_imbalanced.csv.
        split:     One of 'retain', 'forget', 'retain+forget',
                   'forget_step_N', or 'forget_step_poisson_N'.
        transform: torchvision transform (default: val_transform at 224).
        img_size:  Target resolution (default 224, matching pretrained
                   ResNet-18 input size).
    """

    VALID_SPLITS: set[str] = {"retain", "forget", "retain+forget"}

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "retain",
        transform: T.Compose | None = None,
        img_size: int = 224,
        subset: str | None = None,  # None → fail-fast; "all"|"train"|"holdout"
        order_seed: int | None = None,  # None → CSV order; int → order-stability permutation
    ) -> None:
        csv_path = Path(csv_path)
        # Auto-detect format: .parquet → read_parquet, else read_csv
        if csv_path.suffix.lower() == ".parquet":
            df_full = pd.read_parquet(csv_path)
        else:
            df_full = pd.read_csv(csv_path)

        # ── Order-stability permutation ─────────────────────────────────
        # order_seed remaps WHICH forget identities sit at WHICH uniform
        # step, deterministically, before any split filtering.  This is the
        # order-stability test (P3): same pretrained model, 5 seeds ×
        # different forget orderings, report μ±σ.  The permutation is a pure
        # function of (sorted forget identity list, order_seed), so every
        # dataset constructed with the same CSV + order_seed — in the method,
        # in _step_eval, in the retrain oracle — sees the SAME remap.
        # Only the uniform forget_step column is remapped: the Poisson
        # schedule is a single fixed stress test, not an order axis.
        if order_seed is not None and "forget_step" in df_full.columns:
            fg_mask = df_full["split"] == "forget"
            fg_ids = sorted(df_full.loc[fg_mask, "identity_id"].unique())
            steps = sorted(df_full.loc[fg_mask, "forget_step"].dropna().unique())
            if fg_ids and len(steps) > 0:
                n_steps = len(steps)
                rng = np.random.RandomState(order_seed)
                perm = rng.permutation(len(fg_ids))
                # Contiguous chunks: identity at perm[idx] → step idx // per_step.
                per_step = int(np.ceil(len(fg_ids) / n_steps))
                step_of = {}
                for idx, id_pos in enumerate(perm):
                    step_of[fg_ids[id_pos]] = int(idx // per_step)
                df_full.loc[fg_mask, "forget_step"] = (
                    df_full.loc[fg_mask, "identity_id"].map(step_of)
                )

        # Data root: csv parent's parent (e.g. data/dataset/dataset.csv → data/).
        # Image paths are resolved relative to either the CSV's own directory
        # (data/dataset/images/…) or this root (data/images/…) — see __getitem__.
        self.csv_dir = csv_path.parent
        self.data_dir = csv_path.parent.parent

        # ── Split filter ──────────────────────────────────────────────────
        if split == "retain+forget":
            df = df_full[df_full["split"].isin(["retain", "forget"])].copy()
        elif split.startswith("forget_step_poisson_"):
            # Format: forget_step_poisson_{N} — seeded-Poisson schedule batch.
            if "forget_step_poisson" not in df_full.columns:
                raise ValueError(
                    f"Split '{split}' requires a 'forget_step_poisson' column, "
                    f"but {csv_path.name} has none. The Poisson schedule is "
                    f"balanced-only — the imbalanced CSV has no time axis."
                )
            step = int(split.split("_")[-1])
            df = df_full[df_full["forget_step_poisson"] == step].copy()
        elif split.startswith("forget_step_"):
            # Format: forget_step_{N} — uniform-schedule batch.
            if "forget_step" not in df_full.columns:
                raise ValueError(
                    f"Split '{split}' requires a 'forget_step' column, "
                    f"but {csv_path.name} has none. The uniform schedule is "
                    f"balanced-only — the imbalanced CSV has no time axis."
                )
            step = int(split.split("_")[-1])
            df = df_full[df_full["forget_step"] == step].copy()
        elif split in self.VALID_SPLITS:
            df = df_full[df_full["split"] == split].copy()
        else:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Choose from {self.VALID_SPLITS}, "
                f"'forget_step_N', or 'forget_step_poisson_N'."
            )

        # ── Subset filter (per-image train/holdout) ───────────────────────
        has_subset_col = "image_subset" in df.columns
        if has_subset_col:
            if subset is None:
                raise ValueError(
                    f"CSV {csv_path.name} has an 'image_subset' column but no "
                    f"subset was given. Pass subset='train' or subset='holdout' "
                    f"explicitly — subset='all' silently leaks holdout images "
                    f"into training and corrupts every evaluation metric. "
                    f"(If 'all' is genuinely intended, pass subset='all' explicitly.)"
                )
            if subset in ("train", "holdout"):
                df = df[df["image_subset"] == subset]
            elif subset != "all":
                raise ValueError(
                    f"Unknown subset '{subset}'. "
                    f"Choose 'all', 'train', or 'holdout'."
                )
            # subset="all" (explicit) → no filtering, deliberate use
        # If image_subset column is missing (old CSV), all images are
        # treated as "all" regardless of the subset parameter — back‑compat.

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
        # Relative paths are tried against the CSV's own directory first
        # (data/dataset/images/…), then against the data root (data/images/…).
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            candidate = (self.csv_dir / img_path).resolve()
            if candidate.exists():
                img_path = candidate
            else:
                img_path = (self.data_dir / img_path).resolve()

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path} (original: {row['image_path']})"
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        identity_label = int(row["identity_id"])
        age_label      = int(row.get("age_group", -1))

        return img, identity_label, age_label

    # ── metadata access ───────────────────────────────────────────────────

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Return full row as a dict, including optional columns."""
        return self.df.iloc[idx].to_dict()

    def get_identity_labels(self) -> list[int]:
        """Unique identity labels present in this split's subset."""
        return sorted(self.df["identity_id"].unique().tolist())

    def identity_label_at(self, idx: int) -> int:
        """Identity label at index *idx* without loading the image.

        For code that needs the label alone (e.g. relabelling loops),
        this avoids the full ``__getitem__`` (image decode + transform).
        """
        return int(self.df.iloc[idx]["identity_id"])

    @property
    def meta_columns(self) -> list[str]:
        """Optional metadata columns detected in this CSV."""
        return self._meta_columns

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_ids = self.df["identity_id"].nunique()
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
        "age", "gender", "arcface_similarity", "laplacian_variance",
        "popularity_bin", "images_per_identity", "detection_confidence",
    }
    return sorted(c for c in known if c in df.columns)
