"""
Data utilities for SpectralVFL multimodal skin lesion dataset.
Handles HAM10000: images + tabular metadata with stratified group splitting.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import config


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------


def get_image_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
) -> transforms.Compose:
    """Standard torchvision transforms for skin lesion images."""
    base = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *base,
        ])
    return transforms.Compose(base)


# -----------------------------------------------------------------------------
# Tabular Preprocessing
# -----------------------------------------------------------------------------


class TabularPreprocessor:
    """Encodes and scales HAM10000 tabular features using OneHotEncoder for categoricals."""

    def __init__(self, tabular_config=None):
        self.cfg = tabular_config or config.tabular
        self._age_scaler = StandardScaler()
        self._sex_encoder = OneHotEncoder(
            categories=[list(self.cfg.sex_values)],
            sparse_output=False,
            handle_unknown="ignore",
        )
        self._localization_encoder = OneHotEncoder(
            categories=[list(self.cfg.localization_values)],
            sparse_output=False,
            handle_unknown="ignore",
        )
        self._fitted = False
        self._tabular_dim: Optional[int] = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit scalers/encoders on training data."""
        # Age: handle NaN with median
        age = df[self.cfg.age_col].fillna(df[self.cfg.age_col].median()).values.reshape(-1, 1)
        self._age_scaler.fit(age)

        # Categorical: OneHotEncoder with handle_unknown='ignore'
        sex = self._prepare_categorical(df[self.cfg.sex_col])
        self._sex_encoder.fit(sex)

        loc = self._prepare_categorical(df[self.cfg.localization_col])
        self._localization_encoder.fit(loc)

        # Compute output dimension dynamically
        age_dim = 1
        sex_dim = self._sex_encoder.categories_[0].shape[0]
        loc_dim = self._localization_encoder.categories_[0].shape[0]
        self._tabular_dim = age_dim + sex_dim + loc_dim

        self._fitted = True
        return self

    def _prepare_categorical(self, series: pd.Series) -> np.ndarray:
        """Prepare categorical column for OneHotEncoder (2D, str)."""
        return series.astype(str).str.lower().replace("", "unknown").values.reshape(-1, 1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform tabular data to flattened float array: [age | sex_onehot | loc_onehot]."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        age = df[self.cfg.age_col].fillna(df[self.cfg.age_col].median()).values.reshape(-1, 1)
        age_scaled = self._age_scaler.transform(age)

        sex = self._prepare_categorical(df[self.cfg.sex_col])
        sex_onehot = self._sex_encoder.transform(sex)

        loc = self._prepare_categorical(df[self.cfg.localization_col])
        loc_onehot = self._localization_encoder.transform(loc)

        return np.hstack([age_scaled, sex_onehot, loc_onehot]).astype(np.float32)

    @property
    def tabular_dim(self) -> int:
        """Output dimension of tabular vector (computed dynamically after fit)."""
        if self._tabular_dim is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self._tabular_dim


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class MultimodalSkinDataset(Dataset):
    """
    PyTorch Dataset for HAM10000: image + tabular metadata.
    Returns (image_tensor, tabular_tensor, label).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dirs: Union[Tuple[Path, Path], List[Path]],
        tabular_preprocessor: TabularPreprocessor,
        label_encoder: LabelEncoder,
        transform: Optional[transforms.Compose] = None,
        image_ext: str = ".jpg",
    ):
        self.df = df.reset_index(drop=True)
        self.image_dirs = list(image_dirs) if isinstance(image_dirs, tuple) else image_dirs
        self.preprocessor = tabular_preprocessor
        self.label_encoder = label_encoder
        self.transform = transform or get_image_transforms(is_training=False)
        self.image_ext = image_ext

        self.tabular_tensors = torch.from_numpy(
            self.preprocessor.transform(self.df)
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_id: str) -> Path:
        """Resolve image path across part1/part2 directories."""
        fname = f"{image_id}{self.image_ext}"
        for d in self.image_dirs:
            p = Path(d) / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found: {image_id} in {self.image_dirs}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_id = row[config.tabular.image_id_col]
        label = row[config.tabular.label_col]

        # Load image
        path = self._resolve_image_path(image_id)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tabular = self.tabular_tensors[idx]
        label_idx = self.label_encoder.transform([label])[0]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return image, tabular, label_tensor


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------


def stratified_group_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "lesion_id",
    label_col: str = "dx",
    val_fold: int = 0,
    test_fold: int = 1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe using StratifiedGroupKFold to prevent data leakage.
    Groups by lesion_id (or patient_id) so no lesion appears in multiple splits.

    Returns:
        train_df, val_df, test_df
    """
    cfg = config.tabular
    group_col = group_col or cfg.lesion_id_col
    label_col = label_col or cfg.label_col

    if n_splits < 3:
        raise ValueError("n_splits must be >= 3 for train/val/test.")

    df_clean = df.dropna(subset=[label_col, group_col]).copy()
    valid_labels = set(config.tabular.dx_classes)
    df_clean = df_clean[df_clean[label_col].isin(valid_labels)]
    groups = df_clean[group_col].values
    labels = df_clean[label_col].values

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = list(sgkf.split(df_clean, labels, groups))
    all_val_idx = folds[val_fold][1]
    all_test_idx = folds[test_fold][1]
    train_idx = np.setdiff1d(
        np.arange(len(df_clean)),
        np.union1d(all_val_idx, all_test_idx),
    )

    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    val_df = df_clean.iloc[all_val_idx].reset_index(drop=True)
    test_df = df_clean.iloc[all_test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
# DataLoaders
# -----------------------------------------------------------------------------


def get_dataloaders(
    metadata_path: Optional[Union[str, Path]] = None,
    image_dirs: Optional[Union[Tuple[Path, Path], List[Path]]] = None,
    data_root: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    n_splits: int = 5,
    val_fold: int = 0,
    test_fold: int = 1,
    num_workers: Optional[int] = None,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, TabularPreprocessor, LabelEncoder]:
    """
    Load metadata, split with StratifiedGroupKFold, and return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, tabular_preprocessor, label_encoder
    """
    hp = config.hyperparams
    batch_size = batch_size or hp.batch_size
    num_workers = num_workers if num_workers is not None else hp.num_workers

    # Resolve paths
    if data_root:
        root = Path(data_root)
        meta_path = root / config.paths.metadata_filename
        img_dirs = [
            root / config.paths.image_dir_part1,
            root / config.paths.image_dir_part2,
        ]
    else:
        if metadata_path is None or image_dirs is None:
            raise ValueError("Provide either data_root or both metadata_path and image_dirs.")
        meta_path = Path(metadata_path)
        img_dirs = list(image_dirs) if isinstance(image_dirs, tuple) else image_dirs

    df = pd.read_csv(meta_path)
    train_df, val_df, test_df = stratified_group_split(
        df,
        n_splits=n_splits,
        val_fold=val_fold,
        test_fold=test_fold,
        random_state=hp.random_seed,
    )

    # Fit preprocessor on train only
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    label_encoder = LabelEncoder()
    label_encoder.fit(list(config.tabular.dx_classes))

    train_ds = MultimodalSkinDataset(
        train_df,
        img_dirs,
        preprocessor,
        label_encoder,
        transform=get_image_transforms(hp.image_size, is_training=augment_train),
    )
    val_ds = MultimodalSkinDataset(
        val_df,
        img_dirs,
        preprocessor,
        label_encoder,
        transform=get_image_transforms(hp.image_size, is_training=False),
    )
    test_ds = MultimodalSkinDataset(
        test_df,
        img_dirs,
        preprocessor,
        label_encoder,
        transform=get_image_transforms(hp.image_size, is_training=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=hp.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=hp.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=hp.pin_memory,
    )

    return train_loader, val_loader, test_loader, preprocessor, label_encoder


def get_kfold_dataloaders(
    df: pd.DataFrame,
    label_col: str,
    k: int = 5,
    batch_size: int = 32,
    *,
    image_dirs: Union[Tuple[Path, Path], List[Path]],
    num_workers: Optional[int] = None,
    random_state: int = 42,
    augment_train: bool = True,
) -> Iterator[Tuple[int, DataLoader, DataLoader]]:
    """
    Stratified K-fold splits (sample-level). Yields one fold at a time.

    Fits ``TabularPreprocessor`` on the training split of each fold only.

    Yields:
        (fold_idx, train_loader, val_loader) with ``fold_idx`` in ``0 .. k-1``.
    """
    hp = config.hyperparams
    num_workers = num_workers if num_workers is not None else hp.num_workers
    cfg = config.tabular

    valid_labels = set(config.tabular.dx_classes)
    df_clean = df.dropna(subset=[label_col]).copy()
    df_clean = df_clean[df_clean[label_col].isin(valid_labels)].reset_index(drop=True)

    y = df_clean[label_col].values
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    img_dirs = list(image_dirs) if isinstance(image_dirs, tuple) else image_dirs

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df_clean)), y)):
        train_df = df_clean.iloc[train_idx].reset_index(drop=True)
        val_df = df_clean.iloc[val_idx].reset_index(drop=True)

        preprocessor = TabularPreprocessor()
        preprocessor.fit(train_df)

        label_encoder = LabelEncoder()
        label_encoder.fit(list(config.tabular.dx_classes))

        train_ds = MultimodalSkinDataset(
            train_df,
            img_dirs,
            preprocessor,
            label_encoder,
            transform=get_image_transforms(hp.image_size, is_training=augment_train),
        )
        val_ds = MultimodalSkinDataset(
            val_df,
            img_dirs,
            preprocessor,
            label_encoder,
            transform=get_image_transforms(hp.image_size, is_training=False),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=hp.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=hp.pin_memory,
        )

        yield fold_idx, train_loader, val_loader
