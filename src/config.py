"""
Centralized configuration for SpectralVFL.
IEEE TDSC - Inference-time frequency evasion attack against Multi-Modal VFL.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


@dataclass(frozen=True)
class PathsConfig:
    """Kaggle-compatible dataset paths. Update for local/Kaggle execution."""

    # Kaggle input/output (placeholders - configure per environment)
    kaggle_input: str = "/kaggle/input"
    kaggle_working: str = "/kaggle/working"

    # HAM10000 dataset paths (relative to kaggle_input or local data root)
    # Typical Kaggle: skin-cancer-mnist-ham10000
    dataset_name: str = "skin-cancer-mnist-ham10000"
    metadata_filename: str = "HAM10000_metadata.csv"
    image_dir_part1: str = "HAM10000_images_part_1"
    image_dir_part2: str = "HAM10000_images_part_2"

    def get_metadata_path(self, data_root: Optional[Union[str, Path]] = None) -> Path:
        """Resolve metadata CSV path."""
        root = Path(data_root) if data_root else Path(self.kaggle_input) / self.dataset_name
        return root / self.metadata_filename

    def get_image_dirs(self, data_root: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Resolve image directory paths (part1, part2)."""
        root = Path(data_root) if data_root else Path(self.kaggle_input) / self.dataset_name
        return root / self.image_dir_part1, root / self.image_dir_part2


@dataclass(frozen=True)
class HyperparametersConfig:
    """Standard hyperparameters for training and evaluation."""

    batch_size: int = 32
    random_seed: int = 42
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(frozen=True)
class HAM10000TabularConfig:
    """
    Expected tabular features for HAM10000 metadata.
    Schema: lesion_id, image_id, dx, dx_type, age, sex, localization
    """

    # Column names
    lesion_id_col: str = "lesion_id"
    image_id_col: str = "image_id"
    label_col: str = "dx"
    dx_type_col: str = "dx_type"
    age_col: str = "age"
    sex_col: str = "sex"
    localization_col: str = "localization"

    # Tabular feature columns used for model input (excludes IDs and label)
    numeric_features: Tuple[str, ...] = ("age",)
    categorical_features: Tuple[str, ...] = ("sex", "localization")

    # Diagnosis classes (HAM10000)
    dx_classes: Tuple[str, ...] = (
        "akiec",  # Actinic keratoses
        "bcc",    # Basal cell carcinoma
        "bkl",    # Benign keratosis
        "df",     # Dermatofibroma
        "mel",    # Melanoma
        "nv",     # Melanocytic nevi
        "vasc",   # Vascular lesions
    )

    # Categorical value mappings (for consistent encoding)
    sex_values: Tuple[str, ...] = ("female", "male", "unknown")
    localization_values: Tuple[str, ...] = (
        "abdomen", "acral", "back", "chest", "ear", "face", "foot",
        "genital", "hand", "lower extremity", "neck", "scalp",
        "trunk", "unknown", "upper extremity",
    )


@dataclass
class Config:
    """Unified configuration container."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    hyperparams: HyperparametersConfig = field(default_factory=HyperparametersConfig)
    tabular: HAM10000TabularConfig = field(default_factory=HAM10000TabularConfig)


# Singleton config instance
config = Config()
