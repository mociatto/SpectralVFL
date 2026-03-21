"""
Centralized configuration for SpectralVFL.
IEEE TDSC - Inference-time frequency evasion attack against Multi-Modal VFL.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# Dataset path specs (registry entries)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetPathsSpec:
    """
    Relative layout under a dataset *data root* (folder that contains metadata + image dirs).
    """

    dataset_folder: str  # Subfolder name under kaggle_input when using default resolution
    metadata_filename: str
    image_dir_part1: str
    image_dir_part2: str


def _default_dataset_registry() -> Dict[str, DatasetPathsSpec]:
    """Registry: dataset_key -> path layout. Add new datasets here."""
    return {
        "ham10000": DatasetPathsSpec(
            dataset_folder="skin-cancer-mnist-ham10000",
            metadata_filename="HAM10000_metadata.csv",
            image_dir_part1="HAM10000_images_part_1",
            image_dir_part2="HAM10000_images_part_2",
        ),
        # Placeholder for future BCN20000 integration (paths TBD)
        "bcn20000": DatasetPathsSpec(
            dataset_folder="",
            metadata_filename="",
            image_dir_part1="",
            image_dir_part2="",
        ),
    }


@dataclass(frozen=True)
class PathsConfig:
    """
    Kaggle-compatible paths and multi-dataset registry.

    Use ``datasets`` / ``get_dataset()`` for explicit multi-dataset support.
    Properties ``metadata_filename``, ``image_dir_part1``, etc. delegate to
    ``default_dataset_key`` for backward compatibility with ``data_utils``.
    """

    kaggle_input: str = "/kaggle/input"
    kaggle_working: str = "/kaggle/working"
    datasets: Dict[str, DatasetPathsSpec] = field(default_factory=_default_dataset_registry)
    default_dataset_key: str = "ham10000"

    def get_dataset(self, key: Optional[str] = None) -> DatasetPathsSpec:
        """Return path spec for a dataset key (default: ``default_dataset_key``)."""
        k = key or self.default_dataset_key
        if k not in self.datasets:
            raise KeyError(f"Unknown dataset key '{k}'. Available: {list(self.datasets.keys())}")
        return self.datasets[k]

    # Backward compatibility: same as legacy PathsConfig fields for HAM10000
    @property
    def dataset_name(self) -> str:
        return self.get_dataset().dataset_folder

    @property
    def metadata_filename(self) -> str:
        return self.get_dataset().metadata_filename

    @property
    def image_dir_part1(self) -> str:
        return self.get_dataset().image_dir_part1

    @property
    def image_dir_part2(self) -> str:
        return self.get_dataset().image_dir_part2

    def get_metadata_path(self, data_root: Optional[Union[str, Path]] = None) -> Path:
        """Resolve metadata CSV path (uses default dataset)."""
        root = Path(data_root) if data_root else Path(self.kaggle_input) / self.dataset_name
        return root / self.metadata_filename

    def get_image_dirs(self, data_root: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Resolve image directory paths (uses default dataset)."""
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
class TrainConfig:
    """Training hyperparameters (backend). Front-end should read from ``config.train``."""

    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8  # early stopping on val_loss


@dataclass(frozen=True)
class HAM10000TabularConfig:
    """
    Expected tabular features for HAM10000 metadata.
    Schema: lesion_id, image_id, dx, dx_type, age, sex, localization
    """

    lesion_id_col: str = "lesion_id"
    image_id_col: str = "image_id"
    label_col: str = "dx"
    dx_type_col: str = "dx_type"
    age_col: str = "age"
    sex_col: str = "sex"
    localization_col: str = "localization"

    numeric_features: Tuple[str, ...] = ("age",)
    categorical_features: Tuple[str, ...] = ("sex", "localization")

    dx_classes: Tuple[str, ...] = (
        "akiec",
        "bcc",
        "bkl",
        "df",
        "mel",
        "nv",
        "vasc",
    )

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
    train: TrainConfig = field(default_factory=TrainConfig)
    tabular: HAM10000TabularConfig = field(default_factory=HAM10000TabularConfig)

    @property
    def dataset_paths(self) -> Dict[str, DatasetPathsSpec]:
        """Convenience: read-only view of the dataset registry."""
        return dict(self.paths.datasets)


# Singleton config instance
config = Config()
