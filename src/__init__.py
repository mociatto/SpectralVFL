"""SpectralVFL - Inference-time frequency evasion attack against Multi-Modal VFL."""

from .attacks import (
    AdaptiveSpectralPGD,
    BaseEmbeddingAttack,
    SpatialFGSM,
    SpatialPGD,
    adaptive_spectral_filter_gradient,
    denormalize_to_01,
    normalize_from_01,
)
from .config import DatasetPathsSpec, ExperimentConfig, TrainConfig, config
from .data_utils import (
    MultimodalSkinDataset,
    TabularPreprocessor,
    get_dataloaders,
    get_image_transforms,
    get_kfold_dataloaders,
    stratified_group_split,
)
from .models import ImageClient, TabularClient, VFLServer, get_vfl_system
from .metrics import compute_attack_success_rate, compute_stealth_metrics
from .trainer import run_kfold_vfl_training
from .training import (
    EarlyStopping,
    compute_class_weights,
    evaluate_vfl,
    generate_evaluation_report,
    get_trainable_params,
    train_vfl_epoch,
    train_vfl_system,
)

__all__ = [
    "AdaptiveSpectralPGD",
    "BaseEmbeddingAttack",
    "SpatialFGSM",
    "SpatialPGD",
    "adaptive_spectral_filter_gradient",
    "denormalize_to_01",
    "normalize_from_01",
    "compute_attack_success_rate",
    "compute_stealth_metrics",
    "config",
    "DatasetPathsSpec",
    "ExperimentConfig",
    "TrainConfig",
    "MultimodalSkinDataset",
    "TabularPreprocessor",
    "get_dataloaders",
    "get_image_transforms",
    "get_kfold_dataloaders",
    "stratified_group_split",
    "ImageClient",
    "TabularClient",
    "VFLServer",
    "get_vfl_system",
    "EarlyStopping",
    "compute_class_weights",
    "evaluate_vfl",
    "generate_evaluation_report",
    "get_trainable_params",
    "train_vfl_epoch",
    "train_vfl_system",
    "run_kfold_vfl_training",
]
