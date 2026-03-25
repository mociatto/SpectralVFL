"""
K-fold cross-validation orchestration for SpectralVFL.
Keeps a single best-fold checkpoint on disk (selected by validation macro AUC).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .config import config
from .data_utils import get_kfold_dataloaders
from .models import get_vfl_system
from .training import generate_evaluation_report, train_vfl_system

logger = logging.getLogger(__name__)


def _load_vfl_checkpoint(
    path: Union[str, Path],
    device: torch.device,
    image_client,
    tabular_client,
    vfl_server,
) -> None:
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    image_client.load_state_dict(ckpt["image_client"])
    tabular_client.load_state_dict(ckpt["tabular_client"])
    vfl_server.load_state_dict(ckpt["vfl_server"])


def _val_metrics_from_report(
    image_client,
    tabular_client,
    vfl_server,
    val_loader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return (Accuracy %, Macro AUC %, Macro-F1 %) on the validation loader."""
    report = generate_evaluation_report(
        image_client,
        tabular_client,
        vfl_server,
        val_loader,
        device,
        num_classes=len(config.tabular.dx_classes),
    )
    row = report.iloc[0]
    acc = float(row["Accuracy (%)"])
    auc = float(row["Macro AUC-ROC (%)"])
    f1 = float(row["Macro-F1 (%)"])
    if auc is None or (isinstance(auc, float) and np.isnan(auc)):
        auc = 0.0
    return acc, auc, f1


def _fmt_mean_std(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{m:.1f} ± {s:.1f}"


def run_kfold_vfl_training(
    df: pd.DataFrame,
    label_col: str,
    image_dirs: Union[Tuple[Path, Path], List[Path]],
    model_name: str,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    random_state: int = 42,
    best_checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Stratified K-fold training; saves **only** the checkpoint from the fold with
    highest validation macro AUC.

    Returns formatted summary strings for Acc, AUC, and Macro-F1 (mean ± std over folds).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    out_path = (
        Path(best_checkpoint_path)
        if best_checkpoint_path
        else Path(config.paths.kaggle_working) / f"best_vfl_kfold_{model_name}.pth"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="svfl_kfold_")
    fold_ckpts: List[Path] = []
    accs: List[float] = []
    aucs: List[float] = []
    f1s: List[float] = []

    try:
        for fold_idx, train_loader, val_loader in get_kfold_dataloaders(
            df,
            label_col,
            k=k,
            batch_size=batch_size,
            image_dirs=image_dirs,
            num_workers=num_workers,
            random_state=random_state,
            augment_train=True,
        ):
            train_df = train_loader.dataset.df
            tab_dim = train_loader.dataset.preprocessor.tabular_dim

            image_client, tabular_client, vfl_server = get_vfl_system(
                model_name=model_name,
                tabular_dim=tab_dim,
                pretrained=True,
            )

            fold_ckpt = Path(tmpdir) / f"fold_{fold_idx}.pth"
            train_vfl_system(
                image_client,
                tabular_client,
                vfl_server,
                train_loader,
                val_loader,
                train_df,
                label_col=label_col,
                save_path=fold_ckpt,
                save_checkpoint=True,
                verbose=False,
                device=device,
            )

            _load_vfl_checkpoint(fold_ckpt, device, image_client, tabular_client, vfl_server)
            acc, auc, f1 = _val_metrics_from_report(
                image_client,
                tabular_client,
                vfl_server,
                val_loader,
                device,
            )

            fold_ckpts.append(fold_ckpt)
            accs.append(acc)
            aucs.append(auc)
            f1s.append(f1)
            logger.info(
                "Fold %d/%d | val Acc=%.2f%% AUC=%.2f%% F1=%.2f%%",
                fold_idx + 1,
                k,
                acc,
                auc,
                f1,
            )

            del image_client, tabular_client, vfl_server
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not fold_ckpts or not aucs:
            return {
                "Clean_Acc": "nan ± nan",
                "Clean_AUC": "nan ± nan",
                "Clean_F1": "nan ± nan",
            }

        best_i = int(np.argmax(np.asarray(aucs)))
        shutil.copy(fold_ckpts[best_i], out_path)
        logger.info(
            "Best fold by val macro AUC: %d (AUC=%.2f%%). Checkpoint: %s",
            best_i,
            aucs[best_i],
            out_path,
        )

        return {
            "Clean_Acc": _fmt_mean_std(accs),
            "Clean_AUC": _fmt_mean_std(aucs),
            "Clean_F1": _fmt_mean_std(f1s),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
