"""
Training and validation engine for SpectralVFL.
Handles class weighting, optimizer setup, and epoch loops.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import config
from .models import ImageClient, TabularClient, VFLServer


# -----------------------------------------------------------------------------
# Class Weighting
# -----------------------------------------------------------------------------


def compute_class_weights(
    train_df: pd.DataFrame,
    label_col: str,
    classes: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced HAM10000 using sklearn.

    Args:
        train_df: Training dataframe with labels.
        label_col: Column name containing class labels (e.g. 'dx').
        classes: Ordered list of class labels. If None, uses config.tabular.dx_classes.
                 Must match the order used by the label encoder for correct indexing.

    Returns:
        PyTorch tensor of shape (num_classes,) with dtype float32.
    """
    classes = classes or list(config.tabular.dx_classes)
    y = train_df[label_col].values

    # Filter to only classes present in y to avoid division by zero
    classes_in_data = np.unique(y)
    classes_in_data = [c for c in classes if c in classes_in_data]

    if len(classes_in_data) == 0:
        raise ValueError("No valid classes found in train_df.")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes_in_data),
        y=y,
    )

    # Build full weight tensor in order of `classes` (model output order)
    weight_map = dict(zip(classes_in_data, weights))
    full_weights = np.array(
        [weight_map.get(c, 1.0) for c in classes],
        dtype=np.float32,
    )

    # Replace inf/nan (e.g. from zero-count classes) with 1.0
    full_weights = np.nan_to_num(full_weights, nan=1.0, posinf=1.0, neginf=1.0)

    return torch.tensor(full_weights, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Optimizer Setup
# -----------------------------------------------------------------------------


def get_trainable_params(
    image_client: ImageClient,
    tabular_client: TabularClient,
    vfl_server: VFLServer,
) -> List[torch.nn.Parameter]:
    """Collect trainable parameters from all three VFL components."""
    all_params = list(image_client.parameters()) + list(tabular_client.parameters()) + list(vfl_server.parameters())
    return [p for p in all_params if p.requires_grad]


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------


def train_vfl_epoch(
    image_client: ImageClient,
    tabular_client: TabularClient,
    vfl_server: VFLServer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one training epoch.

    Returns:
        Mean epoch loss.
    """
    image_client.train()
    tabular_client.train()
    vfl_server.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Train", leave=False)

    for images, tabular, labels in pbar:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        img_emb = image_client(images)
        tab_emb = tabular_client(tabular)
        logits = vfl_server(img_emb, tab_emb)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


# -----------------------------------------------------------------------------
# Validation Loop
# -----------------------------------------------------------------------------


def evaluate_vfl(
    image_client: ImageClient,
    tabular_client: TabularClient,
    vfl_server: VFLServer,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate model on validation/test set.

    Returns:
        (loss, accuracy, balanced_accuracy)
    """
    image_client.eval()
    tabular_client.eval()
    vfl_server.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, tabular, labels in tqdm(data_loader, desc="Eval", leave=False):
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            img_emb = image_client(images)
            tab_emb = tabular_client(tabular)
            logits = vfl_server(img_emb, tab_emb)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / max(num_batches, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    correct = (all_preds == all_labels).sum()
    accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return float(mean_loss), float(accuracy), float(balanced_acc)


# -----------------------------------------------------------------------------
# Main Routine
# -----------------------------------------------------------------------------


def train_vfl_system(
    image_client: ImageClient,
    tabular_client: TabularClient,
    vfl_server: VFLServer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    label_col: str = "dx",
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Orchestrate the full training pipeline.

    Args:
        image_client: Image client model.
        tabular_client: Tabular client model.
        vfl_server: VFL server model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        train_df: Training dataframe (for class weights).
        label_col: Label column name.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        save_path: Path to save best checkpoint. Default: /kaggle/working/best_vfl_model.pth
        device: Device to run on. Default: cuda if available else cpu.

    Returns:
        history: Dict with keys 'train_loss', 'val_loss', 'val_acc', 'val_balanced_acc'.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    save_path = Path(save_path) if save_path else Path(config.paths.kaggle_working) / "best_vfl_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Move models to device
    image_client = image_client.to(device)
    tabular_client = tabular_client.to(device)
    vfl_server = vfl_server.to(device)

    # Class weights
    class_weights = compute_class_weights(train_df, label_col).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer: only trainable parameters
    optimizer = torch.optim.AdamW(
        get_trainable_params(image_client, tabular_client, vfl_server),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
    }
    best_balanced_acc = 0.0

    for epoch in range(num_epochs):
        train_loss = train_vfl_epoch(
            image_client,
            tabular_client,
            vfl_server,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_acc, val_balanced_acc = evaluate_vfl(
            image_client,
            tabular_client,
            vfl_server,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)

        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            torch.save(
                {
                    "image_client": image_client.state_dict(),
                    "tabular_client": tabular_client.state_dict(),
                    "vfl_server": vfl_server.state_dict(),
                },
                save_path,
            )
            print(f"Epoch {epoch + 1}/{num_epochs} | New best Balanced Acc: {val_balanced_acc:.4f} | Saved to {save_path}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Balanced Acc: {val_balanced_acc:.4f}"
        )

    return history
