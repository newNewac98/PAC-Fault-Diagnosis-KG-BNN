"""
Evaluation utilities — Precision, Recall, F1-score.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from config import BATCH_SIZE, FAULT_NAMES


def evaluate_model(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    print_report: bool = False,
) -> dict:
    """
    Evaluate the trained BNN on the test set.
    Returns dict with macro Precision, Recall, F1.
    """
    model.eval()
    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if print_report:
        print(classification_report(
            y_true, y_pred,
            target_names=FAULT_NAMES,
            digits=4,
            zero_division=0,
        ))

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_sklearn_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate a fitted sklearn / xgboost / lightgbm model."""
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}
