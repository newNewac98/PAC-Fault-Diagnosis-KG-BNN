"""
Preprocessing: stratified train/val/test split + Min-Max normalisation.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, DATA_DIR

def load_dataset(data_dir: str = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features.csv and labels.csv from *data_dir*.

    To swap in real data, simply replace these two CSV files:
      - features.csv: columns = sensor_0 … sensor_19, kg_0 … kg_9
      - labels.csv:   single column "label" with integer class ids (0-3)
    """
    features_path = os.path.join(data_dir, "features.csv")
    labels_path = os.path.join(data_dir, "labels.csv")

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}. "
            f"Run `python data_generator.py` first to create synthetic data, "
            f"or place your own features.csv and labels.csv there."
        )

    X = pd.read_csv(features_path).values.astype(np.float64)
    y = pd.read_csv(labels_path)["label"].values.astype(np.int64)

    print(f"Loaded dataset from {data_dir}: X={X.shape}, y={y.shape}")
    return X, y


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple:
    """
    1. Stratified split 70 / 15 / 15.
    2. Min-Max scale continuous features to [0, 1].
    3. Return six PyTorch tensors:
       (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # First split: train (70 %) vs. temp (30 %)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=seed,
    )

    # Second split: val (50 % of temp → 15 %) vs. test (50 % of temp → 15 %)
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_frac),
        stratify=y_temp,
        random_state=seed,
    )

    # ── Min-Max Normalisation (fit on train only) ───────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── Convert to PyTorch tensors ──────────────────────────────────────
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_data_numpy(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple:
    """Same split + scaling but returns NumPy arrays (for sklearn baselines)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=seed,
    )

    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_frac),
        stratify=y_temp,
        random_state=seed,
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
