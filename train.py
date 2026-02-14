"""
Training loop for the BNN model.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, DEVICE


def train_model(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device = DEVICE,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> dict:
    """
    Train the BNN, returning a history dict with train/val losses per epoch.
    """
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = model.bayesian_loss(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / n_batches

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += model.bayesian_loss(logits, yb).item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>3d}/{epochs}  "
                  f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    return history
