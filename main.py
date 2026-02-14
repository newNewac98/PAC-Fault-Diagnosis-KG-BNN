"""
Main driver — 5-fold independent runs producing Table 2.

For each fold the SAME dataset (loaded from data/) is re-split with a
different seed, the BNN is trained from scratch, and all 6 baselines
are fitted.  Final output: Mean ± Std of Precision, Recall, F1.

Prerequisites:
    python data_generator.py   # generate synthetic data (or place real CSVs)
    python main.py             # run experiments
"""

import numpy as np
import torch

from config import SEED, N_FOLDS, DEVICE, FAULT_NAMES
from preprocessing import prepare_data, prepare_data_numpy,load_dataset
from model import BayesianNN
from train import train_model
from evaluate import evaluate_model
from baselines import train_and_evaluate_baselines


def set_seed(seed: int):
    """Ensure reproducibility for a given fold."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_fold(X: np.ndarray, y: np.ndarray, fold: int) -> dict:
    """Run one fold and return metrics for all models."""
    fold_seed = SEED + fold
    set_seed(fold_seed)
    print(f"\n{'='*60}")
    print(f"  Fold {fold}/{N_FOLDS}  (seed={fold_seed})")
    print(f"{'='*60}")

    # ── Load dataset from disk ──────────────────────────────────────────
    X, y = load_dataset()
    # ── Preprocess (split + scale) ──────────────────────────────────────
    # PyTorch tensors for BNN
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(X, y, seed=fold_seed)
    # NumPy arrays for baselines
    X_train_np, y_train_np, _, _, X_test_np, y_test_np = prepare_data_numpy(X, y, seed=fold_seed)

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Device: {DEVICE}")

    # ── BNN ─────────────────────────────────────────────────────────────
    print("\n  [KG + BNN]")
    bnn = BayesianNN()
    train_model(bnn, X_train, y_train, X_val, y_val, device=DEVICE)
    bnn_metrics = evaluate_model(bnn, X_test, y_test, device=DEVICE, print_report=True)
    print(f"  → P={bnn_metrics['precision']:.4f}  "
          f"R={bnn_metrics['recall']:.4f}  F1={bnn_metrics['f1']:.4f}")

    # ── Baselines ───────────────────────────────────────────────────────
    print("\n  [Baselines]")
    baseline_metrics = train_and_evaluate_baselines(
        X_train_np, y_train_np, X_test_np, y_test_np, seed=fold_seed,
    )
    for name, m in baseline_metrics.items():
        print(f"    {name:<15s}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}")

    # Merge
    all_metrics = {"KG + BNN": bnn_metrics}
    all_metrics.update(baseline_metrics)
    return all_metrics


def print_table2(results_per_fold: list[dict]):
    """Print Table 2: Mean ± Std for every model."""
    model_names = list(results_per_fold[0].keys())

    print("\n")
    print("=" * 78)
    print("  TABLE 2 — Performance Comparison (Mean ± Std over 5 folds)")
    print("=" * 78)
    header = f"  {'Model':<20s}  {'Precision':>16s}  {'Recall':>16s}  {'F1-score':>16s}"
    print(header)
    print("-" * 78)

    for name in model_names:
        precs = [fold[name]["precision"] for fold in results_per_fold]
        recs = [fold[name]["recall"] for fold in results_per_fold]
        f1s = [fold[name]["f1"] for fold in results_per_fold]

        p_str = f"{np.mean(precs):.4f} ± {np.std(precs):.4f}"
        r_str = f"{np.mean(recs):.4f} ± {np.std(recs):.4f}"
        f_str = f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        print(f"  {name:<20s}  {p_str:>16s}  {r_str:>16s}  {f_str:>16s}")

    print("=" * 78)


def main():
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device selected: {DEVICE}")
    print(f"Fault classes: {FAULT_NAMES}")

    # ── Load dataset from disk ──────────────────────────────────────────
    X, y = load_dataset()

    results_per_fold = []
    for fold in range(1, N_FOLDS + 1):
        fold_results = run_fold(X, y, fold)
        results_per_fold.append(fold_results)

    print_table2(results_per_fold)


if __name__ == "__main__":
    main()
