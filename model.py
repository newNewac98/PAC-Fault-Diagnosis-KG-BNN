"""
Bayesian Neural Network with Evidence Framework for fault classification.

The loss combines:
  - Cross-entropy (scaled by β) for data-fit
  - L2 weight regularisation (scaled by α/2) as Gaussian weight prior
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from config import ALPHA, BETA, NUM_FEATURES, NUM_CLASSES


class BayesianNN(nn.Module):
    """
    Feed-forward BNN:  Input(30) → 128 → 64 → 4
    """

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
        alpha: float = ALPHA,
        beta: float = BETA,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        # Xavier initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── Evidence-framework loss ─────────────────────────────────────────
    def bayesian_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        L = β * CE(logits, targets)  +  (α / 2) * Σ w²
        """
        ce_loss = nn.functional.cross_entropy(logits, targets)
        data_term = self.beta * ce_loss

        # Weight-decay (Gaussian prior) term
        reg_term = torch.tensor(0.0, device=logits.device)
        for param in self.parameters():
            reg_term = reg_term + (param ** 2).sum()
        reg_term = (self.alpha / 2.0) * reg_term

        return data_term + reg_term
