"""Block-size predictor MLP.

Maps a state vector (scalars + hidden-state pool) to a distribution over
the K candidate block sizes. Optionally a regression head over log block
size for future continuous extension.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .features import CANDIDATE_BLOCK_SIZES, n_scalar_features


class BlockSizePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int = 256,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        n_classes: int = len(CANDIDATE_BLOCK_SIZES),
        with_regression_head: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.n_classes = n_classes
        self.with_regression_head = with_regression_head

        in_dim = hidden_dim + n_scalar_features()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.body = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(mlp_hidden, n_classes)
        if with_regression_head:
            self.reg_head = nn.Linear(mlp_hidden, 1)
        else:
            self.reg_head = None

    def forward(
        self,
        scalars: torch.Tensor,
        hidden_pool: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scalars: [B, n_scalar_features()]
            hidden_pool: [B, hidden_dim]

        Returns:
            logits: [B, n_classes]
            reg: [B, 1] log-block-size prediction (zeros if regression head off)
        """
        x = torch.cat([scalars, hidden_pool], dim=-1)
        x = self.proj(x)
        x = self.body(x)
        logits = self.cls_head(x)
        if self.reg_head is not None:
            reg = self.reg_head(x)
        else:
            reg = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return logits, reg

    def predict_block_size(
        self,
        scalars: torch.Tensor,
        hidden_pool: torch.Tensor,
    ) -> int:
        """Single-example argmax prediction. Inputs may be 1-D."""
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
        if hidden_pool.dim() == 1:
            hidden_pool = hidden_pool.unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(scalars, hidden_pool)
            idx = int(logits.argmax(dim=-1).item())
        return CANDIDATE_BLOCK_SIZES[idx]


def save_predictor(predictor: BlockSizePredictor, path: str) -> None:
    payload = {
        "state_dict": predictor.state_dict(),
        "config": {
            "hidden_dim": predictor.hidden_dim,
            "proj_dim": predictor.proj_dim,
            "n_classes": predictor.n_classes,
            "with_regression_head": predictor.with_regression_head,
        },
    }
    torch.save(payload, path)


def load_predictor(path: str, map_location: str = "cpu") -> BlockSizePredictor:
    payload = torch.load(path, map_location=map_location)
    cfg = payload["config"]
    model = BlockSizePredictor(
        hidden_dim=cfg["hidden_dim"],
        proj_dim=cfg.get("proj_dim", 256),
        n_classes=cfg.get("n_classes", len(CANDIDATE_BLOCK_SIZES)),
        with_regression_head=cfg.get("with_regression_head", False),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
