"""Cached (state, label) dataset for predictor training.

Shards on disk are PyTorch save() files with this schema:

    {
        "scalars":     FloatTensor [N, n_scalars + n_block_size_onehot],
        "hidden_pool": FloatTensor [N, hidden_dim],
        "labels":      LongTensor  [N],   # index into CANDIDATE_BLOCK_SIZES
        "prompt_ids":  LongTensor  [N],
        "meta": {
            "model": "llada" | "dream",
            "label_source": "teacher" | "oracle",
            "benchmark": str,
            "candidate_block_sizes": [4, 8, 16, 32],
            "hidden_dim": int,
        }
    }

The dataset reads multiple shards, holds out a fraction of prompt-ids for
validation (so no leakage between train and val splits), and yields
random-access items.
"""

from __future__ import annotations

import glob
import hashlib
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _prompt_in_val(prompt_id: int, val_frac: float, salt: str = "v2-2026") -> bool:
    """Deterministic prompt-id-keyed split. Hash the (salt, id) pair to a uniform
    in [0, 1) and route the smallest val_frac to validation. Stable across
    shard layouts."""
    h = hashlib.sha256(f"{salt}:{prompt_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return u < val_frac


class PredictorDataset(Dataset):
    def __init__(
        self,
        shard_glob: str,
        split: str = "train",
        val_frac: float = 0.1,
        max_examples: Optional[int] = None,
    ) -> None:
        assert split in ("train", "val", "all")
        self.split = split
        self.val_frac = val_frac

        shard_paths = sorted(glob.glob(shard_glob))
        if not shard_paths:
            raise FileNotFoundError(f"no shards matched: {shard_glob}")

        scalars_chunks: List[torch.Tensor] = []
        hidden_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        prompt_id_chunks: List[torch.Tensor] = []
        meta_first: Optional[Dict] = None

        for sp in shard_paths:
            shard = torch.load(sp, map_location="cpu")
            scalars_chunks.append(shard["scalars"].to(torch.float32))
            hidden_chunks.append(shard["hidden_pool"].to(torch.float32))
            label_chunks.append(shard["labels"].to(torch.long))
            prompt_id_chunks.append(shard["prompt_ids"].to(torch.long))
            if meta_first is None:
                meta_first = shard.get("meta", {})

        scalars = torch.cat(scalars_chunks, dim=0)
        hidden = torch.cat(hidden_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        prompt_ids = torch.cat(prompt_id_chunks, dim=0)

        if split != "all":
            mask = torch.tensor(
                [_prompt_in_val(int(pid), val_frac) for pid in prompt_ids],
                dtype=torch.bool,
            )
            if split == "val":
                keep = mask
            else:
                keep = ~mask
            scalars = scalars[keep]
            hidden = hidden[keep]
            labels = labels[keep]
            prompt_ids = prompt_ids[keep]

        if max_examples is not None and len(labels) > max_examples:
            scalars = scalars[:max_examples]
            hidden = hidden[:max_examples]
            labels = labels[:max_examples]
            prompt_ids = prompt_ids[:max_examples]

        self.scalars = scalars
        self.hidden_pool = hidden
        self.labels = labels
        self.prompt_ids = prompt_ids
        self.meta = meta_first or {}

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.scalars[idx], self.hidden_pool[idx], self.labels[idx]

    def class_balance(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for v in self.labels.tolist():
            counts[int(v)] = counts.get(int(v), 0) + 1
        return counts


def class_weights(dataset: PredictorDataset, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy on imbalanced labels."""
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for v in dataset.labels.tolist():
        counts[int(v)] += 1
    counts = counts.clamp_min(1.0)
    w = counts.sum() / (n_classes * counts)
    return w
