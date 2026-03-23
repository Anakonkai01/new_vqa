"""Shared inference-time decode policy for Model H."""

from __future__ import annotations

from typing import Optional

import torch


LENGTH_BIN_SHORT = 0
LENGTH_BIN_MEDIUM = 1
LENGTH_BIN_LONG = 2

_LENGTH_NAME_TO_BIN = {
    "short": LENGTH_BIN_SHORT,
    "medium": LENGTH_BIN_MEDIUM,
    "long": LENGTH_BIN_LONG,
}

# Derived from the dominant full-target length bin on the official val sets.
_DATASET_LENGTH_BIN = {
    "vqa_e": LENGTH_BIN_MEDIUM,
    "vqa_x": LENGTH_BIN_MEDIUM,
    "aokvqa": LENGTH_BIN_MEDIUM,
}

# Conservative per-dataset lower bounds based on the lower tail of full target lengths.
_DATASET_MIN_DECODE_LEN = {
    "vqa_e": 8,
    "vqa_x": 6,
    "aokvqa": 6,
}


def resolve_length_bin(length_policy: Optional[str], dataset_name: Optional[str] = None) -> int:
    policy = (length_policy or "auto").lower()
    ds_name = (dataset_name or "").lower()

    if policy in {"auto", "dataset_mode"}:
        return _DATASET_LENGTH_BIN.get(ds_name, LENGTH_BIN_LONG)
    if policy not in _LENGTH_NAME_TO_BIN:
        raise ValueError(f"Unsupported infer length policy: {length_policy!r}")
    return _LENGTH_NAME_TO_BIN[policy]


def build_length_bin_tensor(
    batch_size: int,
    device,
    *,
    length_policy: Optional[str] = "auto",
    dataset_name: Optional[str] = None,
) -> torch.Tensor:
    length_bin = resolve_length_bin(length_policy, dataset_name)
    return torch.full((batch_size,), length_bin, dtype=torch.long, device=device)


def resolve_min_decode_len(
    min_decode_len: Optional[int],
    dataset_name: Optional[str] = None,
) -> int:
    if min_decode_len is not None:
        return int(min_decode_len)
    return _DATASET_MIN_DECODE_LEN.get((dataset_name or "").lower(), 8)
