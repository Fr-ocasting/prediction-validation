# denoising/median.py
"""Causal median filter implemented in PyTorch."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

from .denoising import BaseDenoiser


class MedianFilter(BaseDenoiser):
    """Causal median filter, sliding window of odd size."""

    name: Final[str] = "median"

    def __init__(self, kernel_size: int = 3) -> None:
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("`kernel_size` must be a positive odd integer.")
        self.k = kernel_size

    # ---------------------------------------------------------------------

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        """Apply the median filter along the time axis (axis 0)."""
        pad = self.k - 1  # causal: shifted toward the past
        # Add a fictitious "channel" dimension for conv1d/unfold
        x = series.transpose(0, 1)  # [N,(C,)T] -> [N, T,(C)]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N,1,T]
        else:  # [N,C,T]
            x = x.transpose(1, 2)  # [N,T,C]

        x = F.pad(x, (pad, 0), mode="replicate")
        # Unfold to get the window of size k
        x_unfold = x.unfold(dimension=2, size=self.k, step=1)  # [...,T,k]
        median = x_unfold.median(dim=-1).values
        # Restore the initial shape
        if series.dim() == 2:
            return median.squeeze(1).transpose(0, 1)  # [T,N]
        return median.transpose(1, 2).transpose(0, 1)  # [T,N,C]
