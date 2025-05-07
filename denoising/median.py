# denoising/median.py
"""Causal median filter implemented in PyTorch."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

from .utils import BaseDenoiser

class MedianFilter(BaseDenoiser):
    """Causal median filter with odd-sized sliding window."""

    name: Final[str] = "median"

    def __init__(self, kernel_size: int = 3, agg_func = 'median', mode = 'causal') -> None:
        self.k = kernel_size
        self.agg_func = agg_func
        self.mode = mode

    # ---------------------------------------------------------------------

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        """Apply the median filter along the time axis (axis 0)."""
        if self.mode == 'causal':
            pad = self.k - 1  # causal: shift towards the past
        elif self.mode == 'centered':
            pad = self.k // 2
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'causal' or 'centered'.")
        # Add a fictitious "channel" dimension for conv1d/unfold
        x = series.transpose(0, 1)  # [N,(C,)T] -> [N, T,(C)]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N,1,T]
        else:  # [N,C,T]
            x = x.transpose(1, 2)  # [N,T,C]
        if self.mode == 'causal':
            x = F.pad(x, (pad, 0), mode="replicate")
        if self.mode == 'centered':
            x = F.pad(x, (pad, pad), mode="replicate")
        # Unfold to get window of size k
        x_unfold = x.unfold(dimension=2, size=self.k, step=1)  # [...,T,k]

        if self.agg_func == 'median':
            median_x = x_unfold.median(dim=-1).values
        elif self.agg_func == 'mean':
            median_x = x_unfold.mean(dim=-1)
        # Restore the initial shape
        if series.dim() == 2:
            return median_x.squeeze(1).transpose(0, 1)  # [T,N]
        return median_x.transpose(1, 2).transpose(0, 1)  # [T,N,C]
