# denoising/savitzky_golay_causal.py
"""Savitzky–Golay *causal* filter implemented in PyTorch.

The filter is obtained by solving a least squares system
on the past window only: it never looks at the future,
which avoids any information leakage.
"""

from __future__ import annotations

from math import factorial
from typing import Final

import torch

from .denoising import BaseDenoiser


def _savgol_coeffs(window: int, poly: int) -> torch.Tensor:
    """Causal filter coefficients (convolution column)."""
    # Vandermonde matrix (time 0 to -(window-1))
    t = torch.arange(0, -window, -1.0)
    A = torch.stack([t ** i for i in range(poly + 1)], dim=1)  # [window, poly+1]
    # Solves min ||A·c - e₀||₂  (e₀: vector [1,0,0,...])
    e0 = torch.zeros(window)
    e0[0] = 1.0
    c, _ = torch.lstsq(e0.unsqueeze(1), A)  # (poly+1,1)
    h = (A @ c).squeeze()  # [window]
    return h.flip(0)  # causal: we flip for direct convolution


class SavitzkyGolayCausal(BaseDenoiser):
    """Causal Savitzky–Golay filter."""

    name: Final[str] = "savitzky_golay_causal"

    def __init__(self, window: int = 5, poly: int = 2) -> None:
        if window % 2 == 0 or window < 3:
            raise ValueError("`window` must be odd and ≥ 3.")
        if poly >= window:
            raise ValueError("`poly` < `window` is required.")
        self.window = window
        self.poly = poly
        self.coeffs = _savgol_coeffs(window, poly).view(1, 1, -1)  # [1,1,L]

    # ---------------------------------------------------------------------

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        """Causal 1-D convolution on axis 0 (time)."""
        x = series.transpose(0, 1)  # [N,(C),T] -> [N,T,(C)]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N,1,T]
        else:
            x = x.transpose(1, 2)  # [N,C,T]

        pad = self.window - 1
        x = torch.nn.functional.pad(x, (pad, 0), mode="replicate")
        y = torch.nn.functional.conv1d(x, self.coeffs.to(x), groups=x.size(1))
        if series.dim() == 2:
            return y.squeeze(1).transpose(0, 1)
        return y.transpose(1, 2).transpose(0, 1)
