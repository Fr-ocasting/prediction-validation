# denoising/exponential.py
"""Causal exponential smoothing (simple smoothing) for time series."""

from __future__ import annotations

from typing import Final

import torch

from .denoising import BaseDenoiser


class ExponentialSmoother(BaseDenoiser):
    """Causal smoothing: ŷₜ = α·xₜ + (1‑α)·ŷₜ₋₁."""

    name: Final[str] = "exponential"

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("`alpha` ∈ (0,1].")
        self.alpha = alpha

    # ---------------------------------------------------------------------

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        """Applies smoothing along axis 0 (time)."""
        y = series.clone()
        for t in range(1, series.size(0)):
            y[t] = self.alpha * series[t] + (1 - self.alpha) * y[t - 1]
        return y
