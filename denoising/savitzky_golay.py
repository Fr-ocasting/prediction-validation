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

from .utils import BaseDenoiser
from scipy.signal import savgol_filter
import numpy as np


class SavitzkyGolay(BaseDenoiser):
    """Causal Savitzky–Golay filter using scipy implementation.
    
    This filter smooths time series data by fitting local polynomials using
    the least-squares method, but only considers past values (causal).
    """

    name: Final[str] = "savitzky_golay"

    def __init__(self, window: int = 5, poly: int = 2) -> None:
        """
        Args:
            window: Length of the filter window. Must be odd and ≥ 3.
            poly: Order of the polynomial used to fit the samples. Must be < window.
        """
        
        if window % 2 == 0 or window < 3:
            raise ValueError("`window` must be odd and ≥ 3.")
        if poly >= window:
            raise ValueError("`poly` < `window` is required.")
        
        self.window = window
        self.poly = poly
        self.filter = lambda x: savgol_filter(x, window, poly, deriv=0, delta=1.0, mode='interp', axis=0)

    def __call__(self, series: torch.Tensor) -> torch.Tensor:
        """Apply causal Savitzky-Golay filter to time series data.
        
        Args:
            series: Input tensor of shape [T, N] or [T, N, C]
                   where T is time, N is batch size, C is channels
                   
        Returns:
            Filtered tensor with the same shape as input
        """
        
        # Convert to numpy for scipy processing
        was_cuda = series.is_cuda
        device = series.device
        numpy_data = series.cpu().numpy()
        
        # Apply filter - works directly on numpy arrays
        filtered_data = self.filter(numpy_data)
        
        # Convert back to tensor
        result = torch.from_numpy(filtered_data)
        
        # Return to original device
        if was_cuda:
            result = result.to(device)
            
        return result
