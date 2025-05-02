import torch

class BaseDenoiser:
     """Common interface for all denoising algorithms."""

     name: str = "base"

     def __call__(self, series: torch.Tensor) -> torch.Tensor:  # pragma: no cover
          raise NotImplementedError