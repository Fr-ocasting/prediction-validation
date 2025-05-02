# denoising/denoising.py
"""Main orchestrator of the denoising pipeline.

The logic is as follows:
1. Choose which *training_modes* will undergo denoising (by default: ``["train"]``).
2. For each *dataset* listed in ``args.denoising_names``:
    • we instantiate the requested denoiser(s);
    • we apply the algorithm *in-place* on `dataset.raw_values`
      at indices corresponding to the chosen *training_mode*.

By convention, normalization statistics are calculated AFTER
denoising on the train data only.
This option avoids any bias: the model will see, during training
and inference, centered and scaled data in the same space as
the one used for weight optimization.
"""

from __future__ import annotations

import inspect
from typing import Dict, Iterable, List, Sequence, Type

import torch

from .median import MedianFilter
from .exponential import ExponentialSmoother
from .savitzky_golay import SavitzkyGolay
from .utils import BaseDenoiser

# ---------------------------------------------------------------------------





# Register of available denoisers.
_DENOISERS: Dict[str, Type[BaseDenoiser]] = {
     cls.name: cls
     for cls in (MedianFilter, ExponentialSmoother, SavitzkyGolay)
}


# ---------------------------------------------------------------------------


class DenoisingManager:
     """Applies one or more denoising methods on one or more datasets."""

     def __init__(
          self,
          denoiser_names: Sequence[str],
          *,
          training_mode: str | None = None,
          denoiser_kwargs: Dict[str, dict] | None = None,
     ) -> None:
          """
          Parameters
          ----------
          denoiser_names :
                List of algorithms to chain (e.g. ``["median", "exponential"]``).
          training_mode :
                training_mode on which to apply
                the denoising. Default: ``"train"``.
          denoiser_kwargs :
                Optional dictionary to pass hyperparameters specific
                to each denoiser, e.g.
                ``{"median": {"kernel_size": 5}, "exponential": {"alpha": 0.2}}``.
          """
          self.training_mode: str = training_mode or "train"
          self.pipeline: List[BaseDenoiser] = [
                self._instantiate(name, denoiser_kwargs or {}) for name in denoiser_names
          ]

     # ---------------------------------------------------------------------

     def _instantiate(
          self, name: str, denoiser_kwargs: Dict[str, dict]
     ) -> BaseDenoiser:
          if name not in _DENOISERS:
                raise ValueError(
                     f"'{name}' is not implemented. "
                     f"Available options: {list(_DENOISERS)}."
                )
          cls = _DENOISERS[name]
          kwargs = denoiser_kwargs.get(name, {})
          sig = inspect.signature(cls)  # safeguard: check the kwargs
          sig.bind_partial(**kwargs)
          return cls(**kwargs)

     # ---------------------------------------------------------------------

     def apply(self,tensor_limits_keeper, dataset, /) -> None:
          """
          Modifies ``dataset.raw_values`` and possibly shifts
          ``dataset.df`` if a filter delay is introduced.
          """
          df_dates_copy = dataset.df_dates.copy()
          df_dates_copy['index'] = df_dates_copy.index
          raw = dataset.raw_values.clone()
          
          indices_i = torch.tensor(getattr(tensor_limits_keeper,f"{self.training_mode}_indices"),dtype=torch.long)
          raw_value_i = torch.index_select(raw, 0, indices_i) # Select specific indices 
          
          # Chained denoiser.
          for denoiser in self.pipeline:
            raw_value_i = denoiser(raw_value_i)
            
          return raw_value_i  
