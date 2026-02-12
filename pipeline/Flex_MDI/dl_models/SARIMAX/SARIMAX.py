# models.py

import torch
import torch.nn as nn
import statsmodels.api as sm

class SARIMAX(nn.Module):
    """
    Wrapper torch.Module around statsmodels SARIMAX to fit once on init 
    and then predict in forward, accepting x (B,C,N,L) and x_calendar (B,Z).
    """
    def __init__(self,
                 order=(1,0,0),
                 seasonal_order=(0,0,0,0),
                 enforce_stationarity=True,
                 enforce_invertibility=True,
                 **kwargs):
        super().__init__()
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.models = None  # will hold one SARIMAX per node

    def _fit_per_node(self, x: torch.Tensor, exog: torch.Tensor):
        # x: [B,1,N,L], exog: [B,Z] or [B,1,N,Z]
        B, C, N, L = x.shape
        # prepare exog per node: [B, Z]
        z = exog.shape[-1]
        if exog.ndim == 3:
            exog = exog.squeeze(1)  # [B,N,Z]
        # fit one model per node on the first batch only
        self.models = []
        for i in range(N):
            y_i = x[:, 0, i, :].reshape(-1).cpu().numpy()        # length B*L?
            # here we assume L == time dimension of training series
            exog_i = None
            if exog is not None:
                exog_i = exog[:, i, :].cpu().numpy()
            mod = sm.tsa.statespace.SARIMAX(
                endog=y_i,
                exog=exog_i,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            res = mod.fit(disp=False)
            self.models.append(res)

    def forward(self,
                x: torch.Tensor,
                x_vision: torch.Tensor = None,
                x_calendar: torch.Tensor = None) -> torch.Tensor:
        # x: [B,C,N,L], x_calendar: [B,Z] or [B,1,N,Z]
        B, C, N, L = x.shape
        if self.models is None:
            self._fit_per_node(x, x_calendar)

        # predict one step ahead per node
        preds = []
        for i, res in enumerate(self.models):
            # predict for next time = L to L (single step)
            y_hat = res.get_prediction(start=L, end=L, exog=(x_calendar[:, i, :] if x_calendar is not None and x_calendar.ndim==3 else None))
            val = y_hat.predicted_mean  # shape (1,) or (B,) depending fit
            preds.append(torch.tensor(val, device=x.device).reshape(B, 1))
        # preds: list of [B,1] × N  -> [B,N]
        out = torch.cat(preds, dim=1)
        return out.unsqueeze(-1)  # [B,N,1]
    
