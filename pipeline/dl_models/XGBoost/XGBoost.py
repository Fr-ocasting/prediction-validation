# models.py

import torch
import torch.nn as nn
try: 
    import xgboost as xgb
except:
    print('No module xgboost')
import numpy as np 

class XGBoost(nn.Module):
    """
    Wrapper torch.Module around xgboost.XGBRegressor: fits one regressor per node on first forward,
    then in forward returns predictions, taking x ([B,C,N,L]) and x_calendar ([B,Z] or [B,1,N,Z]).
    """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=6,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 gamma=0.0,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 objective='reg:squarederror',
                 eval_metric='rmse',
                 **kwargs):
        super().__init__()
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
        )
        self.models = None

    def _fit_per_node(self, x: torch.Tensor, exog: torch.Tensor, y: torch.Tensor):
        # x: [B,C,N,L], exog: [B,Z] or [B,1,N,Z], y: [B,N,1]
        B, C, N, L = x.shape
        X_base = x.view(B, C * N * L).cpu().numpy()
        if exog is not None:
            if exog.ndim == 3:
                exog = exog.squeeze(1).cpu().numpy()  # [B,N,Z]
                exog = exog.reshape(B, -1)            # [B, N*Z]
            else:
                exog = exog.cpu().numpy()             # [B,Z]
            X_base = np.concatenate([X_base, exog], axis=1)
        y_np = y.squeeze(-1).cpu().numpy()  # [B,N]

        self.models = []
        for i in range(N):
            xi = X_base  # same X for all nodes
            yi = y_np[:, i]
            reg = xgb.XGBRegressor(**self.params)
            reg.fit(xi, yi)
            self.models.append(reg)

    def forward(self,
                x: torch.Tensor,
                x_vision: torch.Tensor = None,
                x_calendar: torch.Tensor = None) -> torch.Tensor:
        # x: [B,C,N,L], x_calendar: [B,Z] or [B,1,N,Z]
        B, C, N, L = x.shape
        # assemble feature matrix
        X_base = x.view(B, C * N * L).cpu().numpy()
        if x_calendar is not None:
            if x_calendar.ndim == 3:
                exog = x_calendar.squeeze(1).cpu().numpy().reshape(B, -1)
            else:
                exog = x_calendar.cpu().numpy()
            X_base = np.concatenate([X_base, exog], axis=1)

        if self.models is None:
            # need true targets to train; assume trainer injected them as attribute
            self._fit_per_node(x, x_calendar, self._y_true)

        preds = []
        for reg in self.models:
            p = reg.predict(X_base)  # [B,]
            preds.append(torch.tensor(p, device=x.device).reshape(B, 1))
        out = torch.cat(preds, dim=1)  # [B,N]
        return out.unsqueeze(-1)      # [B,N,1]
