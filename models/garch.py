"""
GARCH(1,1) — Generalized AutoRegressive Conditional Heteroskedasticity.

Models time-varying volatility of the target's returns. Used for risk
analysis: how much price uncertainty exists right now vs the long-run avg?
"""
from __future__ import annotations
from typing import Optional
import warnings
import numpy as np
import pandas as pd

from .base import Model, VolatilityResult
from .registry import register_model


@register_model("garch")
class GARCHModel(Model):

    kind = "volatility"

    def __init__(self, config: dict):
        super().__init__(config)
        self.p = int(config.get("p", 1))
        self.q = int(config.get("q", 1))
        self.mean = config.get("mean", "Constant")
        self.vol = config.get("vol", "GARCH")
        self.dist = config.get("dist", "normal")
        self.forecast_horizon = int(config.get("forecast_horizon", 12))

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "GARCHModel":
        from arch import arch_model

        # Compute log returns from price level
        returns = (np.log(y).diff() * 100).dropna()      # in %
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns, mean=self.mean, vol=self.vol,
                            p=self.p, q=self.q, dist=self.dist)
            self._fit_artifact = am.fit(disp="off", show_warning=False)
        self._returns = returns
        self.is_fitted = True
        return self

    def forecast(
        self,
        steps: int = None,
        X_future: Optional[pd.DataFrame] = None,
        region: str = "",
    ) -> VolatilityResult:
        if not self.is_fitted:
            return VolatilityResult(name=self.name, region=region,
                                    error="Model not fitted")
        steps = steps or self.forecast_horizon
        res = self._fit_artifact

        # Conditional volatility (in % returns)
        cond_vol = pd.Series(res.conditional_volatility, index=self._returns.index)
        std_resid = pd.Series(res.std_resid, index=self._returns.index)

        # Forecast volatility
        try:
            fc = res.forecast(horizon=steps, reindex=False)
            fc_var = fc.variance.iloc[-1]            # last row, h-step variances
            fc_vol = np.sqrt(fc_var)
            last = self._returns.index[-1]
            future_idx = pd.date_range(start=last + pd.DateOffset(months=1),
                                       periods=steps, freq="MS")
            fc_vol = pd.Series(fc_vol.values, index=future_idx)
        except Exception:
            fc_vol = None

        return VolatilityResult(
            name=self.name,
            region=region,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            forecast_volatility=fc_vol,
            diagnostics={
                "aic": float(res.aic),
                "bic": float(res.bic),
                "log_likelihood": float(res.loglikelihood),
                "params": {k: float(v) for k, v in res.params.to_dict().items()},
                "n_returns": int(len(self._returns)),
            },
        )
