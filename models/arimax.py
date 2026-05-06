"""
ARIMAX — AutoRegressive Integrated Moving Average with eXogenous regressors.

Removing this model: set `arimax: enabled: false` in config.yaml.
Replacing this model: create a new file alongside this one with a different
class name and registration tag, then point config.yaml to the new name.
"""
from __future__ import annotations
from typing import Optional
import warnings
import numpy as np
import pandas as pd

from .base import Model, ForecastResult
from .registry import register_model


@register_model("arimax")
class ARIMAXModel(Model):
    """Wraps statsmodels SARIMAX with the project's standard interface."""

    kind = "forecast"

    def __init__(self, config: dict):
        super().__init__(config)
        self.order = tuple(config.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(config.get("seasonal_order", [0, 0, 0, 0]))
        self.forecast_horizon = int(config.get("forecast_horizon", 12))
        self.test_size = int(config.get("test_size", 12))

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ARIMAXModel":
        # Lazy import so the registry doesn't crash if statsmodels is missing
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y, exog=X,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit_artifact = model.fit(disp=False)
        self._y_train = y
        self._X_train = X
        self.is_fitted = True
        return self

    def forecast(
        self,
        steps: int = None,
        X_future: Optional[pd.DataFrame] = None,
        region: str = "",
    ) -> ForecastResult:
        if not self.is_fitted:
            return ForecastResult(name=self.name, region=region,
                                  error="Model not fitted")

        steps = steps or self.forecast_horizon
        res = self._fit_artifact

        # In-sample fitted values + residuals
        fitted = res.fittedvalues
        residuals = res.resid

        # If model has exogenous drivers and no X_future provided, build a naive
        # carry-forward (hold last observed driver values constant). This is a
        # deliberate, transparent assumption — analyst can override by passing
        # X_future explicitly.
        if X_future is None and self._X_train is not None:
            last_vals = self._X_train.iloc[-1].values
            future_idx = pd.date_range(
                start=self._y_train.index[-1] + pd.DateOffset(months=1),
                periods=steps, freq="MS"
            )
            X_future = pd.DataFrame(
                np.tile(last_vals, (steps, 1)),
                columns=self._X_train.columns, index=future_idx,
            )

        # Out-of-sample forecast
        try:
            fc = res.get_forecast(steps=steps, exog=X_future)
            forecast_mean = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
            lower = ci.iloc[:, 0]
            upper = ci.iloc[:, 1]
            # Build a future index if statsmodels gave us integers
            if not isinstance(forecast_mean.index, pd.DatetimeIndex):
                last = self._y_train.index[-1]
                future_idx = pd.date_range(
                    start=last + pd.DateOffset(months=1),
                    periods=steps, freq="MS"
                )
                forecast_mean.index = future_idx
                lower.index = future_idx
                upper.index = future_idx
        except Exception as e:
            return ForecastResult(name=self.name, region=region,
                                  fitted=fitted, residuals=residuals,
                                  error=f"Forecast failed: {e}")

        # Out-of-sample test: refit on (y minus last test_size obs), predict the held-out
        oos_pred, oos_actual = self._oos_test()

        metrics = {}
        if oos_actual is not None:
            metrics = self.compute_metrics(oos_actual, oos_pred)

        # Coefficients table
        try:
            coef_df = pd.DataFrame({
                "coefficient": res.params,
                "std_err": res.bse,
                "z": res.tvalues,
                "p_value": res.pvalues,
            })
        except Exception:
            coef_df = None

        return ForecastResult(
            name=self.name,
            region=region,
            fitted=fitted,
            residuals=residuals,
            forecast=forecast_mean,
            forecast_lower=lower,
            forecast_upper=upper,
            oos_predictions=oos_pred,
            oos_actuals=oos_actual,
            metrics=metrics,
            coefficients=coef_df,
            diagnostics={
                "aic": float(res.aic),
                "bic": float(res.bic),
                "order": str(self.order),
                "seasonal_order": str(self.seasonal_order),
                "n_obs": int(len(self._y_train)),
            },
        )

    def _oos_test(self):
        """Hold out last `test_size` obs, fit on the rest, predict the holdout."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        if len(self._y_train) <= self.test_size + 12:
            return None, None
        try:
            split = -self.test_size
            y_tr = self._y_train.iloc[:split]
            y_te = self._y_train.iloc[split:]
            X_tr = self._X_train.iloc[:split] if self._X_train is not None else None
            X_te = self._X_train.iloc[split:] if self._X_train is not None else None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    y_tr, exog=X_tr,
                    order=self.order, seasonal_order=self.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            pred = m.get_forecast(steps=len(y_te), exog=X_te).predicted_mean
            pred.index = y_te.index
            return pred, y_te
        except Exception:
            return None, None
