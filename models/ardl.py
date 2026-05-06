"""
ARDL — Autoregressive Distributed Lag.

Captures both short-run dynamics and a long-run equilibrium relationship
between the target and its drivers. Useful when drivers are I(0) or I(1).
"""
from __future__ import annotations
from typing import Optional
import warnings
import numpy as np
import pandas as pd

from .base import Model, ForecastResult
from .registry import register_model


@register_model("ardl")
class ARDLModel(Model):

    kind = "forecast"

    def __init__(self, config: dict):
        super().__init__(config)
        # Fixed ARDL orders — much faster + more deterministic than grid search.
        # For a 9-driver problem, ardl_select_order would explore ~10M combos;
        # fixed order is a single fit. Defaults follow standard practice.
        self.ar_order = int(config.get("ar_order", config.get("max_lag", 2)))
        self.dl_order = int(config.get("dl_order", config.get("max_lag", 2)))
        # Optionally fall back to grid search ONLY if user explicitly opts in
        # AND the problem is small enough to be tractable.
        self.use_order_selection = bool(config.get("use_order_selection", False))
        self.forecast_horizon = int(config.get("forecast_horizon", 12))
        self.test_size = int(config.get("test_size", 12))

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ARDLModel":
        from statsmodels.tsa.ardl import ARDL

        # Order selection is exponential in n_drivers — only run if user opted
        # in and the problem is small enough. Otherwise use fixed order.
        n_drivers = X.shape[1] if X is not None else 0
        do_search = self.use_order_selection and n_drivers <= 4

        if do_search:
            from statsmodels.tsa.ardl import ardl_select_order
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sel = ardl_select_order(
                    endog=y, exog=X,
                    maxlag=min(self.ar_order, 3),
                    maxorder=min(self.dl_order, 3),
                    ic="aic", trend="c",
                )
                self._fit_artifact = sel.model.fit()
                self._selected_orders = (sel.ar_lags, sel.dl_lags)
        else:
            # Fixed-order ARDL — single fit, fast.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARDL(
                    endog=y, lags=self.ar_order,
                    exog=X, order=self.dl_order,
                    trend="c",
                )
                self._fit_artifact = model.fit()
                self._selected_orders = (self.ar_order, self.dl_order)

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

        fitted = res.fittedvalues
        residuals = res.resid

        # Build naive X_future (hold last values constant) if not supplied
        if X_future is None and self._X_train is not None:
            last_vals = self._X_train.iloc[-1].values
            future_idx_pre = pd.date_range(
                start=self._y_train.index[-1] + pd.DateOffset(months=1),
                periods=steps, freq="MS"
            )
            X_future = pd.DataFrame(
                np.tile(last_vals, (steps, 1)),
                columns=self._X_train.columns, index=future_idx_pre,
            )

        try:
            # ARDL's forecast() handles exog properly; build CI manually using
            # residual std (statsmodels ARDL doesn't expose CI directly)
            forecast_mean = res.forecast(steps=steps, exog=X_future)
            sigma = float(np.std(residuals.dropna()))
            # Widening CI by sqrt(h) — standard for short-horizon forecasts
            h_factor = np.sqrt(np.arange(1, steps + 1))
            lower = pd.Series(
                np.asarray(forecast_mean) - 1.96 * sigma * h_factor,
                index=getattr(forecast_mean, "index", None),
            )
            upper = pd.Series(
                np.asarray(forecast_mean) + 1.96 * sigma * h_factor,
                index=getattr(forecast_mean, "index", None),
            )

            last = self._y_train.index[-1]
            future_idx = pd.date_range(
                start=last + pd.DateOffset(months=1),
                periods=steps, freq="MS"
            )
            if not isinstance(forecast_mean, pd.Series):
                forecast_mean = pd.Series(np.asarray(forecast_mean), index=future_idx)
            else:
                forecast_mean.index = future_idx
            lower.index = future_idx
            upper.index = future_idx
        except Exception as e:
            return ForecastResult(name=self.name, region=region,
                                  fitted=fitted, residuals=residuals,
                                  error=f"Forecast failed: {e}")

        oos_pred, oos_actual = self._oos_test()
        metrics = self.compute_metrics(oos_actual, oos_pred) if oos_actual is not None else {}

        try:
            coef_df = pd.DataFrame({
                "coefficient": res.params,
                "std_err": res.bse,
                "t": res.tvalues,
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
                "ar_lags": str(self._selected_orders[0]),
                "dl_lags": str(self._selected_orders[1]),
                "n_obs": int(len(self._y_train)),
            },
        )

    def _oos_test(self):
        from statsmodels.tsa.ardl import ARDL
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
                m = ARDL(
                    endog=y_tr, lags=self.ar_order,
                    exog=X_tr, order=self.dl_order,
                    trend="c",
                ).fit()

            pred = m.forecast(steps=len(y_te), exog=X_te)
            if not isinstance(pred, pd.Series):
                pred = pd.Series(np.asarray(pred), index=y_te.index)
            else:
                pred.index = y_te.index
            return pred, y_te
        except Exception:
            return None, None
