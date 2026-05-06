"""
The Model interface — every forecasting model implements this contract.

This is what makes models swappable: ARIMAX, ARDL, GARCH, Prophet, XGBoost,
or anything else can live behind the same interface. The pipeline never knows
or cares which model(s) are running.

To add a new model:
    1. Create models/your_model.py
    2. Subclass Model and implement fit() and forecast()
    3. Decorate with @register_model("your_model_name")
    4. Add a block to config.yaml under `models:` with the same name
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Standardised container for everything a forecasting model produces."""
    name: str                                          # 'arimax', 'ardl', etc.
    region: str                                        # 'china' or 'india'
    fitted: pd.Series = None                           # in-sample fitted values
    residuals: pd.Series = None                        # in-sample residuals
    forecast: pd.Series = None                         # h-step ahead forecast
    forecast_lower: Optional[pd.Series] = None         # 95% lower CI
    forecast_upper: Optional[pd.Series] = None         # 95% upper CI
    oos_predictions: Optional[pd.Series] = None        # out-of-sample test predictions
    oos_actuals: Optional[pd.Series] = None            # actuals for OOS test
    metrics: Dict[str, float] = field(default_factory=dict)        # RMSE, MAE, MAPE, R²
    coefficients: Optional[pd.DataFrame] = None        # coefficients table
    diagnostics: Dict[str, Any] = field(default_factory=dict)      # AIC, BIC, etc.
    error: Optional[str] = None                        # if fit failed, why


@dataclass
class VolatilityResult:
    """Container for volatility models like GARCH."""
    name: str
    region: str
    conditional_volatility: pd.Series = None
    standardized_residuals: pd.Series = None
    forecast_volatility: Optional[pd.Series] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Model(ABC):
    """Base class for all models."""

    name: str = "base"           # set by @register_model decorator
    kind: str = "forecast"       # 'forecast' or 'volatility'

    def __init__(self, config: dict):
        """`config` is the per-model block from config.yaml."""
        self.config = config
        self.is_fitted = False
        self._fit_artifact = None        # whatever the underlying library returns

    @abstractmethod
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "Model":
        """Fit the model on y (target) and optionally X (drivers)."""
        ...

    @abstractmethod
    def forecast(
        self,
        steps: int,
        X_future: Optional[pd.DataFrame] = None,
        region: str = "",
    ):
        """Produce a ForecastResult or VolatilityResult."""
        ...

    @staticmethod
    def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
        """Standard metric set used across all models for fair comparison."""
        actual = pd.Series(actual).dropna()
        predicted = pd.Series(predicted).dropna()
        common = actual.index.intersection(predicted.index)
        if len(common) == 0:
            return {"rmse": float("nan"), "mae": float("nan"),
                    "mape": float("nan"), "r2": float("nan")}
        a, p = actual.loc[common], predicted.loc[common]
        err = a - p
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        mape_vals = np.abs(err / a.replace(0, np.nan)).dropna()
        mape = float(mape_vals.mean() * 100) if len(mape_vals) else float("nan")
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((a - a.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
