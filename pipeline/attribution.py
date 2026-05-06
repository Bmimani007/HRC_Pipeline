"""
Driver attribution — rolling regression to attribute target moves to drivers.

For each rolling window, we fit OLS y = α + Σβᵢ × xᵢ + ε.
The rolling β's tell us how the relationship changes over time.
The fraction of variance explained by each driver gives attribution weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    region: str
    rolling_betas: pd.DataFrame              # one column per driver, index = date
    rolling_r2: pd.Series                    # one R² per window
    full_sample_betas: pd.Series             # OLS β on full sample
    full_sample_attribution: pd.Series       # fraction of explained variance per driver
    current_attribution: pd.Series           # most recent window's attribution


def _ols_fit(y: pd.Series, X: pd.DataFrame):
    """OLS via numpy (no statsmodels dep — simple closed form)."""
    Xa = np.column_stack([np.ones(len(X)), X.values])
    try:
        beta, *_ = np.linalg.lstsq(Xa, y.values, rcond=None)
    except Exception:
        return None, None, None
    yhat = Xa @ beta
    resid = y.values - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y.values - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, yhat, r2


def _attribution_weights(y: pd.Series, X: pd.DataFrame, betas: np.ndarray) -> pd.Series:
    """
    Fraction of explained variance attributed to each driver.
    Decomposition: var(yhat) ≈ Σ βᵢ²·var(xᵢ) + cross-terms (ignored).
    Simpler proxy: |βᵢ × σ(xᵢ)| normalised to sum to 100%.
    """
    contributions = np.array([
        abs(betas[i + 1]) * X.iloc[:, i].std()
        for i in range(X.shape[1])
    ])
    total = contributions.sum()
    if total <= 0 or not np.isfinite(total):
        return pd.Series([np.nan] * X.shape[1], index=X.columns)
    weights = contributions / total * 100
    return pd.Series(weights, index=X.columns)


def rolling_attribution(y: pd.Series, X: pd.DataFrame,
                         window: int = 24, region: str = "") -> AttributionResult:
    """Run a rolling-window OLS and attribute variance per window."""
    df = pd.concat([y.rename("target"), X], axis=1).dropna()
    if len(df) < window + 5:
        # Not enough data — return empty
        return AttributionResult(
            region=region,
            rolling_betas=pd.DataFrame(columns=X.columns),
            rolling_r2=pd.Series(dtype=float),
            full_sample_betas=pd.Series(dtype=float),
            full_sample_attribution=pd.Series(dtype=float),
            current_attribution=pd.Series(dtype=float),
        )

    rolling_betas = []
    rolling_attrs = []
    rolling_r2 = []
    dates = []

    for i in range(window, len(df) + 1):
        sub = df.iloc[i - window:i]
        y_sub = sub["target"]
        X_sub = sub.drop(columns="target")
        beta, _, r2 = _ols_fit(y_sub, X_sub)
        if beta is None:
            continue
        w = _attribution_weights(y_sub, X_sub, beta)
        rolling_betas.append(pd.Series(beta[1:], index=X_sub.columns))
        rolling_attrs.append(w)
        rolling_r2.append(r2)
        dates.append(sub.index[-1])

    rb_df = pd.DataFrame(rolling_betas, index=pd.Index(dates, name="date"))
    rr2 = pd.Series(rolling_r2, index=pd.Index(dates, name="date"))

    # Full-sample
    fs_beta, _, _ = _ols_fit(df["target"], df.drop(columns="target"))
    fs_betas = pd.Series(fs_beta[1:], index=X.columns) if fs_beta is not None \
               else pd.Series([np.nan] * X.shape[1], index=X.columns)
    fs_attr = _attribution_weights(df["target"], df.drop(columns="target"), fs_beta) \
              if fs_beta is not None else pd.Series([np.nan] * X.shape[1], index=X.columns)

    current = rolling_attrs[-1] if rolling_attrs else pd.Series([np.nan] * X.shape[1],
                                                                  index=X.columns)

    return AttributionResult(
        region=region,
        rolling_betas=rb_df,
        rolling_r2=rr2,
        full_sample_betas=fs_betas,
        full_sample_attribution=fs_attr,
        current_attribution=current,
    )
