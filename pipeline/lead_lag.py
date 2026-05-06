"""
Lead-lag analysis: CCF (cross-correlation), Granger causality, rolling correlations.
"""
from __future__ import annotations
import warnings
from typing import Dict, List
import numpy as np
import pandas as pd


def cross_correlation(y: pd.Series, x: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    """
    CCF: at each lag k, correlation between y[t] and x[t-k].
        positive k  → x leads y
        negative k  → y leads x
        k = 0       → contemporaneous
    """
    y = y.dropna(); x = x.dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]
    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            corr = y.corr(x.shift(k))
        else:
            corr = y.shift(-k).corr(x)
        rows.append({"lag": k, "correlation": corr})
    return pd.DataFrame(rows)


def best_lead_lag(ccf_df: pd.DataFrame) -> dict:
    """Pick the lag with the largest absolute correlation."""
    best = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
    direction = ("x leads y" if best["lag"] > 0 else
                 "y leads x" if best["lag"] < 0 else
                 "contemporaneous")
    return {
        "best_lag": int(best["lag"]),
        "correlation": float(best["correlation"]),
        "direction": direction,
    }


def granger_test(y: pd.Series, x: pd.Series, max_lag: int = 12,
                 significance: float = 0.05) -> dict:
    """
    Granger causality: does x help predict y?
    Tests at lag = 1..max_lag, reports the smallest p-value.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    y = y.dropna(); x = x.dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < max_lag + 5:
        return {"x_causes_y": False, "min_pvalue": np.nan,
                "best_lag": np.nan, "details": []}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(df.values, maxlag=max_lag, verbose=False)
        details = []
        for lag, r in res.items():
            p = r[0]["ssr_ftest"][1]
            details.append({"lag": lag, "p_value": float(p)})
        details_df = pd.DataFrame(details)
        best = details_df.loc[details_df["p_value"].idxmin()]
        return {
            "x_causes_y": bool(best["p_value"] < significance),
            "min_pvalue": float(best["p_value"]),
            "best_lag": int(best["lag"]),
            "details": details,
        }
    except Exception as e:
        return {"x_causes_y": False, "min_pvalue": np.nan,
                "best_lag": np.nan, "details": [], "error": str(e)}


def lead_lag_summary(y: pd.Series, X: pd.DataFrame,
                     max_lag: int = 12, significance: float = 0.05) -> pd.DataFrame:
    """One-row-per-driver summary of CCF + Granger findings."""
    rows = []
    for col in X.columns:
        x = X[col]
        ccf = cross_correlation(y, x, max_lag=max_lag)
        bl = best_lead_lag(ccf)
        gc = granger_test(y, x, max_lag=max_lag, significance=significance)
        rows.append({
            "driver": col,
            "best_lag_months": bl["best_lag"],
            "ccf_at_best_lag": bl["correlation"],
            "lead_lag_direction": bl["direction"],
            "granger_x_causes_y": gc["x_causes_y"],
            "granger_min_pvalue": gc["min_pvalue"],
            "granger_best_lag": gc["best_lag"],
        })
    return pd.DataFrame(rows).sort_values(
        "ccf_at_best_lag", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def rolling_correlations(y: pd.Series, X: pd.DataFrame,
                         window: int = 12) -> pd.DataFrame:
    """Rolling correlation of each driver vs target. One column per driver."""
    out = pd.DataFrame(index=y.index)
    for col in X.columns:
        out[col] = y.rolling(window).corr(X[col])
    return out
