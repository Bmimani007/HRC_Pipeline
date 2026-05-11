"""
Diagnostics — ADF stationarity tests, VIF (multicollinearity), correlations.
"""
from __future__ import annotations
import warnings
from typing import Dict
import numpy as np
import pandas as pd


def adf_test(series: pd.Series, name: str = "") -> dict:
    """Augmented Dickey-Fuller test on a single series."""
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    if len(s) < 10:
        return {"variable": name, "adf_stat": np.nan, "p_value": np.nan,
                "stationary": False, "n_obs": len(s)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = adfuller(s, autolag="AIC")
    return {
        "variable": name,
        "adf_stat": float(r[0]),
        "p_value": float(r[1]),
        "stationary": bool(r[1] < 0.05),
        "n_obs": int(len(s)),
        "critical_5pct": float(r[4]["5%"]),
    }


def adf_table(df: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
    """ADF on every column — both levels and first differences."""
    rows = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        # Levels
        r = adf_test(s, name=col)
        r["transform"] = "levels"
        rows.append(r)
        # First differences
        r2 = adf_test(s.diff(), name=col)
        r2["transform"] = "first_diff"
        rows.append(r2)
    out = pd.DataFrame(rows)
    out["stationary"] = out["p_value"] < significance
    return out


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor — flags multicollinearity. VIF > 10 = severe."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    empty = pd.DataFrame(columns=["variable", "vif", "severity"])
    if X is None or X.shape[1] == 0:
        return empty
    Xc = X.dropna().copy()
    if len(Xc) < len(Xc.columns) + 5:
        return empty
    Xc = Xc.replace([np.inf, -np.inf], np.nan).dropna()
    if Xc.shape[1] == 0:
        return empty
    Xc_with_const = Xc.assign(_const=1.0)
    rows = []
    for i, col in enumerate(Xc.columns):
        try:
            v = variance_inflation_factor(Xc_with_const.values, i)
        except Exception:
            v = np.nan
        sev = ("severe" if v > 10 else
               "moderate" if v > 5 else
               "low")
        rows.append({"variable": col, "vif": float(v), "severity": sev})
    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Standard correlation matrix."""
    return df.corr(method=method)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Quick summary stats per column."""
    s = df.describe().T
    s["missing"] = df.isna().sum()
    s["coef_var"] = s["std"] / s["mean"].replace(0, np.nan)
    return s
