"""
Lead-lag analysis: STL-residual-based CCF + Granger causality + rolling correlations.

Methodology note
----------------
Cross-correlation is computed on STL residuals — the idiosyncratic 'surprise'
component of each series after removing trend and 12-month seasonality. This
isolates true short-run transmission from shared trend and seasonal pulses
(e.g. spring construction season) that otherwise mask real lead-lag signals
on commodity price levels.

Convention for lags
-------------------
    positive k  -> driver leads HRC by k months
    negative k  -> HRC leads driver by |k| months
    k = 0       -> contemporaneous
"""
from __future__ import annotations
import warnings
from typing import Optional
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Transformation: STL residual extraction
# ----------------------------------------------------------------------

def _stl_residual(series: pd.Series, period: int = 12) -> Optional[pd.Series]:
    """
    Decompose a series into trend + seasonal + residual via STL and return
    the residual. Returns None if the series is too short or STL fails;
    callers fall back to first-difference as a degraded alternative.
    """
    from statsmodels.tsa.seasonal import STL

    s = series.dropna()
    if len(s) < 2 * period + 4:        # STL needs at least 2 full cycles
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = STL(s, period=period, robust=True).fit()
        return result.resid
    except Exception:
        return None


def _transform(series: pd.Series, period: int = 12) -> pd.Series:
    """
    Convert a price series to its STL residual. If STL is not feasible
    (e.g. fewer than ~28 observations), fall back to first-difference of
    log (for positive series) or raw first-difference (otherwise) so the
    pipeline still produces something sensible.
    """
    resid = _stl_residual(series, period=period)
    if resid is not None:
        return resid
    s = series.dropna()
    if (s > 0).all():
        return np.log(s).diff().dropna()
    return s.diff().dropna()


# ----------------------------------------------------------------------
# Cross-correlation
# ----------------------------------------------------------------------

def cross_correlation(y: pd.Series, x: pd.Series, max_lag: int = 6) -> pd.DataFrame:
    """
    CCF on STL residuals.

    At each lag k, correlation between residual(y)[t] and residual(x)[t-k].
        positive k  -> x leads y
        negative k  -> y leads x
        k = 0       -> contemporaneous
    """
    y_t = _transform(y)
    x_t = _transform(x)
    common = y_t.index.intersection(x_t.index)
    y_t, x_t = y_t.loc[common], x_t.loc[common]

    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            corr = y_t.corr(x_t.shift(k))
        else:
            corr = y_t.shift(-k).corr(x_t)
        rows.append({"lag": k, "correlation": corr})
    return pd.DataFrame(rows)


def best_lead_lag(ccf_df: pd.DataFrame) -> dict:
    """Pick the lag with the largest absolute correlation."""
    valid = ccf_df.dropna(subset=["correlation"])
    if len(valid) == 0:
        return {"best_lag": 0, "correlation": np.nan, "direction": "n/a"}
    best = valid.loc[valid["correlation"].abs().idxmax()]
    direction = ("x leads y" if best["lag"] > 0 else
                 "y leads x" if best["lag"] < 0 else
                 "contemporaneous")
    return {
        "best_lag": int(best["lag"]),
        "correlation": float(best["correlation"]),
        "direction": direction,
    }


# ----------------------------------------------------------------------
# Full lag matrix (driver x lag) — for the heatmap
# ----------------------------------------------------------------------

def lag_matrix(y: pd.Series, X: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """
    Full lag-by-driver correlation matrix on STL residuals.

    Returns a DataFrame with:
        index   = driver names
        columns = lag values from -max_lag to +max_lag
        values  = Pearson correlation between residual(y)[t] and residual(x)[t-k]
    """
    if X is None or X.shape[1] == 0:
        return pd.DataFrame()

    lags = list(range(-max_lag, max_lag + 1))
    y_t = _transform(y)

    rows = {}
    for col in X.columns:
        x_t = _transform(X[col])
        common = y_t.index.intersection(x_t.index)
        if len(common) < 6:
            rows[col] = {k: np.nan for k in lags}
            continue
        yc, xc = y_t.loc[common], x_t.loc[common]
        row = {}
        for k in lags:
            if k >= 0:
                row[k] = yc.corr(xc.shift(k))
            else:
                row[k] = yc.shift(-k).corr(xc)
        rows[col] = row

    return pd.DataFrame.from_dict(rows, orient="index", columns=lags)


# ----------------------------------------------------------------------
# Granger causality (on residuals)
# ----------------------------------------------------------------------

def granger_test(y: pd.Series, x: pd.Series, max_lag: int = 6,
                 significance: float = 0.05) -> dict:
    """Granger causality on STL residuals. Tests lags 1..max_lag."""
    from statsmodels.tsa.stattools import grangercausalitytests

    y_t = _transform(y)
    x_t = _transform(x)
    common = y_t.index.intersection(x_t.index)
    y_t, x_t = y_t.loc[common], x_t.loc[common]
    df = pd.concat([y_t, x_t], axis=1).dropna()

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


# ----------------------------------------------------------------------
# Summary (backward-compatible with report & narrator)
# ----------------------------------------------------------------------

def lead_lag_summary(y: pd.Series, X: pd.DataFrame,
                     max_lag: int = 6, significance: float = 0.05) -> pd.DataFrame:
    """
    One-row-per-driver summary of CCF + Granger findings (residual-based).
    Kept for backward compatibility with the narrator and HTML report.
    """
    empty = pd.DataFrame(columns=["driver", "best_lag_months", "ccf_at_best_lag",
                                   "lead_lag_direction", "granger_x_causes_y",
                                   "granger_min_pvalue", "granger_best_lag"])
    if X is None or X.shape[1] == 0:
        return empty

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
    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values(
        "ccf_at_best_lag", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


# ----------------------------------------------------------------------
# Rolling correlations — kept on log-returns for stability views
# ----------------------------------------------------------------------

def rolling_correlations(y: pd.Series, X: pd.DataFrame,
                         window: int = 12) -> pd.DataFrame:
    """Rolling correlation of each driver vs target on log-returns."""
    if (y > 0).all():
        y_r = np.log(y).diff()
    else:
        y_r = y.diff()
    out = pd.DataFrame(index=y_r.index)
    for col in X.columns:
        x = X[col]
        if (x > 0).all():
            x_r = np.log(x).diff()
        else:
            x_r = x.diff()
        out[col] = y_r.rolling(window).corr(x_r)
    return out
