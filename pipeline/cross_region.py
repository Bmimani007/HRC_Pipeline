"""
Cross-region analysis — pairwise lead-lag, Granger causality, and cointegration
across HRC markets.

All analysis on monthly RETURNS (currency-neutral, unit-free).
Cointegration uses LEVELS (long-run equilibrium concept).

Public entry point:
    analyse_cross_region(dataset) → CrossRegionResult
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import warnings
import numpy as np
import pandas as pd


@dataclass
class PairwiseResult:
    region_a: str
    region_b: str
    ccf_lags: List[int] = field(default_factory=list)
    ccf_values: List[float] = field(default_factory=list)
    best_lag: Optional[int] = None
    best_corr: Optional[float] = None
    contemporaneous_corr: Optional[float] = None
    granger_a_to_b_pvalue: Optional[float] = None
    granger_b_to_a_pvalue: Optional[float] = None
    granger_lag_used: int = 6
    cointegration_pvalue: Optional[float] = None
    cointegrated: bool = False
    n_obs: int = 0
    overlap_start: Optional[str] = None
    overlap_end: Optional[str] = None
    leader: Optional[str] = None
    leader_strength: str = "weak"


@dataclass
class CrossRegionResult:
    pairs: List[PairwiseResult] = field(default_factory=list)
    overlap_start: Optional[str] = None
    overlap_end: Optional[str] = None
    n_overlap: int = 0
    regions_analysed: List[str] = field(default_factory=list)
    error_msg: str = ""
    success: bool = True


def _to_returns(prices: pd.Series) -> pd.Series:
    s = prices.dropna().sort_index()
    return np.log(s).diff().dropna()


def _align_two_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series, int]:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    return a.loc[common], b.loc[common], len(common)


def _compute_ccf(a: pd.Series, b: pd.Series, max_lag: int = 24) -> Tuple[List[int], List[float]]:
    """Cross-correlation function from -max_lag to +max_lag.
    Positive lag k = a leads b by k months (a_t correlates with b_{t+k})."""
    a_v = a.values
    b_v = b.values
    n = len(a_v)
    if n < max_lag + 5:
        max_lag = max(1, n - 5)

    lags = list(range(-max_lag, max_lag + 1))
    values = []
    for k in lags:
        if k > 0:
            x = a_v[:-k]; y = b_v[k:]
        elif k < 0:
            x = a_v[-k:]; y = b_v[:k]
        else:
            x, y = a_v, b_v
        if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            values.append(0.0)
            continue
        c = np.corrcoef(x, y)[0, 1]
        values.append(float(c) if not np.isnan(c) else 0.0)
    return lags, values


def _granger_test(a: pd.Series, b: pd.Series, max_lag: int = 6) -> Optional[float]:
    """Does a Granger-cause b? Returns smallest p-value across lags 1..max_lag."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return None
    try:
        df = pd.concat([b, a], axis=1).dropna()
        if len(df) < max_lag + 10:
            max_lag = max(1, len(df) - 10)
        if max_lag < 1:
            return None
        df.columns = ["target", "candidate_cause"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(df, maxlag=max_lag, verbose=False)
        pvalues = []
        for lag, (test_stats, _) in results.items():
            ssr_chi2 = test_stats.get("ssr_chi2test")
            if ssr_chi2 and len(ssr_chi2) >= 2:
                pvalues.append(ssr_chi2[1])
        return float(min(pvalues)) if pvalues else None
    except Exception:
        return None


def _engle_granger_cointegration(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Engle-Granger cointegration on LEVELS. Returns p-value."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        return None
    try:
        a_aligned, b_aligned, n = _align_two_series(a, b)
        if n < 30:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, pvalue, _ = coint(a_aligned.values, b_aligned.values)
        return float(pvalue) if not np.isnan(pvalue) else None
    except Exception:
        return None


def _interpret_pair(p: PairwiseResult) -> None:
    pa_b = p.granger_a_to_b_pvalue
    pb_a = p.granger_b_to_a_pvalue
    a_causes_b = pa_b is not None and pa_b < 0.05
    b_causes_a = pb_a is not None and pb_a < 0.05

    if a_causes_b and not b_causes_a:
        p.leader = "a"
    elif b_causes_a and not a_causes_b:
        p.leader = "b"
    elif a_causes_b and b_causes_a:
        p.leader = "both"
    else:
        if p.best_lag is not None and abs(p.best_lag) >= 1 \
                and p.best_corr is not None and abs(p.best_corr) >= 0.4:
            p.leader = "a" if p.best_lag > 0 else "b"
        else:
            p.leader = "neither"

    if p.best_corr is None:
        p.leader_strength = "weak"
    elif abs(p.best_corr) >= 0.7:
        p.leader_strength = "strong"
    elif abs(p.best_corr) >= 0.4:
        p.leader_strength = "moderate"
    else:
        p.leader_strength = "weak"


def analyse_cross_region(dataset, max_lag: int = 24,
                            granger_lag: int = 6) -> CrossRegionResult:
    """Compute pairwise cross-region lead-lag, Granger causality, cointegration."""
    region_keys = list(dataset.regions.keys())
    if len(region_keys) < 2:
        return CrossRegionResult(
            success=False,
            error_msg=f"Need ≥2 regions; have {len(region_keys)}.",
            regions_analysed=region_keys,
        )

    returns = {}
    levels = {}
    for k in region_keys:
        levels[k] = dataset.regions[k].y.dropna().sort_index()
        returns[k] = _to_returns(levels[k])

    # Common-overlap across ALL regions
    common_idx = returns[region_keys[0]].index
    for k in region_keys[1:]:
        common_idx = common_idx.intersection(returns[k].index)

    if len(common_idx) < 30:
        return CrossRegionResult(
            success=False,
            error_msg=f"Insufficient overlap: {len(common_idx)} months. Need ≥30.",
            regions_analysed=region_keys,
        )

    overall_start = str(common_idx.min().date())
    overall_end = str(common_idx.max().date())

    pairs = []
    for i, a_key in enumerate(region_keys):
        for b_key in region_keys[i + 1:]:
            a_aligned, b_aligned, n = _align_two_series(returns[a_key], returns[b_key])
            if n < 30:
                continue

            lags, values = _compute_ccf(a_aligned, b_aligned, max_lag=max_lag)
            best_idx = int(np.argmax(np.abs(values)))
            best_lag = lags[best_idx]
            best_corr = values[best_idx]
            zero_idx = lags.index(0)
            contemp = values[zero_idx]

            pa_b = _granger_test(a_aligned, b_aligned, max_lag=granger_lag)
            pb_a = _granger_test(b_aligned, a_aligned, max_lag=granger_lag)
            coint_p = _engle_granger_cointegration(levels[a_key], levels[b_key])

            pair = PairwiseResult(
                region_a=a_key, region_b=b_key,
                ccf_lags=lags, ccf_values=values,
                best_lag=best_lag, best_corr=best_corr,
                contemporaneous_corr=contemp,
                granger_a_to_b_pvalue=pa_b,
                granger_b_to_a_pvalue=pb_a,
                granger_lag_used=granger_lag,
                cointegration_pvalue=coint_p,
                cointegrated=(coint_p is not None and coint_p < 0.05),
                n_obs=n,
                overlap_start=str(a_aligned.index.min().date()),
                overlap_end=str(a_aligned.index.max().date()),
            )
            _interpret_pair(pair)
            pairs.append(pair)

    return CrossRegionResult(
        pairs=pairs,
        overlap_start=overall_start, overlap_end=overall_end,
        n_overlap=len(common_idx),
        regions_analysed=region_keys, success=True,
    )
