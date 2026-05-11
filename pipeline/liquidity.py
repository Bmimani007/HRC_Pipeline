"""
pipeline/liquidity.py
=====================
India liquidity & monetary policy analytics.

Functions
---------
- compute_derived_series(df)           : WACR_Spread, GSec_Repo_Spread, Bank_Credit_YoY from base inputs
- classify_liquidity_regime(spread)    : Tight / Neutral / Surplus using std-band thresholds
- compute_stress_index(df)             : 0-100 z-score composite stress index
- detect_policy_regime(repo)           : Hiking / Pause / Easing / Aggressive Easing
- regime_performance(price, regime)    : Per-regime HRC return/vol/drawdown table
- liquidity_lead_lag(price, liq, lags) : Correlation of HRC returns vs lagged liquidity vars
- summarize_current_state(df)          : Dict with the latest readings + interpretation text
- regime_periods(regime)               : Contiguous runs of each regime for chart shading

All functions are pure: they take pandas Series / DataFrames in and return new objects.
No side effects, no I/O.

Author: HRC Pipeline / Liquidity module
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# =============================================================================
# DERIVED SERIES
# =============================================================================

def compute_derived_series(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived liquidity columns to a dataframe with the 5 base inputs.

    Required input columns
    ----------------------
    WACR, Repo_Rate, GSec_10Y, CRR, Bank_Credit

    Adds
    ----
    WACR_Spread        = WACR - Repo_Rate                (bps of liquidity tightness)
    GSec_Repo_Spread   = GSec_10Y - Repo_Rate            (term-premium / risk signal)
    Bank_Credit_YoY    = 12-month % change in Bank_Credit
    Repo_6M_Change     = 6-month change in Repo_Rate     (policy stance signal)
    """
    out = df.copy()

    if 'WACR' in out.columns and 'Repo_Rate' in out.columns:
        out['WACR_Spread'] = out['WACR'] - out['Repo_Rate']

    if 'GSec_10Y' in out.columns and 'Repo_Rate' in out.columns:
        out['GSec_Repo_Spread'] = out['GSec_10Y'] - out['Repo_Rate']

    if 'Bank_Credit' in out.columns:
        out['Bank_Credit_YoY'] = out['Bank_Credit'].pct_change(12) * 100

    if 'Repo_Rate' in out.columns:
        out['Repo_6M_Change'] = out['Repo_Rate'] - out['Repo_Rate'].shift(6)

    return out


# =============================================================================
# LIQUIDITY REGIME CLASSIFICATION
# =============================================================================

def classify_liquidity_regime(
    spread: pd.Series,
    method: str = 'std_band',
    band_width: float = 0.5,
) -> pd.Series:
    """Classify each observation into Tight / Neutral / Surplus.

    Parameters
    ----------
    spread     : WACR - Repo_Rate series
    method     : 'std_band' (default) uses mean ± band_width * std
                 'fixed'   uses [-0.10, +0.05] thresholds (in %)
                 'rolling' uses rolling 24m std-band (adaptive)
    band_width : multiplier of std for the band (only 'std_band' / 'rolling')

    Returns
    -------
    Series of strings: 'Tight', 'Neutral', 'Surplus'
    """
    s = spread.copy()

    if method == 'std_band':
        m, sd = s.mean(), s.std()
        lo, hi = m - band_width * sd, m + band_width * sd
        return pd.Series(
            np.where(s > hi, 'Tight', np.where(s < lo, 'Surplus', 'Neutral')),
            index=s.index, name='Liquidity_Regime'
        )

    if method == 'fixed':
        return pd.Series(
            np.where(s > 0.05, 'Tight', np.where(s < -0.10, 'Surplus', 'Neutral')),
            index=s.index, name='Liquidity_Regime'
        )

    if method == 'rolling':
        m = s.rolling(24, min_periods=12).mean()
        sd = s.rolling(24, min_periods=12).std()
        lo, hi = m - band_width * sd, m + band_width * sd
        return pd.Series(
            np.where(s > hi, 'Tight', np.where(s < lo, 'Surplus', 'Neutral')),
            index=s.index, name='Liquidity_Regime'
        )

    raise ValueError(f"Unknown method: {method!r}")


def regime_periods(regime: pd.Series) -> List[Dict]:
    """Collapse a regime series into contiguous {start, end, regime} dicts.

    Useful for shading chart backgrounds with horizontal bands per regime.
    """
    s = regime.dropna()
    if s.empty:
        return []

    blocks: List[Dict] = []
    current = s.iloc[0]
    start = s.index[0]
    for ts, r in s.items():
        if r != current:
            blocks.append({'start': start, 'end': prev_ts, 'regime': current})
            current = r
            start = ts
        prev_ts = ts
    blocks.append({'start': start, 'end': s.index[-1], 'regime': current})
    return blocks


# =============================================================================
# LIQUIDITY STRESS INDEX (0-100)
# =============================================================================

def compute_stress_index(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """A z-score composite mapped onto 0-100.

    Components (each z-scored over the full sample, then averaged):
      + WACR_Spread          (positive z → tight)
      + GSec_Repo_Spread     (positive z → term premium widening → tight)
      - Bank_Credit_YoY      (low credit growth → stress; sign flipped)
      + Repo_6M_Change       (recent hikes → stress)

    Mapping: 50 = average conditions, >60 = stress, >75 = severe stress,
             <40 = abundant liquidity, <25 = very abundant.

    Returns a Series in 0-100 with name 'Stress_Index'.
    """
    if weights is None:
        weights = {
            'WACR_Spread': 0.40,
            'GSec_Repo_Spread': 0.20,
            'Bank_Credit_YoY': 0.20,   # sign flipped inside
            'Repo_6M_Change': 0.20,
        }

    z_components = []
    used_weights = []

    for col, w in weights.items():
        if col not in df.columns:
            continue
        x = df[col]
        if col == 'Bank_Credit_YoY':
            x = -x  # higher credit growth → less stress
        mu, sd = x.mean(), x.std()
        if sd == 0 or np.isnan(sd):
            continue
        z = (x - mu) / sd
        z_components.append(z * w)
        used_weights.append(w)

    if not z_components:
        return pd.Series(np.nan, index=df.index, name='Stress_Index')

    total_w = sum(used_weights)
    composite_z = sum(z_components) / total_w

    # Map z-scores to 0-100: 0 z = 50, ±2.5 z = 0/100 (clipped)
    stress = 50 + composite_z * 20
    stress = stress.clip(0, 100)
    stress.name = 'Stress_Index'
    return stress


# =============================================================================
# MONETARY POLICY REGIME
# =============================================================================

def detect_policy_regime(repo: pd.Series, lookback_months: int = 6) -> pd.Series:
    """Classify the monetary-policy stance based on rolling repo changes.

    Categories
    ----------
    'Aggressive Hiking' : >=125 bps cumulative hike over lookback
    'Hiking'            : 25-124 bps hike
    'Pause'             : -24 to +24 bps
    'Easing'            : 25-124 bps cut
    'Aggressive Easing' : >=125 bps cut
    """
    change = repo - repo.shift(lookback_months)
    change = change * 100  # convert to bps

    def classify(c):
        if pd.isna(c):
            return np.nan
        if c >= 125:
            return 'Aggressive Hiking'
        if c >= 25:
            return 'Hiking'
        if c > -25:
            return 'Pause'
        if c > -125:
            return 'Easing'
        return 'Aggressive Easing'

    return change.map(classify).rename('Policy_Regime')


# =============================================================================
# REGIME PERFORMANCE TABLE
# =============================================================================

def regime_performance(price: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """HRC performance statistics conditional on liquidity regime.

    Aligns the two series on index, computes month-over-month returns,
    and aggregates by regime label.

    Returns columns: Months, Avg_Return_MoM, Annualized_Return,
                     Volatility_Ann, Max_Drawdown, Hit_Rate_Pct
    """
    df = pd.concat([price.rename('price'), regime.rename('regime')], axis=1).dropna()
    df = df.sort_index()
    df['ret'] = df['price'].pct_change()

    rows = []
    for r, grp in df.groupby('regime'):
        ret = grp['ret'].dropna()
        if len(ret) == 0:
            continue
        # Max drawdown within this regime's months (peak-to-trough on cumulative)
        cum = (1 + ret).cumprod()
        running_max = cum.cummax()
        dd = (cum / running_max - 1)
        rows.append({
            'Regime': r,
            'Months': len(ret),
            'Avg_Return_MoM_%': ret.mean() * 100,
            'Annualized_Return_%': (((1 + ret.mean()) ** 12) - 1) * 100,
            'Volatility_Ann_%': ret.std() * np.sqrt(12) * 100,
            'Max_Drawdown_%': dd.min() * 100,
            'Hit_Rate_%': (ret > 0).mean() * 100,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index('Regime')
    # Order: Surplus → Neutral → Tight if all present
    pref = ['Surplus', 'Neutral', 'Tight']
    order = [r for r in pref if r in out.index] + [r for r in out.index if r not in pref]
    return out.loc[order].round(2)


# =============================================================================
# LEAD-LAG: does liquidity lead HRC?
# =============================================================================

def liquidity_lead_lag(
    price: pd.Series,
    liquidity_vars: pd.DataFrame,
    lags: List[int] = [0, 1, 3, 6, 12],
) -> pd.DataFrame:
    """Correlation of HRC log returns against lagged liquidity variables.

    Parameters
    ----------
    price          : HRC price series, indexed by month
    liquidity_vars : DataFrame with one column per liquidity variable
    lags           : list of lag months to test (positive = liquidity leads)

    Returns
    -------
    DataFrame with rows = variables, columns = lags (in months),
    values = Pearson correlation of HRC return with the lagged variable.
    A positive lag means liquidity from k months ago correlates with HRC today.
    """
    hrc_ret = np.log(price / price.shift(1)).dropna()
    out = {}

    for var in liquidity_vars.columns:
        v = liquidity_vars[var]
        row = {}
        for k in lags:
            v_lag = v.shift(k)
            df = pd.concat([hrc_ret.rename('r'), v_lag.rename('v')], axis=1).dropna()
            if len(df) < 12:
                row[f'lag_{k}m'] = np.nan
            else:
                row[f'lag_{k}m'] = df['r'].corr(df['v'])
        out[var] = row

    return pd.DataFrame(out).T.round(3)


# =============================================================================
# CURRENT-STATE SUMMARY (for dashboard headline)
# =============================================================================

def summarize_current_state(df: pd.DataFrame) -> Dict:
    """Return a compact dict with the latest readings for header cards.

    Looks at the most recent row with non-null core values.
    """
    cols_needed = ['WACR', 'Repo_Rate', 'WACR_Spread', 'GSec_10Y', 'Stress_Index',
                   'Liquidity_Regime', 'Policy_Regime', 'Bank_Credit_YoY']
    latest = df[df['WACR_Spread'].notna()].tail(1)
    if latest.empty:
        return {}

    row = latest.iloc[0]
    date = latest.index[0] if latest.index.dtype.kind == 'M' else row.get('Date', None)

    state = {'date': date}
    for c in cols_needed:
        if c in row.index:
            state[c] = row[c]

    # Trend signals: vs 3 months ago
    if len(df) >= 4:
        prev = df[df['WACR_Spread'].notna()].iloc[-4]
        for c in ['WACR_Spread', 'GSec_10Y', 'Stress_Index']:
            if c in row.index and c in prev.index:
                state[f'{c}_3m_change'] = row[c] - prev[c]

    return state


def interpret_current_state(state: Dict) -> str:
    """Produce a 2-3 sentence interpretation of the current liquidity state.

    Template-based: reads regime + stress index + trend and picks language.
    """
    if not state or 'Liquidity_Regime' not in state:
        return "Insufficient data to assess current liquidity state."

    regime = state.get('Liquidity_Regime')
    stress = state.get('Stress_Index', 50)
    spread_bps = state.get('WACR_Spread', 0) * 100  # to basis points
    policy = state.get('Policy_Regime', 'Pause')
    spread_chg = state.get('WACR_Spread_3m_change', 0) * 100

    # --- Headline line based on regime ---
    if regime == 'Surplus':
        head = (f"Banking-system liquidity is in **surplus** — WACR is trading {abs(spread_bps):.0f} bps "
                f"below the policy repo rate. ")
    elif regime == 'Tight':
        head = (f"Liquidity conditions are **tight** — WACR has moved {spread_bps:+.0f} bps above the policy repo, "
                f"signalling stress in the overnight call money market. ")
    else:
        head = (f"Liquidity is **broadly neutral** — WACR is trading within {abs(spread_bps):.0f} bps of the policy "
                f"repo rate. ")

    # --- Stress contextualisation ---
    if stress < 30:
        stress_line = "The composite stress index sits in the *abundant-liquidity* band, historically supportive of industrial activity and steel demand. "
    elif stress < 50:
        stress_line = "The composite stress index is below average, consistent with accommodative funding conditions. "
    elif stress < 65:
        stress_line = "The composite stress index is mildly elevated; conditions are not stressed but bear monitoring. "
    elif stress < 80:
        stress_line = "The composite stress index is elevated, indicating tighter funding for distributors and steel consumers. "
    else:
        stress_line = "The composite stress index is in the *high-stress* band, historically associated with weak steel demand and inventory destocking. "

    # --- Policy stance and direction ---
    if policy in ('Easing', 'Aggressive Easing'):
        policy_line = (f"RBI is in an **{policy.lower()}** cycle, which historically supports Indian HRC prices "
                       f"with a 2-5 month lag through stronger credit growth and demand recovery.")
    elif policy in ('Hiking', 'Aggressive Hiking'):
        policy_line = (f"RBI is in a **{policy.lower()}** cycle. Historically this dampens steel demand by raising "
                       f"financing costs for distributors and downstream consumers.")
    else:
        policy_line = ("RBI is on **pause**; the policy stance is currently neutral and HRC direction will be driven "
                       "by global demand and input-cost developments more than by domestic monetary forces.")

    return head + stress_line + policy_line


# =============================================================================
# SELF-TEST WHEN RUN AS SCRIPT
# =============================================================================

if __name__ == '__main__':
    import sys
    # Tiny smoke test on synthetic data
    dates = pd.date_range('2020-01-31', periods=60, freq='ME')
    df = pd.DataFrame({
        'WACR': np.random.normal(5, 0.5, 60),
        'Repo_Rate': np.full(60, 5.0),
        'GSec_10Y': np.random.normal(6.5, 0.3, 60),
        'CRR': np.full(60, 4.0),
        'Bank_Credit': np.linspace(10_000_000, 18_000_000, 60),
    }, index=dates)

    df = compute_derived_series(df)
    df['Liquidity_Regime'] = classify_liquidity_regime(df['WACR_Spread'])
    df['Stress_Index'] = compute_stress_index(df)
    df['Policy_Regime'] = detect_policy_regime(df['Repo_Rate'])

    price = pd.Series(np.cumsum(np.random.normal(0, 200, 60)) + 50000, index=dates, name='India_HRC')

    print("Derived series OK")
    print(df.tail(3))
    print("\nRegime performance:")
    print(regime_performance(price, df['Liquidity_Regime']))
    print("\nLead-lag:")
    print(liquidity_lead_lag(price, df[['WACR_Spread', 'GSec_Repo_Spread']]))
    print("\nState summary:")
    state = summarize_current_state(df)
    print(state)
    print("\nInterpretation:")
    print(interpret_current_state(state))
