"""
Cyclicity & regime analysis for HRC steel prices.

Implements the methodology from Tata's reference cyclicity report:
    1. Multi-feature GMM regime identification (4 regimes by default)
    2. Peak/trough detection with prominence + minimum-distance constraints
    3. Per-regime descriptive statistics (mean, std dev, returns, spell durations)
    4. Spectral analysis (FFT periodogram) for dominant cycle period per regime
    5. Per-regime GARCH(1,1) volatility persistence
    6. Markov transition probability matrix between regimes
    7. Macro cycle identification from regime sequence

Outputs everything as a CyclicityResult dataclass — the report builder and
dashboard consume the same object. Module is self-contained and degrades
gracefully if optional dependencies (e.g., arch for GARCH) aren't installed.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ---------- Data containers ----------

@dataclass
class RegimeProfile:
    """One regime's full statistical profile."""
    regime_id: int
    label: str                                    # e.g., 'Demand Destruction'
    n_months: int
    mean_target: float
    std_target: float
    avg_monthly_return_pct: float                 # average month-on-month % change
    n_spells: int                                 # number of times the market entered this regime
    avg_spell_duration: float                     # average length of each spell in months
    first_seen: Optional[str]
    last_seen: Optional[str]


@dataclass
class CycleStats:
    """Peak/trough cycle statistics — full sample or per regime."""
    n_peaks: int
    n_troughs: int
    avg_peak_to_peak_months: Optional[float]
    avg_trough_to_trough_months: Optional[float]
    avg_amplitude_pct: Optional[float]            # peak vs adjacent trough, %
    dominant_spectral_period_months: Optional[float]


@dataclass
class CyclicityResult:
    """Master container for all cyclicity outputs for one region."""
    region: str
    target: str
    currency: str

    # Regimes
    n_regimes: int
    regime_labels: pd.Series                      # int label per month, indexed by date
    regime_profiles: List[RegimeProfile]
    current_regime: int

    # Peak/trough cycles (full sample)
    peaks: pd.Series                              # date -> peak target value
    troughs: pd.Series                            # date -> trough target value
    cycle_stats: CycleStats

    # Per-regime cycles
    per_regime_cycles: Dict[int, CycleStats]

    # Per-regime GARCH persistence
    garch_persistence: Dict[int, float]           # regime_id -> alpha + beta

    # Markov transitions
    transition_matrix: pd.DataFrame               # rows = from, cols = to
    self_persistence: Dict[int, float]            # diagonal of transition matrix
    expected_duration: Dict[int, float]           # 1 / (1 - p_self) in months

    # Macro cycle identification
    macro_cycles: pd.DataFrame                    # one row per identified cycle

    # Canonical regime ordering used by labeller
    regime_canonical_order: List[int] = field(default_factory=list)

    # Behavioural fingerprint of each regime (vol/trend/drawdown/coherence)
    regime_fingerprint: Optional[pd.DataFrame] = None
    # Per-regime driver correlation + gated lead/lag (the funnel stress-test)
    per_regime_driver_stats: Dict[int, dict] = field(default_factory=dict)
    # Window used for rolling features / coherence
    feature_window: int = 6


# ---------- Step 1: Multi-feature GMM ----------

def _driver_coherence(target: pd.Series, drivers: pd.DataFrame,
                       window: int = 6, max_drivers: int = 3) -> pd.Series:
    """
    Rolling explanatory power of the drivers over HRC returns.

    For each month we regress HRC returns on driver returns over a trailing
    `window`-month window and take the R-squared. High coherence => price
    moves are explained by fundamentals. Low coherence => price is detaching
    from its drivers (the statistical signature of a 'disrupted' regime).

    IMPORTANT: with monthly data a 6-month window has only 6 rows, so the
    regression must use a SMALL driver set or R-squared is mechanically 1.0
    (more predictors than observations). We therefore use the `max_drivers`
    drivers most correlated with HRC returns on the full sample — a proxy for
    the trusted leading drivers surfaced by the diagnostics + lead/lag steps.

    Returns a Series indexed by date. If no drivers are available the caller
    should simply omit this feature (US / overview_only regions).
    """
    y_ret = np.log(target.replace(0, np.nan)).diff()

    # Per-driver returns: log-return for strictly-positive price series,
    # simple difference for anything that can be zero or negative (spreads,
    # rates, WACR_Spread etc.). Taking log of a negative spread produces NaN
    # and would silently destroy the sample.
    x_ret_cols = {}
    for col in drivers.columns:
        s = drivers[col]
        if (s.dropna() > 0).all():
            x_ret_cols[col] = np.log(s.replace(0, np.nan)).diff()
        else:
            x_ret_cols[col] = s.diff()
    x_ret = pd.DataFrame(x_ret_cols)
    full = pd.concat([y_ret.rename("_y"), x_ret], axis=1).dropna()
    if len(full) < window + 2 or full.shape[1] < 2:
        return pd.Series(dtype=float)

    # Pick the most informative drivers (|correlation| with HRC returns)
    driver_cols = [c for c in full.columns if c != "_y"]
    corrs = {c: abs(full["_y"].corr(full[c])) for c in driver_cols}
    keep = sorted(corrs, key=corrs.get, reverse=True)[:max_drivers]
    aligned = full[["_y"] + keep]

    coh = {}
    for i in range(window - 1, len(aligned)):
        win = aligned.iloc[i - window + 1:i + 1]
        yv = win["_y"].values
        Xv = win[keep].values
        if np.std(yv) < 1e-9:
            coh[aligned.index[i]] = 0.0
            continue
        Xa = np.column_stack([np.ones(len(Xv)), Xv])
        try:
            beta, *_ = np.linalg.lstsq(Xa, yv, rcond=None)
            resid = yv - Xa @ beta
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            # adjust for small-sample inflation
            n, k = len(yv), len(keep)
            if n - k - 1 > 0:
                r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
            coh[aligned.index[i]] = float(min(max(r2, 0.0), 1.0))
        except Exception:
            coh[aligned.index[i]] = 0.0
    s = pd.Series(coh, name="coherence")
    # Back-fill the warm-up months rather than dropping them — coherence is
    # the most lag-hungry feature and must not shrink the usable sample.
    s = s.reindex(aligned.index).bfill().ffill()
    return s


def _build_features(target: pd.Series,
                     drivers: Optional[pd.DataFrame] = None,
                     window: int = 6) -> pd.DataFrame:
    """
    Construct the behavioural feature vector for regime clustering.

    Price LEVEL is deliberately excluded — regimes describe how the price is
    behaving, not where it sits. Features:
        (1) 1-month log-return            -> short-term direction
        (2) 3-month cumulative log-return -> medium-term direction
        (3) 6-month momentum (% change)   -> trend
        (4) 3-month rolling volatility    -> turbulence
        (5) drawdown from running peak    -> distance below the highs
        (6) driver coherence (optional)   -> fundamental vs disrupted

    Feature (6) is included only when `drivers` is provided and non-empty
    (China, India). For overview_only regions (US) the engine runs on the
    five price-only features — the 'lite' engine.
    """
    log_p = np.log(target.replace(0, np.nan))
    r1 = log_p.diff()
    r3 = log_p.diff(3)
    vol3 = r1.rolling(3).std()
    mom6 = target.pct_change(6) * 100
    running_peak = target.cummax()
    drawdown = (target - running_peak) / running_peak * 100

    cols = {
        "ret_1m": r1,
        "ret_3m": r3,
        "vol_3m": vol3,
        "mom_6m": mom6,
        "drawdown": drawdown,
    }

    if drivers is not None and drivers.shape[1] > 0:
        coh = _driver_coherence(target, drivers, window=window)
        if len(coh) > 0:
            cols["coherence"] = coh

    feats = pd.DataFrame(cols).dropna()
    return feats


def _label_by_fingerprint(labels: np.ndarray, feats: pd.DataFrame,
                           n_regimes: int) -> tuple:
    """
    Assign behavioural labels by reading each cluster's feature fingerprint.

    Regimes are NOT ordered by price level. They are characterised by their
    mean volatility, trend (6m momentum) and driver coherence, then named:

        Stable Range          - low vol, flat trend
        Fundamental Uptrend   - rising trend, coherence holds
        Fundamental Downtrend - falling trend, coherence holds
        Disrupted / High-Vol  - high vol and/or coherence collapse

    The cluster ids are then re-indexed 0..n-1 in a canonical *behavioural*
    order (calm -> trending -> disrupted) so downstream code is deterministic.
    Returns (new_labels array, ordered_label_list, fingerprint DataFrame).
    """
    fp_rows = []
    for c in sorted(set(labels)):
        mask = labels == c
        seg = feats.iloc[mask]
        fp_rows.append({
            "cluster": c,
            "vol": float(seg["vol_3m"].mean()),
            "trend": float(seg["mom_6m"].mean()),
            "drawdown": float(seg["drawdown"].mean()),
            "coherence": float(seg["coherence"].mean())
                          if "coherence" in seg else float("nan"),
            "n": int(mask.sum()),
        })
    fp = pd.DataFrame(fp_rows).set_index("cluster")

    has_coh = fp["coherence"].notna().any() and fp["coherence"].std() > 1e-6

    # 'Disrupted' = the cluster(s) that genuinely stand out, judged RELATIVE
    # to the other clusters rather than against a fixed cutoff:
    #   - the single highest-volatility cluster is always Disrupted
    #   - additionally, any cluster whose volatility exceeds the median by a
    #     clear margin (>1.4x), or whose coherence is the lowest AND well
    #     below the others, is Disrupted.
    vol_median = fp["vol"].median()
    top_vol_cluster = fp["vol"].idxmax()
    disrupted = set()
    for c in fp.index:
        is_top_vol = (c == top_vol_cluster)
        vol_outlier = fp.loc[c, "vol"] > 1.4 * vol_median
        coh_outlier = False
        if has_coh:
            coh_min = fp["coherence"].min()
            coh_outlier = (fp.loc[c, "coherence"] == coh_min and
                           fp.loc[c, "coherence"] < fp["coherence"].median() - 0.15)
        if is_top_vol or vol_outlier or coh_outlier:
            disrupted.add(c)

    def _name(c) -> str:
        if c in disrupted:
            return "Disrupted / High-Vol"
        trend = fp.loc[c, "trend"]
        if trend >= 4.0:
            return "Fundamental Uptrend"
        if trend <= -4.0:
            return "Fundamental Downtrend"
        return "Stable Range"

    fp["label"] = [_name(c) for c in fp.index]

    # Disambiguate same-family duplicates (e.g. two 'Fundamental Downtrend'
    # clusters) by appending a volatility qualifier so every regime label is
    # unique — required for clean transition-language and colour mapping.
    for fam in fp["label"].unique():
        members = fp.index[fp["label"] == fam].tolist()
        if len(members) > 1:
            ranked = fp.loc[members, "vol"].sort_values()
            if len(members) == 2:
                quals = ["(mild)", "(sharp)"]
            else:
                quals = [f"(tier {i+1})" for i in range(len(members))]
            for q, cl in zip(quals, ranked.index):
                fp.loc[cl, "label"] = f"{fam} {q}"

    # Canonical sort: calm first, then up, down, disrupted last.
    # Strip any "(qualifier)" so the family rank still applies.
    rank = {"Stable Range": 0, "Fundamental Uptrend": 1,
            "Fundamental Downtrend": 2, "Disrupted / High-Vol": 3}
    def _fam(lbl: str) -> str:
        return lbl.split(" (")[0]
    fp["sort_key"] = fp["label"].map(lambda l: rank.get(_fam(l), 9)) \
                     + fp["vol"].rank(pct=True) * 0.1
    fp = fp.sort_values("sort_key")
    relabel = {old: new for new, old in enumerate(fp.index)}
    new_labels = np.array([relabel[l] for l in labels])
    fp_ordered = fp.reset_index()
    fp_ordered.index = range(len(fp_ordered))
    ordered_labels = list(fp_ordered["label"])
    return new_labels, ordered_labels, fp_ordered


def _label_regimes_by_target(labels: np.ndarray, target_aligned: pd.Series,
                              n_regimes: int) -> tuple:
    """Legacy price-ordering labeller. Retained for backward compatibility
    only; the engine now uses _label_by_fingerprint."""
    means = pd.Series(target_aligned.values).groupby(labels).mean().sort_values()
    relabel = {old: new for new, old in enumerate(means.index)}
    new_labels = np.array([relabel[l] for l in labels])
    canonical_order = list(range(n_regimes))
    return new_labels, canonical_order


def _fit_gmm(target: pd.Series, drivers: Optional[pd.DataFrame] = None,
             n_regimes: int = 4, random_state: int = 42,
             window: int = 6) -> tuple:
    """Returns (labels Series, ordered label list, fingerprint DataFrame)."""
    feats = _build_features(target, drivers=drivers, window=window)
    if len(feats) < n_regimes * 5:
        raise ValueError(
            f"Not enough observations ({len(feats)}) for {n_regimes}-regime GMM. "
            f"Need at least {n_regimes * 5}."
        )
    X = StandardScaler().fit_transform(feats.values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                                random_state=random_state, n_init=5,
                                max_iter=200)
        raw_labels = gmm.fit_predict(X)
    new_labels, ordered_labels, fingerprint = _label_by_fingerprint(
        raw_labels, feats, n_regimes)
    labels_series = pd.Series(new_labels, index=feats.index, name="regime")
    return labels_series, ordered_labels, fingerprint


# ---------- Step 2: Regime profiles ----------

# Canonical economic labels — sorted ascending by mean target
DEFAULT_REGIME_LABELS_4 = [
    "Demand Destruction / Stagnation",
    "Recovery / Re-Stocking",
    "Tightening / Supply Stress",
    "Super-Cycle / Demand Surge",
]
DEFAULT_REGIME_LABELS_3 = [
    "Compressed / Stagnation",
    "Normal / Range-Bound",
    "Expanded / Demand Surge",
]
DEFAULT_REGIME_LABELS_2 = ["Low-Intensity", "High-Intensity"]


def _label_for(n_regimes: int, idx: int) -> str:
    if n_regimes == 4: return DEFAULT_REGIME_LABELS_4[idx]
    if n_regimes == 3: return DEFAULT_REGIME_LABELS_3[idx]
    if n_regimes == 2: return DEFAULT_REGIME_LABELS_2[idx]
    return f"Regime {idx}"


def _spell_stats(labels: pd.Series, regime_id: int) -> tuple:
    """Count spells of regime_id and average duration in months."""
    is_in = (labels == regime_id).astype(int).values
    if is_in.sum() == 0:
        return 0, 0.0
    diff = np.diff(np.concatenate([[0], is_in, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = ends - starts
    return int(len(starts)), float(durations.mean())


def _build_regime_profiles(target: pd.Series, labels: pd.Series,
                            n_regimes: int,
                            label_names: Optional[List[str]] = None
                            ) -> List[RegimeProfile]:
    target_in = target.loc[labels.index]
    returns = target_in.pct_change() * 100

    def _lbl(r: int) -> str:
        if label_names is not None and r < len(label_names):
            return label_names[r]
        return _label_for(n_regimes, r)

    profiles = []
    for r in range(n_regimes):
        mask = labels == r
        if mask.sum() == 0:
            profiles.append(RegimeProfile(
                regime_id=r, label=_lbl(r),
                n_months=0, mean_target=float("nan"), std_target=float("nan"),
                avg_monthly_return_pct=float("nan"),
                n_spells=0, avg_spell_duration=0.0,
                first_seen=None, last_seen=None,
            ))
            continue
        n_spells, avg_dur = _spell_stats(labels, r)
        profiles.append(RegimeProfile(
            regime_id=r,
            label=_lbl(r),
            n_months=int(mask.sum()),
            mean_target=float(target_in[mask].mean()),
            std_target=float(target_in[mask].std()),
            avg_monthly_return_pct=float(returns[mask].mean()),
            n_spells=n_spells,
            avg_spell_duration=avg_dur,
            first_seen=str(target_in[mask].index.min().date()),
            last_seen=str(target_in[mask].index.max().date()),
        ))
    return profiles


# ---------- Step 3: Peak/trough detection ----------

def _find_peaks_troughs(target: pd.Series,
                         prominence_pct: float = 5.0,
                         min_distance_months: int = 4) -> tuple:
    """
    Detect peaks and troughs. Prominence is expressed as % of target std
    so it auto-scales across China (USD) and India (INR).
    Returns (peaks_series, troughs_series).
    """
    from scipy.signal import find_peaks
    y = target.dropna().values
    idx = target.dropna().index
    prom = (prominence_pct / 100.0) * float(np.std(y))

    peak_idx, _ = find_peaks(y, prominence=prom, distance=min_distance_months)
    trough_idx, _ = find_peaks(-y, prominence=prom, distance=min_distance_months)

    peaks = pd.Series(y[peak_idx], index=idx[peak_idx])
    troughs = pd.Series(y[trough_idx], index=idx[trough_idx])
    return peaks, troughs


def _amplitude_pct(peaks: pd.Series, troughs: pd.Series) -> Optional[float]:
    """Average percentage move between each peak and the nearest adjacent trough."""
    if len(peaks) == 0 or len(troughs) == 0:
        return None
    amps = []
    all_pts = pd.concat([
        pd.DataFrame({"value": peaks, "kind": "P"}),
        pd.DataFrame({"value": troughs, "kind": "T"}),
    ]).sort_index()
    prev_kind = None; prev_val = None
    for _, row in all_pts.iterrows():
        if prev_kind is not None and prev_kind != row["kind"]:
            amps.append(abs(row["value"] - prev_val) / max(prev_val, 1e-9) * 100)
        prev_kind = row["kind"]; prev_val = row["value"]
    return float(np.mean(amps)) if amps else None


def _avg_distance_months(timestamps: pd.DatetimeIndex) -> Optional[float]:
    """Average month gap between consecutive timestamps."""
    if len(timestamps) < 2:
        return None
    diffs = []
    for i in range(1, len(timestamps)):
        d = (timestamps[i] - timestamps[i - 1]).days / 30.44
        diffs.append(d)
    return float(np.mean(diffs))


# ---------- Step 4: Spectral analysis ----------

def _dominant_period_months(target_segment: pd.Series,
                              max_period_months: int = 60) -> Optional[float]:
    """
    DFT periodogram on log-returns. Returns the dominant period in months,
    capped at max_period_months to avoid trivially-long fake cycles.
    """
    s = target_segment.dropna()
    if len(s) < 8:
        return None
    log_r = np.log(s.replace(0, np.nan)).diff().dropna().values
    if len(log_r) < 6:
        return None
    n = len(log_r)
    freqs = np.fft.rfftfreq(n, d=1.0)               # cycles per month
    power = np.abs(np.fft.rfft(log_r - log_r.mean())) ** 2
    # Skip the zero-frequency (mean) component
    mask = freqs > 1.0 / max_period_months
    if not mask.any():
        return None
    power_m = power[mask]; freqs_m = freqs[mask]
    if len(power_m) == 0:
        return None
    dom_freq = freqs_m[np.argmax(power_m)]
    if dom_freq <= 0:
        return None
    return float(1.0 / dom_freq)


# ---------- Step 5: Per-regime GARCH persistence ----------

def _garch_persistence(target_segment: pd.Series) -> Optional[float]:
    """Returns alpha + beta from GARCH(1,1). None if arch isn't installed."""
    try:
        from arch import arch_model
    except ImportError:
        return None
    s = target_segment.dropna()
    if len(s) < 12:
        return None
    returns = (np.log(s).diff() * 100).dropna()
    if len(returns) < 10 or returns.std() < 1e-6:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
        params = res.params
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        return alpha + beta
    except Exception:
        return None


# ---------- Step 6: Markov transition matrix ----------

def _transition_matrix(labels: pd.Series, n_regimes: int) -> tuple:
    """Empirical first-order transition probabilities."""
    counts = np.zeros((n_regimes, n_regimes), dtype=float)
    arr = labels.values
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums
    df = pd.DataFrame(probs,
                       index=[f"From R{i}" for i in range(n_regimes)],
                       columns=[f"To R{i}" for i in range(n_regimes)])
    self_persist = {i: float(probs[i, i]) for i in range(n_regimes)}
    expected_duration = {
        i: float(1.0 / max(1.0 - probs[i, i], 1e-6)) for i in range(n_regimes)
    }
    return df, self_persist, expected_duration


# ---------- Step 7: Macro cycle identification ----------

def _identify_macro_cycles(labels: pd.Series, target: pd.Series,
                            calm_regime: int = 0) -> pd.DataFrame:
    """
    A macro cycle is a contiguous traversal from one calm phase to the next
    that contains a period of real action.

    `calm_regime` is the id of the lowest-volatility regime (the natural
    'reset' state). A reset is a run of that regime; a macro cycle is the
    span between two consecutive resets that contains at least one visit to
    a different regime (calm -> action -> calm).
    """
    if len(labels) == 0:
        return pd.DataFrame(columns=["start", "end", "duration_months",
                                       "peak_target", "primary_regimes"])
    arr = labels.values
    # Runs of the calm anchor regime are 'resets'
    is_low = (arr == calm_regime).astype(int)
    diff = np.diff(np.concatenate([[0], is_low, [0]]))
    low_starts = np.where(diff == 1)[0]
    low_ends = np.where(diff == -1)[0]

    cycles = []
    if len(low_ends) >= 2:
        for i in range(len(low_ends) - 1):
            cycle_start = low_ends[i]               # market exits a reset
            cycle_end = low_starts[i + 1]            # market enters next reset
            if cycle_end <= cycle_start:
                continue
            seg_labels = arr[cycle_start:cycle_end]
            if not (seg_labels != calm_regime).any():
                continue                            # no real action in window
            seg_target = target.loc[labels.index[cycle_start:cycle_end]]
            cycles.append({
                "start": str(labels.index[cycle_start].date()),
                "end": str(labels.index[cycle_end - 1].date()),
                "duration_months": int(cycle_end - cycle_start),
                "peak_target": float(seg_target.max()),
                "primary_regimes": ", ".join(
                    f"R{r}" for r in sorted(set(seg_labels)) if r != calm_regime
                ),
            })
    return pd.DataFrame(cycles)


# ---------- Step 6.5: Per-regime driver statistics ----------

LEADLAG_MIN_MONTHS = 15   # below this, per-regime lead/lag is not reported


def _per_regime_driver_stats(target: pd.Series, drivers: pd.DataFrame,
                              labels: pd.Series, n_regimes: int,
                              max_lag: int = 6) -> Dict[int, dict]:
    """
    Within each regime's months, compute:
      - correlation of HRC returns vs each driver's returns (always)
      - best lead/lag of each driver vs HRC (only if regime has >= 15 months;
        otherwise flagged 'insufficient sample')

    This is the funnel stress-test: the same diagnostics + lead/lag statistics
    from earlier tabs, recomputed regime by regime, to show they are not
    stable across market states.
    """
    out: Dict[int, dict] = {}
    if drivers is None or drivers.shape[1] == 0:
        return out

    y_ret = np.log(target.replace(0, np.nan)).diff()
    # log-return for positive price series, simple diff for spreads/rates
    x_ret_cols = {}
    for col in drivers.columns:
        s = drivers[col]
        if (s.dropna() > 0).all():
            x_ret_cols[col] = np.log(s.replace(0, np.nan)).diff()
        else:
            x_ret_cols[col] = s.diff()
    x_ret = pd.DataFrame(x_ret_cols)

    for r in range(n_regimes):
        months = labels[labels == r].index
        y_r = y_ret.loc[y_ret.index.intersection(months)].dropna()
        corr = {}
        leadlag = {}
        enough = len(y_r) >= LEADLAG_MIN_MONTHS
        for col in drivers.columns:
            xr = x_ret[col]
            joined = pd.concat([y_r.rename("y"), xr.rename("x")], axis=1).dropna()
            if len(joined) >= 3:
                corr[col] = float(joined["y"].corr(joined["x"]))
            else:
                corr[col] = float("nan")
            if enough:
                best_lag, best_abs = 0, -1.0
                full = pd.concat([y_ret.rename("y"), xr.rename("x")],
                                 axis=1).dropna()
                for lag in range(-max_lag, max_lag + 1):
                    shifted = full["x"].shift(lag)
                    sub = pd.concat([full["y"], shifted], axis=1).dropna()
                    sub = sub.loc[sub.index.intersection(months)]
                    if len(sub) >= 5:
                        c = sub.iloc[:, 0].corr(sub.iloc[:, 1])
                        if c is not None and abs(c) > best_abs:
                            best_abs, best_lag = abs(c), lag
                leadlag[col] = {"best_lag": best_lag, "abs_corr": best_abs}
            else:
                leadlag[col] = None
        out[r] = {
            "n_return_obs": int(len(y_r)),
            "correlation": corr,
            "leadlag": leadlag,
            "leadlag_available": enough,
        }
    return out




def analyse_cyclicity(target: pd.Series, region: str = "",
                       currency: str = "USD",
                       drivers: Optional[pd.DataFrame] = None,
                       n_regimes: int = 4,
                       random_state: int = 42,
                       prominence_pct: float = 5.0,
                       min_distance_months: int = 4,
                       feature_window: int = 6) -> CyclicityResult:
    """
    Run the full cyclicity pipeline on one target series.

    If `drivers` is provided, the regime engine adds a driver-coherence
    feature and computes per-regime driver statistics. For overview_only
    regions (US, no drivers) pass drivers=None — the engine runs the
    five-feature 'lite' version automatically.
    """
    target = target.dropna().sort_index()
    if len(target) < n_regimes * 5:
        # Fall back to fewer regimes if not enough data
        n_regimes = max(2, len(target) // 5)

    # Align drivers to target if provided
    if drivers is not None and drivers.shape[1] > 0:
        drivers = drivers.reindex(target.index)
    else:
        drivers = None

    # 1. Regime identification (behavioural fingerprint labelling)
    # Guard: ensure enough USABLE feature rows, not just raw months.
    _feat_n = len(_build_features(target, drivers=drivers, window=feature_window))
    if _feat_n < n_regimes * 5:
        n_regimes = max(2, _feat_n // 5)
    labels, ordered_labels, fingerprint = _fit_gmm(
        target, drivers=drivers, n_regimes=n_regimes,
        random_state=random_state, window=feature_window)
    canonical = list(range(n_regimes))
    profiles = _build_regime_profiles(target, labels, n_regimes,
                                       label_names=ordered_labels)

    # 2. Peaks/troughs (full sample)
    peaks, troughs = _find_peaks_troughs(target,
                                           prominence_pct=prominence_pct,
                                           min_distance_months=min_distance_months)
    cycle_stats = CycleStats(
        n_peaks=int(len(peaks)),
        n_troughs=int(len(troughs)),
        avg_peak_to_peak_months=_avg_distance_months(peaks.index),
        avg_trough_to_trough_months=_avg_distance_months(troughs.index),
        avg_amplitude_pct=_amplitude_pct(peaks, troughs),
        dominant_spectral_period_months=_dominant_period_months(target),
    )

    # 3. Per-regime cycles + spectral period
    per_regime_cycles = {}
    for r in range(n_regimes):
        seg = target.loc[labels[labels == r].index]
        if len(seg) < 4:
            per_regime_cycles[r] = CycleStats(0, 0, None, None, None, None)
            continue
        rp, rt = _find_peaks_troughs(seg,
                                       prominence_pct=prominence_pct,
                                       min_distance_months=min_distance_months)
        per_regime_cycles[r] = CycleStats(
            n_peaks=int(len(rp)),
            n_troughs=int(len(rt)),
            avg_peak_to_peak_months=_avg_distance_months(rp.index),
            avg_trough_to_trough_months=_avg_distance_months(rt.index),
            avg_amplitude_pct=_amplitude_pct(rp, rt),
            dominant_spectral_period_months=_dominant_period_months(seg),
        )

    # 4. Per-regime GARCH persistence
    garch_persist = {}
    for r in range(n_regimes):
        seg = target.loc[labels[labels == r].index]
        garch_persist[r] = _garch_persistence(seg) if len(seg) >= 12 else None

    # 5. Markov transitions
    trans_df, self_persist, expected_dur = _transition_matrix(labels, n_regimes)

    # 6. Macro cycles — anchored on the calmest (lowest-vol) regime
    _calm = 0
    if fingerprint is not None and "vol" in fingerprint.columns:
        _calm = int(fingerprint["vol"].astype(float).idxmin())
    macro_cycles = _identify_macro_cycles(labels, target, calm_regime=_calm)

    # 7. Per-regime driver stats (funnel stress-test)
    per_regime_stats = _per_regime_driver_stats(
        target, drivers, labels, n_regimes) if drivers is not None else {}

    return CyclicityResult(
        region=region,
        target=target.name if hasattr(target, "name") else "target",
        currency=currency,
        n_regimes=n_regimes,
        regime_labels=labels,
        regime_profiles=profiles,
        current_regime=int(labels.iloc[-1]),
        peaks=peaks,
        troughs=troughs,
        cycle_stats=cycle_stats,
        per_regime_cycles=per_regime_cycles,
        garch_persistence=garch_persist,
        transition_matrix=trans_df,
        self_persistence=self_persist,
        expected_duration=expected_dur,
        macro_cycles=macro_cycles,
        regime_canonical_order=canonical,
        regime_fingerprint=fingerprint,
        per_regime_driver_stats=per_regime_stats,
        feature_window=feature_window,
    )
