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


# ---------- Step 1: Multi-feature GMM ----------

def _build_features(target: pd.Series) -> pd.DataFrame:
    """
    Construct the 5-dim feature vector per the reference report:
        (1) log-level
        (2) 1-month log-return
        (3) 3-month cumulative log-return
        (4) 3-month rolling volatility (std of returns)
        (5) 6-month momentum (% change over 6 months)
    """
    log_p = np.log(target.replace(0, np.nan))
    r1 = log_p.diff()
    r3 = log_p.diff(3)
    vol3 = r1.rolling(3).std()
    mom6 = target.pct_change(6) * 100
    feats = pd.DataFrame({
        "log_level": log_p,
        "ret_1m": r1,
        "ret_3m": r3,
        "vol_3m": vol3,
        "mom_6m": mom6,
    }).dropna()
    return feats


def _label_regimes_by_target(labels: np.ndarray, target_aligned: pd.Series,
                              n_regimes: int) -> tuple:
    """Relabel so regime 0 has lowest mean target, regime n-1 has highest."""
    means = pd.Series(target_aligned.values).groupby(labels).mean().sort_values()
    relabel = {old: new for new, old in enumerate(means.index)}
    new_labels = np.array([relabel[l] for l in labels])
    canonical_order = list(range(n_regimes))
    return new_labels, canonical_order


def _fit_gmm(target: pd.Series, n_regimes: int = 4,
             random_state: int = 42) -> tuple:
    """Returns (labels Series indexed by date, canonical_order list)."""
    feats = _build_features(target)
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
    target_aligned = target.loc[feats.index]
    new_labels, canonical = _label_regimes_by_target(raw_labels, target_aligned,
                                                       n_regimes)
    labels_series = pd.Series(new_labels, index=feats.index, name="regime")
    return labels_series, canonical


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
                            n_regimes: int) -> List[RegimeProfile]:
    target_in = target.loc[labels.index]
    returns = target_in.pct_change() * 100
    profiles = []
    for r in range(n_regimes):
        mask = labels == r
        if mask.sum() == 0:
            profiles.append(RegimeProfile(
                regime_id=r, label=_label_for(n_regimes, r),
                n_months=0, mean_target=float("nan"), std_target=float("nan"),
                avg_monthly_return_pct=float("nan"),
                n_spells=0, avg_spell_duration=0.0,
                first_seen=None, last_seen=None,
            ))
            continue
        n_spells, avg_dur = _spell_stats(labels, r)
        profiles.append(RegimeProfile(
            regime_id=r,
            label=_label_for(n_regimes, r),
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

def _identify_macro_cycles(labels: pd.Series, target: pd.Series) -> pd.DataFrame:
    """
    A macro cycle is a contiguous traversal of regimes that contains the highest
    regime (n-1). We find each appearance of the top regime and bracket it with
    the surrounding lower-regime months. Reasonable approximation: every time
    the labels return to their lowest regime is a 'reset', so a macro cycle is
    the period between two consecutive resets that contains a top-regime visit.
    """
    if len(labels) == 0:
        return pd.DataFrame(columns=["start", "end", "duration_months",
                                       "peak_target", "primary_regimes"])
    arr = labels.values
    n_regimes = int(arr.max()) + 1
    top_r = n_regimes - 1
    # Find runs of the lowest regime — each such run is a 'reset'
    is_low = (arr == 0).astype(int)
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
            if not (seg_labels == top_r).any():
                continue                            # no super-cycle peak in this window
            seg_target = target.loc[labels.index[cycle_start:cycle_end]]
            cycles.append({
                "start": str(labels.index[cycle_start].date()),
                "end": str(labels.index[cycle_end - 1].date()),
                "duration_months": int(cycle_end - cycle_start),
                "peak_target": float(seg_target.max()),
                "primary_regimes": ", ".join(
                    f"R{r}" for r in sorted(set(seg_labels)) if r != 0
                ),
            })
    return pd.DataFrame(cycles)


# ---------- Master entry point ----------

def analyse_cyclicity(target: pd.Series, region: str = "",
                       currency: str = "USD",
                       n_regimes: int = 4,
                       random_state: int = 42,
                       prominence_pct: float = 5.0,
                       min_distance_months: int = 4) -> CyclicityResult:
    """
    Run the full cyclicity pipeline on one target series.
    """
    target = target.dropna().sort_index()
    if len(target) < n_regimes * 5:
        # Fall back to fewer regimes if not enough data
        n_regimes = max(2, len(target) // 5)

    # 1. Regime identification
    labels, canonical = _fit_gmm(target, n_regimes=n_regimes,
                                   random_state=random_state)
    profiles = _build_regime_profiles(target, labels, n_regimes)

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

    # 6. Macro cycles
    macro_cycles = _identify_macro_cycles(labels, target)

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
    )
