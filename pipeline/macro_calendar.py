"""
Macro Calendar Module — HRC-relevant events with historical analogue analysis.

Reads `data/macro_calendar.yaml` and, for each upcoming event, computes:
    1. Current market setup (iron ore percentile, spread regime, cyclicity regime)
    2. Conditional historical analogues — past months with similar setup
    3. Simple historical analogues — average HRC move N days after past
       releases of the same event category (fallback when conditional too thin)
    4. Confidence label based on how many usable analogues exist

Output container is consumed by the narrator + report builder + dashboard.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import numpy as np
import pandas as pd


# ---------- Result containers ----------

@dataclass
class Analogue:
    """A single historical analogue — past month + what HRC did afterwards."""
    setup_month: str                              # e.g., "2022-08"
    setup_iron_ore_pct: float                     # iron ore percentile at setup
    setup_spread_regime: str                      # 'compressed', 'normal', etc.
    hrc_change_30d_pct: Optional[float]
    hrc_change_60d_pct: Optional[float]
    hrc_change_90d_pct: Optional[float]


@dataclass
class EventAnalysis:
    """One macro event + its analogues + setup."""
    # Static config
    event_date: str
    name: str
    country: str
    impact: str                                   # HIGH/MED/LOW
    category: str
    consensus: str
    affects_regions: List[str]
    primary_channel: str
    mechanism: str
    expected_hrc_reaction: str

    # Computed setup at the time the report is generated
    days_until: int
    iron_ore_now: Optional[float]
    iron_ore_percentile: Optional[float]
    spread_regime: Optional[str]                  # for primary affected region
    spread_percentile: Optional[float]
    cyclicity_regime: Optional[int]

    # Historical analogues
    conditional_analogues: List[Analogue]         # filtered by setup similarity
    simple_analogues: List[Analogue]              # all past months of same category
    analogue_confidence: str                      # 'high', 'medium', 'low', 'none'

    # Aggregated forward statistics
    cond_avg_30d: Optional[float]
    cond_avg_60d: Optional[float]
    cond_avg_90d: Optional[float]
    cond_n: int
    simple_avg_30d: Optional[float]
    simple_avg_60d: Optional[float]
    simple_avg_90d: Optional[float]
    simple_n: int


@dataclass
class CalendarResult:
    """Master container for the full macro calendar analysis."""
    window_start: str
    window_end: str
    window_description: str
    window_days: int                              # how many days the window spans
    events: List[EventAnalysis]                   # filtered to current window
    n_high: int
    n_med: int
    n_low: int
    n_total_in_library: int                       # total events in YAML
    today: str


# ---------- Helpers ----------

def _percentile_rank(value: float, series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return None
    return float((s < value).mean() * 100)


def _classify_spread_regime(percentile: float) -> str:
    if percentile is None or pd.isna(percentile):
        return "unknown"
    if percentile <= 25: return "compressed"
    if percentile >= 75: return "expanded"
    return "normal"


def _hrc_change_pct(target: pd.Series, ref_idx: pd.Timestamp,
                    months_forward: int) -> Optional[float]:
    """% change in HRC from ref_idx to (ref_idx + months_forward months)."""
    try:
        future_idx = ref_idx + pd.DateOffset(months=months_forward)
        # Find nearest available observation within ±15 days
        candidates = target.index[
            (target.index >= future_idx - pd.Timedelta(days=15)) &
            (target.index <= future_idx + pd.Timedelta(days=15))
        ]
        if len(candidates) == 0:
            return None
        future_val = target.loc[candidates[0]]
        ref_val = target.loc[ref_idx]
        if pd.isna(future_val) or pd.isna(ref_val) or ref_val == 0:
            return None
        return float((future_val - ref_val) / ref_val * 100)
    except Exception:
        return None


# ---------- Analogue computation ----------

def _build_setup_at_each_month(china_data, india_data) -> pd.DataFrame:
    """
    For each historical month, compute the setup vector that we'll match
    against today's setup. Used by the conditional-analogue search.

    Returns DataFrame indexed by month with columns:
        iron_ore_pct, spread_pct_china, spread_regime_china,
        spread_pct_india, spread_regime_india
    """
    rows = []

    # China iron ore (CFR)
    china_df = china_data.df
    china_target = china_data.y
    iron_ore_col = china_data.spread_config.get("iron_ore_column")
    hcc_col = china_data.spread_config.get("hcc_column")
    iron_ore_w = china_data.spread_config.get("iron_ore_weight", 1.6)
    hcc_w = china_data.spread_config.get("hcc_weight", 0.9)
    iron_ore_series = china_df[iron_ore_col] if iron_ore_col in china_df.columns else None
    hcc_series = china_df[hcc_col] if hcc_col in china_df.columns else None

    if iron_ore_series is not None and hcc_series is not None:
        china_spread = (china_target -
                          (iron_ore_w * iron_ore_series + hcc_w * hcc_series))
    else:
        china_spread = pd.Series(dtype=float)

    # India spread (if available)
    india_spread = pd.Series(dtype=float)
    if india_data is not None:
        india_df = india_data.df
        india_target = india_data.y
        iron_ore_in = india_data.spread_config.get("iron_ore_column")
        hcc_in = india_data.spread_config.get("hcc_column")
        iron_ore_w_in = india_data.spread_config.get("iron_ore_weight", 1.6)
        hcc_w_in = india_data.spread_config.get("hcc_weight", 0.9)
        if iron_ore_in in india_df.columns and hcc_in in india_df.columns:
            india_spread = (india_target -
                             (iron_ore_w_in * india_df[iron_ore_in] +
                              hcc_w_in * india_df[hcc_in]))

    # Build per-month setup
    for month in china_target.index:
        if iron_ore_series is None:
            iron_ore_pct = None
        else:
            iron_ore_pct = _percentile_rank(iron_ore_series.loc[month], iron_ore_series)

        if month in china_spread.index:
            spread_pct_china = _percentile_rank(china_spread.loc[month], china_spread)
            regime_china = _classify_spread_regime(spread_pct_china)
        else:
            spread_pct_china = None; regime_china = "unknown"

        if month in india_spread.index:
            spread_pct_india = _percentile_rank(india_spread.loc[month], india_spread)
            regime_india = _classify_spread_regime(spread_pct_india)
        else:
            spread_pct_india = None; regime_india = "unknown"

        rows.append({
            "month": month,
            "iron_ore_pct": iron_ore_pct,
            "spread_pct_china": spread_pct_china,
            "spread_regime_china": regime_china,
            "spread_pct_india": spread_pct_india,
            "spread_regime_india": regime_india,
        })
    return pd.DataFrame(rows).set_index("month")


def _find_conditional_analogues(setup_df: pd.DataFrame,
                                  target: pd.Series,
                                  current_iron_ore_pct: Optional[float],
                                  current_spread_regime: str,
                                  iron_ore_band: float = 20.0,
                                  max_results: int = 6) -> List[Analogue]:
    """
    Find past months where:
        • iron ore percentile was within ±band of current
        • spread regime matched current
    Then compute 30/60/90-day forward HRC change for each match.
    """
    if current_iron_ore_pct is None:
        return []
    candidates = setup_df[
        (setup_df["iron_ore_pct"].notna()) &
        (setup_df["iron_ore_pct"] >= current_iron_ore_pct - iron_ore_band) &
        (setup_df["iron_ore_pct"] <= current_iron_ore_pct + iron_ore_band) &
        (setup_df["spread_regime_china"] == current_spread_regime)
    ]
    # Exclude the most recent 3 months (no forward window available)
    last_idx = target.index.max()
    candidates = candidates[candidates.index <= last_idx - pd.DateOffset(months=3)]

    # Cap results — take the most recent matches for relevance
    candidates = candidates.sort_index(ascending=False).head(max_results)

    analogues = []
    for month, row in candidates.iterrows():
        analogues.append(Analogue(
            setup_month=month.strftime("%Y-%m"),
            setup_iron_ore_pct=float(row["iron_ore_pct"]),
            setup_spread_regime=row["spread_regime_china"],
            hrc_change_30d_pct=_hrc_change_pct(target, month, 1),
            hrc_change_60d_pct=_hrc_change_pct(target, month, 2),
            hrc_change_90d_pct=_hrc_change_pct(target, month, 3),
        ))
    return analogues


def _find_simple_analogues(setup_df: pd.DataFrame,
                             target: pd.Series,
                             max_results: int = 12) -> List[Analogue]:
    """
    Simple analogue: every historical month, regardless of setup match.
    Used as a fallback when conditional analogues are too thin.
    """
    last_idx = target.index.max()
    valid = setup_df[setup_df.index <= last_idx - pd.DateOffset(months=3)]
    valid = valid.sort_index(ascending=False).head(max_results)
    out = []
    for month, row in valid.iterrows():
        out.append(Analogue(
            setup_month=month.strftime("%Y-%m"),
            setup_iron_ore_pct=float(row["iron_ore_pct"]) if pd.notna(row["iron_ore_pct"]) else float("nan"),
            setup_spread_regime=row["spread_regime_china"],
            hrc_change_30d_pct=_hrc_change_pct(target, month, 1),
            hrc_change_60d_pct=_hrc_change_pct(target, month, 2),
            hrc_change_90d_pct=_hrc_change_pct(target, month, 3),
        ))
    return out


def _aggregate_analogues(analogues: List[Analogue]) -> tuple:
    """Average forward % change across analogues. Returns (avg_30, avg_60, avg_90, n)."""
    if not analogues:
        return None, None, None, 0
    a30 = [a.hrc_change_30d_pct for a in analogues if a.hrc_change_30d_pct is not None]
    a60 = [a.hrc_change_60d_pct for a in analogues if a.hrc_change_60d_pct is not None]
    a90 = [a.hrc_change_90d_pct for a in analogues if a.hrc_change_90d_pct is not None]
    avg30 = float(np.mean(a30)) if a30 else None
    avg60 = float(np.mean(a60)) if a60 else None
    avg90 = float(np.mean(a90)) if a90 else None
    return avg30, avg60, avg90, len(analogues)


def _confidence_label(n_conditional: int, n_simple: int) -> str:
    if n_conditional >= 5: return "high"
    if n_conditional >= 3: return "medium"
    if n_simple >= 6: return "low"
    return "none"


# ---------- Master entry point ----------

def analyse_macro_calendar(china_data, india_data,
                             config: dict,
                             window_days: int = 45,
                             reference_date: Optional[pd.Timestamp] = None,
                             show_past: bool = False,
                             past_days: int = 30) -> CalendarResult:
    """
    Load the macro calendar yaml, filter events to a rolling
    [today - past_days, today + window_days] window, compute analogues,
    return a CalendarResult.

    Parameters
    ----------
    window_days : int
        Forward-looking window length in days (default: 45).
    reference_date : Optional[Timestamp]
        Override "today" — useful for testing or historical replay.
    show_past : bool
        If True, also include events from the past `past_days` days.
        Default False (forward-only window).
    past_days : int
        How many days of past events to include when show_past=True.
    """
    # Path resilience: try parent of data file, then project-root data dir
    yaml_candidates = [
        Path(config["data"]["file"]).parent / "macro_calendar.yaml",
        Path("data/macro_calendar.yaml"),
        Path(__file__).parent.parent / "data" / "macro_calendar.yaml",
    ]
    yaml_path = None
    for p in yaml_candidates:
        if p.exists():
            yaml_path = p
            break
    if yaml_path is None:
        raise FileNotFoundError(
            f"Macro calendar not found. Tried:\n" +
            "\n".join(f"  • {p}" for p in yaml_candidates) +
            f"\n\nCreate data/macro_calendar.yaml to enable this section."
        )

    with open(yaml_path) as f:
        cal_cfg = yaml.safe_load(f)

    today = (reference_date if reference_date is not None
             else pd.Timestamp.now()).normalize()
    window_end = today + pd.Timedelta(days=window_days)
    window_start = today - pd.Timedelta(days=past_days) if show_past else today

    # Build setup history for analogue search
    setup_df = _build_setup_at_each_month(china_data, india_data)

    # Current setup — use the latest available month
    latest_month = setup_df.index.max()
    current_setup = setup_df.loc[latest_month]
    current_iron_ore_pct = (float(current_setup["iron_ore_pct"])
                              if pd.notna(current_setup["iron_ore_pct"]) else None)
    current_regime_china = current_setup["spread_regime_china"]

    all_yaml_events = cal_cfg.get("events", [])
    events = []
    for ev in all_yaml_events:
        try:
            ev_date = pd.Timestamp(ev["date"])
        except Exception:
            continue

        # Window filter: respects show_past flag
        if ev_date < window_start or ev_date > window_end:
            continue

        days_until = (ev_date - today).days

        # Find analogues
        cond = _find_conditional_analogues(
            setup_df, china_data.y,
            current_iron_ore_pct, current_regime_china,
        )
        simple = _find_simple_analogues(setup_df, china_data.y)

        cond_avg30, cond_avg60, cond_avg90, cond_n = _aggregate_analogues(cond)
        sim_avg30, sim_avg60, sim_avg90, sim_n = _aggregate_analogues(simple)
        confidence = _confidence_label(cond_n, sim_n)

        events.append(EventAnalysis(
            event_date=ev["date"],
            name=ev["name"],
            country=ev.get("country", ""),
            impact=ev.get("impact", "MED"),
            category=ev.get("category", ""),
            consensus=ev.get("consensus", ""),
            affects_regions=ev.get("affects_regions", []),
            primary_channel=ev.get("primary_channel", ""),
            mechanism=ev.get("mechanism", "").strip(),
            expected_hrc_reaction=ev.get("expected_hrc_reaction", "").strip(),
            days_until=days_until,
            iron_ore_now=float(setup_df.iloc[-1].get("iron_ore_pct"))
                         if pd.notna(setup_df.iloc[-1].get("iron_ore_pct")) else None,
            iron_ore_percentile=current_iron_ore_pct,
            spread_regime=current_regime_china,
            spread_percentile=(float(current_setup["spread_pct_china"])
                                if pd.notna(current_setup["spread_pct_china"]) else None),
            cyclicity_regime=None,
            conditional_analogues=cond,
            simple_analogues=simple,
            analogue_confidence=confidence,
            cond_avg_30d=cond_avg30, cond_avg_60d=cond_avg60,
            cond_avg_90d=cond_avg90, cond_n=cond_n,
            simple_avg_30d=sim_avg30, simple_avg_60d=sim_avg60,
            simple_avg_90d=sim_avg90, simple_n=sim_n,
        ))

    events.sort(key=lambda e: e.event_date)

    desc_suffix = (f" (incl. {past_days} days past)" if show_past else "")
    return CalendarResult(
        window_start=window_start.strftime("%Y-%m-%d"),
        window_end=window_end.strftime("%Y-%m-%d"),
        window_description=f"Forward {window_days}-day window from {today.strftime('%b %d, %Y')}{desc_suffix}",
        window_days=window_days,
        events=events,
        n_high=sum(1 for e in events if e.impact == "HIGH"),
        n_med=sum(1 for e in events if e.impact == "MED"),
        n_low=sum(1 for e in events if e.impact == "LOW"),
        n_total_in_library=len(all_yaml_events),
        today=today.strftime("%Y-%m-%d"),
    )
