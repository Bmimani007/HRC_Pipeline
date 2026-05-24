"""
COVID-window exclusion filter — normal-times statistics.

WHY THIS EXISTS
  The COVID era (2020 collapse + 2021 restocking/supercycle run-up) is the
  single most abnormal stretch in the HRC history. It is genuinely useful
  data — it is the only reason the regime engine has a "Disrupted / High-Vol"
  state at all — so it must NEVER be deleted. But it can also dominate the
  descriptive statistics and drown out the normal-times signal an analyst
  wants for day-to-day direction.

  This module resolves that tension. It does NOT touch the data or the engine.
  The regime engine still trains on the full history and keeps every regime.
  This filter only changes which months are COUNTED when producing the
  displayed statistics, so the dashboard can show a normal-times view ALONGSIDE
  the all-history view — a comparison, not a replacement.

WHAT IT FILTERS
  • Regime profile stats — recomputed on the COVID-masked label series
    (n_months, mean/std, avg monthly return, spell count + duration).
  • Recurrence base rates — turning points whose date falls inside the COVID
    window are dropped before the hit-rate / forward-move averages are taken.

HONEST LIMITATION
  Excluding ~23 months from an already-short monthly series shrinks the
  sample. Some ex-COVID base rates will fall below the engine's existing
  "4 comparable events" sufficiency threshold and will be flagged
  'insufficient' rather than shown as a fragile number. That is the model
  being honest, not broken.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------- Window handling ----------

@dataclass
class CovidWindow:
    """The COVID exclusion window, normally loaded from config.yaml."""
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def label(self) -> str:
        return (f"{self.start.strftime('%b %Y')} – "
                f"{self.end.strftime('%b %Y')}")

    def contains(self, date: pd.Timestamp) -> bool:
        return self.start <= date <= self.end

    def mask_out(self, index: pd.DatetimeIndex) -> pd.Series:
        """Boolean Series — True for months to KEEP (outside the window)."""
        idx = pd.DatetimeIndex(index)
        inside = (idx >= self.start) & (idx <= self.end)
        return pd.Series(~inside, index=idx)


def covid_window_from_config(config: dict) -> Optional[CovidWindow]:
    """
    Build a CovidWindow from the config dict. Returns None if the block is
    absent or malformed — callers then simply skip the ex-COVID feature.
    """
    try:
        block = (config or {}).get("analysis", {}).get("covid_window")
        if not block:
            return None
        start = pd.to_datetime(block["start"])
        end = pd.to_datetime(block["end"])
        if start >= end:
            return None
        return CovidWindow(start=start, end=end)
    except Exception:
        return None


# ---------- Regime statistics, recomputed ex-COVID ----------

@dataclass
class RegimeStatRow:
    """One regime's stats under a given counting basis (all / ex-COVID)."""
    regime_id: int
    label: str
    n_months: int
    n_months_pct: float            # share of the counted sample
    mean_target: float
    avg_monthly_return_pct: float
    n_spells: int
    avg_spell_duration: float


def _spell_stats(mask_values: np.ndarray) -> Tuple[int, float]:
    """Count contiguous spells in a 0/1 array and their average length."""
    if mask_values.sum() == 0:
        return 0, 0.0
    diff = np.diff(np.concatenate([[0], mask_values.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = ends - starts
    return int(len(starts)), float(durations.mean())


def regime_stats_table(cyc_result, target: pd.Series,
                       covid: Optional[CovidWindow],
                       exclude_covid: bool) -> List[RegimeStatRow]:
    """
    Recompute per-regime descriptive statistics on either the full label
    series or the COVID-masked one.

    `cyc_result`    — a CyclicityResult (engine output; never modified).
    `target`        — the region's HRC price series.
    `covid`         — the exclusion window, or None.
    `exclude_covid` — when True (and covid is not None), months inside the
                      window are dropped before the stats are computed.

    Note on spells: when COVID months are masked out, a spell that straddled
    the window is split. This is the honest result — we are reporting how the
    regime behaved IN NORMAL TIMES, and a spell interrupted by COVID genuinely
    is two separate normal-times spells from that point of view.
    """
    labels = cyc_result.regime_labels.copy()
    labels.index = pd.DatetimeIndex(labels.index)
    target = target.dropna()
    target.index = pd.DatetimeIndex(target.index)

    if exclude_covid and covid is not None:
        keep = covid.mask_out(labels.index)
        labels = labels[keep.values]

    target_in = target.reindex(labels.index)
    returns = target_in.pct_change() * 100
    total_counted = len(labels)

    # regime ids present in the engine result
    regime_ids = sorted(set(int(v) for v in cyc_result.regime_labels.unique()))
    # label lookup from the engine's profiles
    id_to_label = {p.regime_id: p.label for p in cyc_result.regime_profiles}

    rows: List[RegimeStatRow] = []
    for r in regime_ids:
        mask = (labels == r)
        n = int(mask.sum())
        if n == 0:
            rows.append(RegimeStatRow(
                regime_id=r, label=id_to_label.get(r, f"R{r}"),
                n_months=0, n_months_pct=0.0,
                mean_target=float("nan"),
                avg_monthly_return_pct=float("nan"),
                n_spells=0, avg_spell_duration=0.0))
            continue
        n_spells, avg_dur = _spell_stats(mask.values)
        rows.append(RegimeStatRow(
            regime_id=r,
            label=id_to_label.get(r, f"R{r}"),
            n_months=n,
            n_months_pct=(100.0 * n / total_counted
                          if total_counted > 0 else 0.0),
            mean_target=float(target_in[mask].mean()),
            avg_monthly_return_pct=float(returns[mask].mean()),
            n_spells=n_spells,
            avg_spell_duration=avg_dur))
    return rows


def regime_stats_comparison(cyc_result, target: pd.Series,
                            covid: Optional[CovidWindow]) -> pd.DataFrame:
    """
    Build a side-by-side DataFrame: each regime's key stats under the
    all-history basis and the ex-COVID basis, for direct comparison.

    Columns: Regime | Months (all) | Months (ex-COVID) | Avg return % (all) |
             Avg return % (ex-COVID) | Avg spell (all) | Avg spell (ex-COVID)
    """
    all_rows = regime_stats_table(cyc_result, target, covid,
                                  exclude_covid=False)
    if covid is None:
        # No window — comparison collapses to the all-history view.
        return pd.DataFrame([{
            "Regime": r.label,
            "Months": r.n_months,
            "Avg monthly return %": round(r.avg_monthly_return_pct, 2)
            if r.avg_monthly_return_pct == r.avg_monthly_return_pct else None,
            "Avg spell (months)": round(r.avg_spell_duration, 1),
        } for r in all_rows])

    ex_rows = regime_stats_table(cyc_result, target, covid,
                                 exclude_covid=True)
    ex_by_id = {r.regime_id: r for r in ex_rows}

    out = []
    for r in all_rows:
        ex = ex_by_id.get(r.regime_id)
        out.append({
            "Regime": r.label,
            "Months (all)": r.n_months,
            "Months (ex-COVID)": ex.n_months if ex else 0,
            "Avg return % (all)": round(r.avg_monthly_return_pct, 2)
            if r.avg_monthly_return_pct == r.avg_monthly_return_pct else None,
            "Avg return % (ex-COVID)": round(ex.avg_monthly_return_pct, 2)
            if ex and ex.avg_monthly_return_pct == ex.avg_monthly_return_pct
            else None,
            "Avg spell (all)": round(r.avg_spell_duration, 1),
            "Avg spell (ex-COVID)": round(ex.avg_spell_duration, 1)
            if ex else 0.0,
        })
    return pd.DataFrame(out)


# ---------- Turning-point filtering for recurrence ----------

def filter_turning_points(turns: pd.Series,
                          covid: Optional[CovidWindow],
                          exclude_covid: bool) -> pd.Series:
    """
    Drop turning points (peaks or troughs) whose date falls inside the COVID
    window. Used to recompute recurrence base rates on normal-times turns.

    `turns`         — a Series indexed by turning-point date (engine output).
    Returns the filtered Series; unchanged if exclude_covid is False or there
    is no window.
    """
    if not exclude_covid or covid is None or turns is None or len(turns) == 0:
        return turns
    idx = pd.DatetimeIndex(turns.index)
    keep = ~((idx >= covid.start) & (idx <= covid.end))
    return turns[keep]


def covid_overlap_note(turns: pd.Series,
                       covid: Optional[CovidWindow]) -> str:
    """
    Human-readable note on how many turning points sit inside the COVID
    window — so the dashboard can tell the analyst what the toggle removes.
    """
    if covid is None or turns is None or len(turns) == 0:
        return ""
    idx = pd.DatetimeIndex(turns.index)
    inside = int(((idx >= covid.start) & (idx <= covid.end)).sum())
    if inside == 0:
        return (f"No detected turning points fall inside the COVID window "
                f"({covid.label}) — the ex-COVID base rate is unchanged here.")
    return (f"{inside} of {len(turns)} detected turning points fall inside the "
            f"COVID window ({covid.label}); excluding them recomputes the base "
            f"rate on normal-times turns only.")
