"""
Event Deep-Dive — Phase 4.

Takes the sharp turning points the market has actually been through and opens
each one up. For a chosen event date it answers four questions:

    1. WHAT MOVED   — how the HRC price and every driver behaved in the months
                      around the event (pre vs post window, % change).
    2. WHY          — a driver decomposition of the move: which drivers'
                      betas were doing the work in the window, using the same
                      rolling-OLS attribution engine as the Attribution tab.
    3. HOW OFTEN    — recurrence: of all the historical peaks/troughs the
                      cyclicity engine detected, how did the price typically
                      resolve in the N months after a turning point of the
                      same kind? This is the empirical base rate.
    4. WHERE NOW    — a live flag: is the market today sitting in the same
                      behavioural regime the event sat in, and is price within
                      a comparable drawdown / momentum band? If yes, the event
                      is "rhyming" with the present.

This module deliberately does NOT touch the macro calendar — that is a
separate, forward-looking surface. The Event Deep-Dive is backward-looking:
it studies turning points that have already happened so the analyst can
recognise them when they recur.

The module is self-contained and degrades gracefully:
  - no drivers (US / overview_only)  -> decomposition panel is empty, the
    what-moved and recurrence panels still work on price alone.
  - too little history               -> recurrence base rate is flagged
    'insufficient sample' rather than reported as a fragile number.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from .events import analyse_event
from .attribution import rolling_attribution


# ---------- Data containers ----------

@dataclass
class EventCandidate:
    """One selectable event in the picker."""
    key: str                       # stable id, e.g. 'cfg_0' or 'peak_2021-05-01'
    label: str                     # human label shown in the picker
    date: pd.Timestamp
    kind: str                      # 'peak' | 'trough' | 'shock' | 'policy' | 'event'
    source: str                    # 'config' | 'detected'


@dataclass
class DecompContribution:
    """One driver's contribution to the move around an event."""
    driver: str
    beta_in_window: float          # avg rolling beta over the event window
    driver_change_pct: float       # how much the driver itself moved (pre->post)
    contribution_pct: float        # share of the explained move, normalised to 100


@dataclass
class RecurrenceStats:
    """Empirical base rate: how turning points of this kind usually resolve."""
    kind: str                      # 'peak' or 'trough'
    n_events: int                  # how many comparable turning points were found
    horizons: List[int]            # months-ahead measured, e.g. [3, 6, 12]
    avg_move_pct: Dict[int, float] # horizon -> average % move after the turn
    hit_rate: Dict[int, float]     # horizon -> share that moved the "expected" way
    sufficient: bool               # False => sample too thin to trust
    note: str = ""


@dataclass
class CrossRegionMove:
    """How one region's HRC behaved around the event date — the 'where' answer."""
    region: str
    currency: str
    available: bool
    pre_avg: float = float("nan")
    post_avg: float = float("nan")
    pct_change: float = float("nan")
    regime_label: str = ""         # behavioural regime that region was in
    note: str = ""


@dataclass
class DriverStory:
    """Driver-level causal context — why a DRIVER itself moved, and where."""
    driver: str
    headline: str = ""
    why: str = ""
    country: str = ""
    sources: List[dict] = field(default_factory=list)   # [{title, url}, ...]


@dataclass
class CuratedContext:
    """The qualitative 'why' for an event, loaded from event_context.yaml."""
    matched: bool
    title: str = ""
    confidence: str = ""
    as_of: str = ""
    what_changed: List[str] = field(default_factory=list)
    why: str = ""
    country_breakdown: Dict[str, str] = field(default_factory=dict)
    watch_next: List[str] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)   # [{title, url}, ...]
    # Driver-level causal tracing, matched per top driver of the event
    driver_stories: List[DriverStory] = field(default_factory=list)


@dataclass
class EventDeepDiveResult:
    """Everything the Event Deep-Dive tab needs for ONE chosen event."""
    region: str
    currency: str
    event: EventCandidate

    # Panel 1 — what moved
    window_table: pd.DataFrame     # variable | pre_avg | post_avg | pct_change | window
    headline_move_pct: float       # HRC pre->post % change at the primary window

    # Panel 2 — decomposition
    contributions: List[DecompContribution]
    decomposition_available: bool
    decomposition_note: str

    # Panel 3 — recurrence
    recurrence: Optional[RecurrenceStats]

    # Panel 4 — live flag
    event_regime: Optional[int]
    event_regime_label: Optional[str]
    current_regime: int
    current_regime_label: str
    regime_match: bool             # is the market in the same regime now?
    rhyme_score: float             # 0..1 — how much "today" rhymes with the event
    rhyme_reasons: List[str]       # plain-English bullets explaining the score

    # Curated qualitative context (the narrative 'why')
    curated: CuratedContext = field(default_factory=lambda: CuratedContext(False))
    # Cross-region 'where did it change' panel
    cross_region: List[CrossRegionMove] = field(default_factory=list)

    primary_window_m: int = 6


# ---------- Event candidate assembly ----------

def build_event_candidates(cyc_result,
                            episodes_cfg: Optional[List[dict]] = None,
                            target: Optional[pd.Series] = None) -> List[EventCandidate]:
    """
    Merge two event sources into one de-duplicated, date-sorted picker list:

      • config episodes  — the named, narratively-important events from
        config.yaml (supercycle peak, Ukraine war, ...). These carry good
        labels and an explicit 'type'.
      • detected turns   — every peak and trough the cyclicity engine found.
        These give complete coverage of the actual turning points.

    When a config episode lands within 2 months of a detected turn we keep
    the config one (its label is better) and drop the detected duplicate.
    """
    candidates: List[EventCandidate] = []

    # 1. Config episodes
    cfg_dates = []
    if episodes_cfg:
        for i, ep in enumerate(episodes_cfg):
            d = pd.to_datetime(ep["date"])
            cfg_dates.append(d)
            candidates.append(EventCandidate(
                key=f"cfg_{i}",
                label=f"{ep['name']}",
                date=d,
                kind=ep.get("type", "event"),
                source="config",
            ))

    # 2. Detected peaks / troughs from the engine
    def _near_config(d: pd.Timestamp) -> bool:
        return any(abs((d - c).days) <= 62 for c in cfg_dates)

    if cyc_result is not None:
        for d, v in getattr(cyc_result, "peaks", pd.Series(dtype=float)).items():
            if _near_config(d):
                continue
            candidates.append(EventCandidate(
                key=f"peak_{d.date()}",
                label=f"Detected peak — {d.strftime('%b %Y')}",
                date=d, kind="peak", source="detected",
            ))
        for d, v in getattr(cyc_result, "troughs", pd.Series(dtype=float)).items():
            if _near_config(d):
                continue
            candidates.append(EventCandidate(
                key=f"trough_{d.date()}",
                label=f"Detected trough — {d.strftime('%b %Y')}",
                date=d, kind="trough", source="detected",
            ))

    # Keep only events that actually sit inside the data range
    if target is not None and len(target) > 0:
        lo, hi = target.index.min(), target.index.max()
        candidates = [c for c in candidates if lo <= c.date <= hi]

    candidates.sort(key=lambda c: c.date)
    return candidates


# ---------- Panel 2: driver decomposition ----------

def _decompose_event(target: pd.Series, drivers: pd.DataFrame,
                      event_date: pd.Timestamp, window_m: int,
                      attr_window: int = 24) -> tuple:
    """
    Decompose the move around an event into driver contributions.

    Method (kept deliberately consistent with attribution.py's
    `_attribution_weights`):
      • Take the rolling-OLS betas from rolling_attribution.
      • Average each driver's beta over the rolling windows whose end-date
        falls inside [event - window, event + window] — i.e. the betas that
        were "in force" while the event played out.
      • Form a SCALE-AWARE contribution: |avg beta_i| × σ(driver_i), where
        σ is the driver's standard deviation measured inside the event
        window. This is the same |β·σ| decomposition the Attribution tab
        uses, and it is essential: a driver quoted in small units (e.g. the
        10-Year Treasury yield, ~2-5) gets a mechanically huge OLS beta to
        fit a price in the hundreds. Weighting by |β·driver_change%| would
        let that scale artefact dominate the panel (it did, in testing —
        the yield showed a fake 97% share). |β·σ| cancels the unit scale
        and reports genuine explanatory weight.

    `driver_change_pct` is still reported per driver as descriptive context
    (how far the driver itself travelled), but it does NOT drive the
    contribution share.

    Returns (list[DecompContribution], available: bool, note: str).
    """
    if drivers is None or drivers.shape[1] == 0:
        return [], False, ("No driver data for this region — the decomposition "
                            "panel needs drivers. Price-path panels still apply.")

    attr = rolling_attribution(target, drivers, window=attr_window)
    if len(attr.rolling_betas) == 0:
        return [], False, ("Not enough history for a rolling decomposition around "
                            "this event.")

    win_lo = event_date - pd.DateOffset(months=window_m)
    win_hi = event_date + pd.DateOffset(months=window_m)
    betas_in = attr.rolling_betas.loc[
        (attr.rolling_betas.index >= win_lo) &
        (attr.rolling_betas.index <= win_hi)
    ]
    note = ""
    if len(betas_in) == 0:
        # Event is older than the first rolling window — fall back to the
        # nearest available window and say so.
        nearest = attr.rolling_betas.index[
            (attr.rolling_betas.index - event_date).map(abs).argmin()
        ]
        betas_in = attr.rolling_betas.loc[[nearest]]
        note = (f"The rolling-attribution window does not reach this event; "
                f"showing the closest available window ({nearest.strftime('%b %Y')}). "
                f"Read the decomposition as indicative, not exact.")

    avg_beta = betas_in.mean()

    # Driver behaviour inside the full event window: standard deviation
    # (drives the scale-aware contribution) and pre->post % change
    # (descriptive only).
    pre = drivers.loc[win_lo:event_date - pd.DateOffset(days=1)]
    post = drivers.loc[event_date:win_hi]
    win = drivers.loc[win_lo:win_hi]
    contribs = []
    raw = {}
    for col in drivers.columns:
        pv, qv = pre[col].dropna(), post[col].dropna()
        if len(pv) == 0 or len(qv) == 0:
            d_chg = float("nan")
        else:
            pa, qa = float(pv.mean()), float(qv.mean())
            d_chg = ((qa - pa) / pa * 100) if pa != 0 else float("nan")
        b = float(avg_beta.get(col, float("nan")))
        sigma = float(win[col].dropna().std())
        # scale-aware contribution: |beta * sigma| — see docstring
        raw[col] = (abs(b) * sigma
                    if np.isfinite(b) and np.isfinite(sigma) else 0.0)
        contribs.append((col, b, d_chg))

    total = sum(raw.values())
    out = []
    for col, b, d_chg in contribs:
        share = (raw[col] / total * 100) if total > 0 else float("nan")
        out.append(DecompContribution(
            driver=col,
            beta_in_window=b,
            driver_change_pct=d_chg,
            contribution_pct=share,
        ))
    out.sort(key=lambda c: (c.contribution_pct if np.isfinite(c.contribution_pct)
                            else -1), reverse=True)
    return out, True, note


# ---------- Panel 3: recurrence base rate ----------

def _forward_move(target: pd.Series, at: pd.Timestamp, h: int) -> Optional[float]:
    """% change in target from `at` to h months later. None if out of range."""
    if at not in target.index:
        # snap to nearest available month
        pos = target.index.get_indexer([at], method="nearest")[0]
        at = target.index[pos]
    start_pos = target.index.get_loc(at)
    end_pos = start_pos + h
    if end_pos >= len(target):
        return None
    v0 = target.iloc[start_pos]
    v1 = target.iloc[end_pos]
    if v0 == 0 or not np.isfinite(v0):
        return None
    return float((v1 - v0) / v0 * 100)


def _recurrence(target: pd.Series, cyc_result, kind: str,
                horizons: List[int] = [3, 6, 12]) -> RecurrenceStats:
    """
    Empirical base rate for turning points of `kind` ('peak' or 'trough').

    For every detected turn of that kind we measure the forward % move at
    each horizon. We then report:
      • avg_move_pct  — the mean forward move
      • hit_rate      — share that resolved the "expected" way (price FALLS
                        after a peak, RISES after a trough)
    A turn needs at least 4 comparable historical instances to be considered
    a trustworthy base rate; below that we still show the numbers but flag
    them 'insufficient'.
    """
    if cyc_result is None:
        return RecurrenceStats(kind, 0, horizons, {}, {}, False,
                               "No cyclicity result available.")
    series = (cyc_result.peaks if kind == "peak"
              else cyc_result.troughs if kind == "trough" else None)
    if series is None or len(series) == 0:
        return RecurrenceStats(kind, 0, horizons, {}, {}, False,
                               f"No detected {kind}s to build a base rate from.")

    avg_move, hit = {}, {}
    n_used = 0
    for h in horizons:
        moves = []
        for d in series.index:
            m = _forward_move(target, d, h)
            if m is not None:
                moves.append(m)
        if not moves:
            avg_move[h] = float("nan")
            hit[h] = float("nan")
            continue
        n_used = max(n_used, len(moves))
        avg_move[h] = float(np.mean(moves))
        # 'expected' direction: down after a peak, up after a trough
        if kind == "peak":
            hit[h] = float(np.mean([1.0 if m < 0 else 0.0 for m in moves]))
        else:
            hit[h] = float(np.mean([1.0 if m > 0 else 0.0 for m in moves]))

    sufficient = n_used >= 4
    note = ("" if sufficient else
            f"Only {n_used} comparable {kind}s in the sample — treat the base "
            f"rate as indicative, not statistically firm.")
    return RecurrenceStats(kind, n_used, horizons, avg_move, hit,
                           sufficient, note)


# ---------- Panel 4: live "does today rhyme" flag ----------

def _regime_at(cyc_result, date: pd.Timestamp) -> Optional[int]:
    """Regime id active at (or nearest to) a date."""
    if cyc_result is None or len(cyc_result.regime_labels) == 0:
        return None
    lbl = cyc_result.regime_labels
    if date in lbl.index:
        return int(lbl.loc[date])
    pos = lbl.index.get_indexer([date], method="nearest")[0]
    return int(lbl.iloc[pos])


def _regime_label(cyc_result, rid: Optional[int]) -> Optional[str]:
    if rid is None or cyc_result is None:
        return None
    for p in cyc_result.regime_profiles:
        if p.regime_id == rid:
            return p.label
    return f"R{rid}"


def _rhyme(target: pd.Series, cyc_result, event_date: pd.Timestamp,
           event_regime: Optional[int], current_regime: int) -> tuple:
    """
    Score 0..1 of how much the present 'rhymes' with the event setup.

    Three equally-weighted checks:
      1. Same behavioural regime now as at the event.
      2. Drawdown-from-peak band within 10pp of the event's.
      3. 6-month momentum sign matches the event's.

    Returns (score, reasons list, regime_match bool).
    """
    reasons = []
    score = 0.0

    regime_match = (event_regime is not None and event_regime == current_regime)
    if regime_match:
        score += 1 / 3
        reasons.append("Market is in the **same behavioural regime** as it was "
                       "at this event.")
    else:
        reasons.append("Market is in a **different regime** than at this event.")

    # drawdown band
    running_peak = target.cummax()
    dd = (target - running_peak) / running_peak * 100
    try:
        dd_event = float(dd.reindex([event_date], method="nearest").iloc[0])
        dd_now = float(dd.iloc[-1])
        if abs(dd_event - dd_now) <= 10:
            score += 1 / 3
            reasons.append(f"Drawdown-from-peak is comparable "
                           f"(event {dd_event:.0f}%, now {dd_now:.0f}%).")
        else:
            reasons.append(f"Drawdown-from-peak differs "
                           f"(event {dd_event:.0f}%, now {dd_now:.0f}%).")
    except Exception:
        pass

    # 6m momentum sign
    mom = target.pct_change(6) * 100
    try:
        mom_event = float(mom.reindex([event_date], method="nearest").iloc[0])
        mom_now = float(mom.iloc[-1])
        if np.sign(mom_event) == np.sign(mom_now):
            score += 1 / 3
            reasons.append(f"6-month momentum has the **same sign** "
                           f"(event {mom_event:+.0f}%, now {mom_now:+.0f}%).")
        else:
            reasons.append(f"6-month momentum sign differs "
                           f"(event {mom_event:+.0f}%, now {mom_now:+.0f}%).")
    except Exception:
        pass

    return float(min(score, 1.0)), reasons, regime_match


# ---------- Curated qualitative context ----------

def load_event_context(path: str = "data/event_context.yaml") -> dict:
    """
    Load the curated context YAML. Tries a few paths so it works both locally
    and on Streamlit Cloud (different working directories).

    Returns a dict {'events': [...], 'driver_context': [...]}. Always returns
    that shape even if the file is missing, so callers never need to guard.
    """
    import yaml
    from pathlib import Path
    candidates = [
        Path(path),
        Path(__file__).parent.parent / "data" / "event_context.yaml",
        Path.cwd() / "data" / "event_context.yaml",
    ]
    for p in candidates:
        try:
            if p.exists():
                with open(p) as f:
                    doc = yaml.safe_load(f) or {}
                return {
                    "events": doc.get("events", []) or [],
                    "driver_context": doc.get("driver_context", []) or [],
                }
        except Exception:
            continue
    return {"events": [], "driver_context": []}


def _norm_sources(raw) -> List[dict]:
    """Normalise a `sources` field to a list of {title, url} dicts.

    Accepts either the structured list form or a legacy plain string (which
    becomes a single title-only entry). This keeps old YAML files working.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        return [{"title": raw.strip(), "url": ""}]
    out = []
    for s in raw:
        if isinstance(s, dict):
            out.append({"title": str(s.get("title", "")).strip(),
                        "url": str(s.get("url", "")).strip()})
        elif isinstance(s, str):
            out.append({"title": s.strip(), "url": ""})
    return out


def _match_driver_stories(event_date: pd.Timestamp,
                          top_drivers: List[str],
                          driver_ctx: List[dict]) -> List[DriverStory]:
    """
    For the event's top contributing drivers, find any driver_context entry
    whose `driver` matches (case-insensitive, loose) and whose [from, to]
    window brackets the event date. Returns one DriverStory per match, in the
    same order as `top_drivers` so the most important drivers surface first.
    """
    stories = []
    for drv in top_drivers:
        drv_l = drv.lower().strip()
        for dc in driver_ctx:
            name = str(dc.get("driver", "")).lower().strip()
            if not name:
                continue
            # loose match: exact, or one contained in the other
            if not (name == drv_l or name in drv_l or drv_l in name):
                continue
            try:
                d_from = pd.to_datetime(dc.get("from"))
                d_to = pd.to_datetime(dc.get("to"))
            except Exception:
                continue
            if d_from <= event_date <= d_to:
                stories.append(DriverStory(
                    driver=drv,
                    headline=str(dc.get("headline", "")).strip(),
                    why=str(dc.get("why", "")).strip(),
                    country=str(dc.get("country", "")).strip(),
                    sources=_norm_sources(dc.get("sources")),
                ))
                break   # one story per driver
    return stories


def match_curated_context(event: EventCandidate,
                          context: Optional[dict] = None,
                          top_drivers: Optional[List[str]] = None,
                          tolerance_days: int = 62) -> CuratedContext:
    """
    Match a chosen event to a curated-context entry BY DATE (within
    `tolerance_days`), and attach driver-level stories for `top_drivers`.

    `context` is the dict returned by load_event_context (with 'events' and
    'driver_context' keys). If None it is loaded. If no event entry is within
    tolerance, returns an unmatched CuratedContext — but driver stories are
    still attached if any match, since a driver story can be useful even when
    the event itself is not written up.
    """
    if context is None:
        context = load_event_context()
    events = context.get("events", []) or []
    driver_ctx = context.get("driver_context", []) or []

    best = None
    best_gap = None
    for ce in events:
        try:
            cd = pd.to_datetime(ce.get("match_date"))
        except Exception:
            continue
        gap = abs((event.date - cd).days)
        if gap <= tolerance_days and (best_gap is None or gap < best_gap):
            best, best_gap = ce, gap

    driver_stories = _match_driver_stories(
        event.date, top_drivers or [], driver_ctx)

    if best is None:
        # No event-level write-up, but driver stories may still exist.
        return CuratedContext(matched=False, driver_stories=driver_stories)

    return CuratedContext(
        matched=True,
        title=best.get("title", ""),
        confidence=str(best.get("confidence", "")),
        as_of=str(best.get("as_of", "")),
        what_changed=list(best.get("what_changed", []) or []),
        why=str(best.get("why", "")).strip(),
        country_breakdown=dict(best.get("country_breakdown", {}) or {}),
        watch_next=list(best.get("watch_next", []) or []),
        sources=_norm_sources(best.get("sources")),
        driver_stories=driver_stories,
    )


# ---------- Cross-region 'where did it change' ----------

def cross_region_event_moves(regions: Dict[str, dict],
                             event_date: pd.Timestamp,
                             window_m: int = 6) -> List[CrossRegionMove]:
    """
    For each available region, measure how its HRC price moved around the
    same calendar event date. This is the 'where did it change' answer —
    the same shock lands differently on China, India and US.

    `regions` is a dict: region_name -> {'y': price Series, 'currency': str,
    'cyc': CyclicityResult or None}. Regions with no usable data around the
    event are returned flagged unavailable rather than dropped, so the panel
    always shows the full country line-up.
    """
    out: List[CrossRegionMove] = []
    for name, info in regions.items():
        y = info.get("y")
        currency = info.get("currency", "")
        cyc = info.get("cyc")
        if y is None or len(y.dropna()) < 4:
            out.append(CrossRegionMove(name, currency, False,
                                       note="No price data."))
            continue
        y = y.dropna().sort_index()
        lo = event_date - pd.DateOffset(months=window_m)
        hi = event_date + pd.DateOffset(months=window_m)
        pre = y.loc[lo:event_date - pd.DateOffset(days=1)]
        post = y.loc[event_date:hi]
        if len(pre) == 0 or len(post) == 0:
            out.append(CrossRegionMove(
                name, currency, False,
                note="Event date is outside this region's data range."))
            continue
        pa, qa = float(pre.mean()), float(post.mean())
        pct = ((qa - pa) / pa * 100) if pa != 0 else float("nan")
        regime_label = ""
        if cyc is not None:
            rid = _regime_at(cyc, event_date)
            regime_label = _regime_label(cyc, rid) or ""
        out.append(CrossRegionMove(
            region=name, currency=currency, available=True,
            pre_avg=pa, post_avg=qa, pct_change=pct,
            regime_label=regime_label,
        ))
    return out


# ---------- Top-level entry point ----------

def analyse_event_deep_dive(target: pd.Series,
                            drivers: Optional[pd.DataFrame],
                            cyc_result,
                            event: EventCandidate,
                            region: str = "",
                            currency: str = "USD",
                            windows_months: List[int] = [3, 6],
                            attr_window: int = 24,
                            all_regions: Optional[Dict[str, dict]] = None,
                            context: Optional[dict] = None
                            ) -> EventDeepDiveResult:
    """
    Run the full Event Deep-Dive for ONE chosen event.

    `target`        — HRC price series for the region.
    `drivers`       — driver DataFrame, or None for overview_only regions.
    `cyc_result`    — the CyclicityResult from analyse_cyclicity (shared engine).
    `event`         — the EventCandidate chosen in the picker.
    `all_regions`   — optional dict for the cross-region panel:
                      region_name -> {'y': Series, 'currency': str,
                      'cyc': CyclicityResult|None}. If None, the cross-region
                      panel is empty (the rest still works).
    `context`       — optional pre-loaded curated-context dict (with 'events'
                      and 'driver_context' keys); if None it is loaded from
                      data/event_context.yaml.
    """
    target = target.dropna().sort_index()
    primary_w = windows_months[-1] if windows_months else 6

    # ----- Panel 1: what moved -----
    X_for_window = (drivers if drivers is not None and drivers.shape[1] > 0
                    else pd.DataFrame(index=target.index))
    raw = analyse_event(target, X_for_window, event.date,
                        window_months=windows_months)
    rows = []
    headline = float("nan")
    for w_name, var_stats in raw["windows"].items():
        for var, st in var_stats.items():
            rows.append({
                "Variable": "HRC price" if var == "target" else var,
                "Window": f"±{w_name}",
                "Pre-avg": st["pre_avg"],
                "Post-avg": st["post_avg"],
                "% Change": st["pct_change"],
            })
            if var == "target" and w_name == f"{primary_w}m":
                headline = st["pct_change"]
    window_table = pd.DataFrame(rows)
    if np.isnan(headline) and len(window_table) > 0:
        # fall back to any HRC row if the primary window had no data
        hrc_rows = window_table[window_table["Variable"] == "HRC price"]
        if len(hrc_rows) > 0:
            headline = float(hrc_rows.iloc[0]["% Change"])

    # ----- Panel 2: decomposition -----
    contributions, decomp_ok, decomp_note = _decompose_event(
        target, drivers, event.date, primary_w, attr_window=attr_window)

    # ----- Panel 3: recurrence -----
    # Recurrence is only meaningful for genuine turning points. For shock /
    # policy episodes we still classify them by their headline move so the
    # analyst gets the closest applicable base rate.
    if event.kind in ("peak", "trough"):
        rec_kind = event.kind
    else:
        rec_kind = "peak" if (np.isfinite(headline) and headline < 0) else "trough"
    recurrence = _recurrence(target, cyc_result, rec_kind)

    # ----- Panel 4: live flag -----
    event_regime = _regime_at(cyc_result, event.date)
    current_regime = (cyc_result.current_regime if cyc_result is not None else -1)
    event_regime_label = _regime_label(cyc_result, event_regime)
    current_regime_label = _regime_label(cyc_result, current_regime) or "—"
    rhyme_score, rhyme_reasons, regime_match = _rhyme(
        target, cyc_result, event.date, event_regime, current_regime)

    # ----- Curated qualitative context (the narrative 'why') -----
    # Driver-level stories are matched against the event's top contributing
    # drivers, so the causal tracing surfaces the drivers that actually
    # mattered for THIS move (not every driver in the region).
    _ranked_drivers = [c.driver for c in contributions
                       if np.isfinite(c.contribution_pct)]
    curated = match_curated_context(event, context,
                                    top_drivers=_ranked_drivers)

    # ----- Cross-region 'where did it change' -----
    cross = []
    if all_regions:
        cross = cross_region_event_moves(all_regions, event.date,
                                         window_m=primary_w)

    return EventDeepDiveResult(
        region=region,
        currency=currency,
        event=event,
        window_table=window_table,
        headline_move_pct=headline,
        contributions=contributions,
        decomposition_available=decomp_ok,
        decomposition_note=decomp_note,
        recurrence=recurrence,
        event_regime=event_regime,
        event_regime_label=event_regime_label,
        current_regime=current_regime,
        current_regime_label=current_regime_label,
        regime_match=regime_match,
        rhyme_score=rhyme_score,
        rhyme_reasons=rhyme_reasons,
        curated=curated,
        cross_region=cross,
        primary_window_m=primary_w,
    )
