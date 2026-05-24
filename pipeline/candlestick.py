"""
Candlestick view + optional Elliott-wave overlay — Phase 5.

IMPORTANT DATA HONESTY NOTE
---------------------------
The HRC series in this pipeline is MONTHLY and CLOSE-ONLY. There is no
intraday, no weekly, and no genuine open/high/low. A real candlestick needs
all four OHLC values. We therefore build *synthetic monthly candles*:

    open   = previous month's close
    close  = this month's close
    high   = max(open, close)
    low    = min(open, close)

This means every candle is a pure body with NO WICKS — it shows the
direction and size of each month's close-to-close move, nothing more. It is
an honest re-rendering of the same close series the rest of the dashboard
uses, NOT a claim about intra-month trading ranges. The dashboard labels it
as such.

UPGRADE PATH: if a future data refresh adds real OHLC columns to the Excel
sheet, replace `build_monthly_candles` with a direct read of those columns
and the wicks become real. Nothing else in this module needs to change.

ELLIOTT WAVE OVERLAY
--------------------
Elliott wave labelling is, by construction, subjective pattern-fitting — it
is NOT a statistical model and carries no significance test. We include it
ONLY as an optional, clearly-labelled visual aid that mechanically annotates
the most recent detected swing sequence with the conventional 1-2-3-4-5 / A-B-C
counts. It must never be presented as a forecast. The dashboard gates it
behind an off-by-default toggle with an explicit disclaimer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


# ---------- Synthetic monthly candles ----------

@dataclass
class CandleResult:
    region: str
    currency: str
    candles: pd.DataFrame          # index=date, cols: open/high/low/close/up
    is_synthetic: bool             # True => open=prior close, no real wicks
    note: str


def build_monthly_candles(target: pd.Series, region: str = "",
                          currency: str = "USD") -> CandleResult:
    """
    Build synthetic monthly candles from a close-only price series.

    See the module docstring for why these candles have no wicks. The
    returned DataFrame has one row per month with open/high/low/close and a
    boolean `up` (close >= open).
    """
    s = target.dropna().sort_index()
    if len(s) < 2:
        return CandleResult(region, currency,
                            pd.DataFrame(columns=["open", "high", "low",
                                                  "close", "up"]),
                            True, "Not enough data to build candles.")

    close = s
    open_ = s.shift(1)
    # First month has no prior close — anchor open to its own close so the
    # first candle is a flat doji rather than dropping the month.
    open_.iloc[0] = close.iloc[0]

    df = pd.DataFrame({
        "open": open_,
        "close": close,
    })
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["up"] = df["close"] >= df["open"]

    note = ("Synthetic monthly candles: open = previous month's close, so each "
            "candle is a pure body with no wicks. This is an honest re-rendering "
            "of the monthly close series — not intra-month trading-range data, "
            "which this pipeline does not have.")
    return CandleResult(region, currency, df, True, note)


# ---------- Swing detection (shared basis for Elliott labelling) ----------

def detect_swings(target: pd.Series,
                  prominence_pct: float = 5.0,
                  min_distance_months: int = 4) -> pd.DataFrame:
    """
    Detect alternating swing highs and lows.

    Uses the SAME prominence/distance peak-detection approach as
    cyclicity.py's `_find_peaks_troughs`, then merges peaks and troughs into
    one strictly-alternating sequence (a clean zig-zag), which is the input
    an Elliott count needs.

    Returns a DataFrame: index=date, cols [price, kind] with kind in
    {'H','L'}, guaranteed to alternate.
    """
    from scipy.signal import find_peaks
    s = target.dropna().sort_index()
    y = s.values
    idx = s.index
    if len(y) < 6:
        return pd.DataFrame(columns=["price", "kind"])

    prom = (prominence_pct / 100.0) * float(np.std(y))
    peak_i, _ = find_peaks(y, prominence=prom, distance=min_distance_months)
    trough_i, _ = find_peaks(-y, prominence=prom, distance=min_distance_months)

    pts = []
    for i in peak_i:
        pts.append((idx[i], y[i], "H"))
    for i in trough_i:
        pts.append((idx[i], y[i], "L"))
    pts.sort(key=lambda t: t[0])

    # Enforce strict alternation: if two same-kind swings are adjacent, keep
    # the more extreme one (higher H / lower L) and drop the other.
    cleaned = []
    for d, p, k in pts:
        if cleaned and cleaned[-1][2] == k:
            pd_, pp_, pk_ = cleaned[-1]
            if (k == "H" and p >= pp_) or (k == "L" and p <= pp_):
                cleaned[-1] = (d, p, k)
            # else: keep existing, drop current
        else:
            cleaned.append((d, p, k))

    return pd.DataFrame(
        [{"price": p, "kind": k} for _, p, k in cleaned],
        index=pd.DatetimeIndex([d for d, _, _ in cleaned], name="date"))


# ---------- Elliott wave overlay (optional, clearly-labelled) ----------

@dataclass
class ElliottOverlay:
    """A mechanically-labelled Elliott count over the most recent swings."""
    available: bool
    labels: pd.DataFrame = field(default_factory=pd.DataFrame)  # date|price|wave
    direction: str = ""            # 'impulse up' / 'impulse down' / 'unclear'
    disclaimer: str = ""
    note: str = ""


# Conventional Elliott label sequences
_IMPULSE = ["1", "2", "3", "4", "5"]
_CORRECTIVE = ["A", "B", "C"]


def label_elliott(swings: pd.DataFrame) -> ElliottOverlay:
    """
    Mechanically annotate the most recent swing sequence with a conventional
    Elliott count (1-2-3-4-5 then A-B-C).

    This is NOT a model and NOT a forecast. It simply takes the last up-to-8
    alternating swings and tags them in order. If a 5-wave impulse fits the
    swing directions it is labelled as such; the following 3 swings (if
    present) get the A-B-C corrective tags.

    The output always carries an explicit disclaimer string.
    """
    disclaimer = (
        "Elliott wave labelling is subjective pattern-fitting, not a "
        "statistical model. It carries no significance test and is shown only "
        "as an optional visual aid — never as a forecast. The same price path "
        "can be counted several valid ways.")

    if swings is None or len(swings) < 4:
        return ElliottOverlay(
            available=False,
            disclaimer=disclaimer,
            note="Need at least 4 detected swings to attempt an Elliott count.")

    # Take the last 8 swings (5 impulse + 3 corrective at most)
    recent = swings.tail(8).copy()

    # Determine the impulse direction from the first two swings of the window:
    # if it starts at a low and rises, it's an up-impulse; vice versa.
    first_kind = recent["kind"].iloc[0]
    direction = "impulse up" if first_kind == "L" else "impulse down"

    wave_tags = []
    seq = _IMPULSE + _CORRECTIVE          # 1..5, A, B, C
    for i in range(len(recent)):
        wave_tags.append(seq[i] if i < len(seq) else "")

    labels = recent.copy()
    labels["wave"] = wave_tags
    labels = labels.reset_index()

    note = (f"Mechanical count over the last {len(recent)} detected swings, "
            f"read as an {direction}. Waves 1-5 are the impulse leg; A-B-C "
            f"(if present) the correction.")
    return ElliottOverlay(
        available=True,
        labels=labels,
        direction=direction,
        disclaimer=disclaimer,
        note=note,
    )
