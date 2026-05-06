"""
Spread analysis — implements the Tata Steel spread metric.

Formula (per BPM pre-read deck):
    Spread = HRC − (iron_ore_weight × iron_ore + hcc_weight × hcc)

For China: HRC China FOB − (1.6 × Iron Ore China CFR + 0.9 × HCC Aus FOB), USD/t
For India: HRC FBD − (1.6 × Iron Ore Odisha + 0.9 × Import HCC India CFR), INR/t

Outputs:
    • Monthly spread series
    • FY (Apr-Mar) average table
    • Current vs historical percentile snapshot
    • Spread regimes (compressed / normal / expanded based on percentiles)
    • Decomposition of HRC into raw material costs vs margin
    • Cross-region comparison (when both regions available)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class SpreadResult:
    """Everything the spread module produces for one region."""
    region: str
    currency: str
    spread_series: pd.Series                 # monthly spread, indexed by date
    rm_cost_series: pd.Series                # raw material cost component
    decomposition: pd.DataFrame              # HRC, IronOre×w, HCC×w, Spread cols
    fy_table: pd.DataFrame                   # FY-average table
    percentiles: Dict[str, float]            # {p10, p25, p50, p75, p90}
    current_snapshot: Dict                   # latest values + percentile rank
    regime_series: pd.Series                 # 'compressed' / 'normal' / 'expanded'
    regime_thresholds: Dict[str, float]
    spread_config: Dict = field(default_factory=dict)   # iron_ore + hcc weights & cols


def fy_label(date: pd.Timestamp) -> str:
    """Indian fiscal year: Apr→Mar. Apr 2023 = FY24."""
    if date.month >= 4:
        return f"FY{(date.year + 1) % 100:02d}"
    return f"FY{date.year % 100:02d}"


def compute_spread(target: pd.Series,
                   iron_ore: pd.Series, iron_ore_weight: float,
                   hcc: pd.Series, hcc_weight: float) -> pd.DataFrame:
    """Build the full decomposition dataframe."""
    df = pd.DataFrame({
        "HRC": target,
        "IronOre_contribution": iron_ore_weight * iron_ore,
        "HCC_contribution": hcc_weight * hcc,
    }).dropna()
    df["RM_Cost"] = df["IronOre_contribution"] + df["HCC_contribution"]
    df["Spread"] = df["HRC"] - df["RM_Cost"]
    df["Spread_pct_of_HRC"] = df["Spread"] / df["HRC"] * 100
    return df


def fy_average_table(spread_series: pd.Series) -> pd.DataFrame:
    """FY-average + count of months in each FY."""
    df = pd.DataFrame({
        "Spread": spread_series,
        "FY": [fy_label(d) for d in spread_series.index],
    })
    g = df.groupby("FY")["Spread"].agg(["mean", "median", "min", "max", "count"])
    g = g.round(2)
    g.columns = ["FY Average", "Median", "Min", "Max", "# Months"]
    g = g.reset_index()
    # Sort FYs chronologically
    g["_fy_num"] = g["FY"].str.replace("FY", "").astype(int)
    g = g.sort_values("_fy_num").drop(columns="_fy_num").reset_index(drop=True)
    return g


def regime_classification(spread: pd.Series,
                           low_pct: float = 25,
                           high_pct: float = 75) -> tuple:
    """Classify each month as compressed/normal/expanded using percentile bands."""
    low_thr = float(np.nanpercentile(spread, low_pct))
    high_thr = float(np.nanpercentile(spread, high_pct))
    regimes = pd.Series(index=spread.index, dtype=object)
    regimes[spread <= low_thr] = "compressed"
    regimes[(spread > low_thr) & (spread < high_thr)] = "normal"
    regimes[spread >= high_thr] = "expanded"
    thresholds = {f"p{int(low_pct)}": low_thr, f"p{int(high_pct)}": high_thr}
    return regimes, thresholds


def percentile_rank(value: float, series: pd.Series) -> float:
    """What percentile of the historical distribution is `value`?"""
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float((s < value).mean() * 100)


def analyse_region(region_data,
                   spread_cfg: Optional[dict] = None) -> Optional[SpreadResult]:
    """Run the full spread analysis for one region (one RegionData)."""
    cfg = spread_cfg or region_data.spread_config
    if not cfg:
        return None
    iron_col = cfg.get("iron_ore_column")
    hcc_col = cfg.get("hcc_column")
    if iron_col not in region_data.df.columns or hcc_col not in region_data.df.columns:
        return None

    decomp = compute_spread(
        target=region_data.y,
        iron_ore=region_data.df[iron_col],
        iron_ore_weight=cfg.get("iron_ore_weight", 1.6),
        hcc=region_data.df[hcc_col],
        hcc_weight=cfg.get("hcc_weight", 0.9),
    )

    spread = decomp["Spread"]
    rm_cost = decomp["RM_Cost"]

    fy_tbl = fy_average_table(spread)

    pcts = {
        "p10": float(np.nanpercentile(spread, 10)),
        "p25": float(np.nanpercentile(spread, 25)),
        "p50": float(np.nanpercentile(spread, 50)),
        "p75": float(np.nanpercentile(spread, 75)),
        "p90": float(np.nanpercentile(spread, 90)),
    }

    regimes, thresholds = regime_classification(spread)

    latest_date = decomp.index[-1]
    snapshot = {
        "as_of": str(latest_date.date()),
        "hrc": float(decomp["HRC"].iloc[-1]),
        "iron_ore_contribution": float(decomp["IronOre_contribution"].iloc[-1]),
        "hcc_contribution": float(decomp["HCC_contribution"].iloc[-1]),
        "rm_cost": float(decomp["RM_Cost"].iloc[-1]),
        "spread": float(decomp["Spread"].iloc[-1]),
        "spread_pct_of_hrc": float(decomp["Spread_pct_of_HRC"].iloc[-1]),
        "spread_percentile": percentile_rank(spread.iloc[-1], spread),
        "current_regime": str(regimes.iloc[-1]),
        "vs_p50": float(spread.iloc[-1] - pcts["p50"]),
        "vs_5y_avg": float(spread.iloc[-1] - spread.iloc[-60:].mean())
                     if len(spread) >= 60 else float("nan"),
    }

    return SpreadResult(
        region=region_data.name,
        currency=region_data.currency,
        spread_series=spread,
        rm_cost_series=rm_cost,
        decomposition=decomp,
        fy_table=fy_tbl,
        percentiles=pcts,
        current_snapshot=snapshot,
        regime_series=regimes,
        regime_thresholds=thresholds,
        spread_config=cfg,
    )


def cross_region_comparison(china_result: SpreadResult,
                             india_result: SpreadResult) -> dict:
    """
    Compare China spread (USD/t) and India spread (INR/t) on a normalised basis.

    We can't just add them — different currencies. We compare each on its own
    percentile within its own history, plus z-scores. This way we can see
    "China is squeezed (in bottom 15%) while India is healthy (in top 40%)".
    """
    # Align dates
    common = china_result.spread_series.index.intersection(
        india_result.spread_series.index)
    if len(common) < 12:
        return {"available": False,
                "reason": "Less than 12 months of overlapping data"}

    cs = china_result.spread_series.loc[common]
    ins = india_result.spread_series.loc[common]

    # Percentile-rank each month within its own region's history
    china_pct = cs.rank(pct=True) * 100
    india_pct = ins.rank(pct=True) * 100

    # Z-scores for divergence
    cz = (cs - cs.mean()) / cs.std()
    iz = (ins - ins.mean()) / ins.std()
    divergence = iz - cz                          # India minus China z-score

    # Current snapshot
    last = common[-1]
    return {
        "available": True,
        "overlap_months": len(common),
        "overlap_start": str(common[0].date()),
        "overlap_end": str(common[-1].date()),
        "china_percentile_now": float(china_pct.iloc[-1]),
        "india_percentile_now": float(india_pct.iloc[-1]),
        "divergence_now": float(divergence.iloc[-1]),
        "china_z_now": float(cz.iloc[-1]),
        "india_z_now": float(iz.iloc[-1]),
        "china_percentile_series": china_pct,
        "india_percentile_series": india_pct,
        "divergence_series": divergence,
        "overall_correlation": float(cs.corr(ins)),
        "interpretation": _interpret_cross_region(
            float(china_pct.iloc[-1]), float(india_pct.iloc[-1])),
    }


def _interpret_cross_region(china_p: float, india_p: float) -> str:
    """Plain-English summary of the current cross-region position."""
    def label(p):
        if p < 25: return "compressed"
        if p < 50: return "below average"
        if p < 75: return "above average"
        return "expanded"
    cl, il = label(china_p), label(india_p)
    if cl == il:
        return f"Both regions in {cl} territory (China P{china_p:.0f}, India P{india_p:.0f})."
    return (f"Divergent: China spread is {cl} (P{china_p:.0f}) while India spread "
            f"is {il} (P{india_p:.0f}).")
