"""
Narrator — generates trader-grade prose interpretations from pipeline output.

This module is the analyst's voice. It takes raw analytical results
(correlations, lead-lag tables, regime statistics, etc.) and produces
declarative, professional commentary in the style of the reference Tata
research reports.

Design principles:
    1. Deterministic — same numbers in, same prose out. No randomness.
    2. Number-grounded — every claim is anchored to a specific statistic.
    3. Variable-count-agnostic — works with 2 drivers or 20.
    4. Currency-aware — formats USD vs INR appropriately.
    5. Honest — never invents findings the data doesn't support.

Each public function returns a list of paragraph strings (HTML-safe). The
report builder concatenates them between charts to create the prose-heavy
narrative.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd


# ---------- Number formatting helpers ----------

def fmt_money(x: float, currency: str, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    sym = "$" if currency == "USD" else "₹" if currency == "INR" else f"{currency} "
    return f"{sym}{x:,.{decimals}f}/t"


def fmt_pct(x: float, decimals: int = 1, with_sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    sign = "+" if (with_sign and x >= 0) else ""
    return f"{sign}{x:.{decimals}f}%"


def fmt_num(x: float, decimals: int = 2, with_sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    sign = "+" if (with_sign and x >= 0) else ""
    return f"{sign}{x:,.{decimals}f}"


def _classify_correlation(r: float) -> str:
    a = abs(r)
    if a >= 0.85: return "very strong"
    if a >= 0.70: return "strong"
    if a >= 0.50: return "moderate"
    if a >= 0.30: return "weak"
    return "negligible"


def _classify_p_value(p: float, alpha: float = 0.05) -> str:
    if p is None or np.isnan(p):
        return "untestable"
    if p < 0.001: return "highly significant"
    if p < 0.01: return "significant"
    if p < alpha: return "marginally significant"
    return "not statistically significant"


def _key_box(title: str, body: str) -> dict:
    """Returns a dict the report builder converts to a KEY INTERPRETATION box."""
    return {"_kind": "key_interpretation", "title": title, "body": body}


# ---------- Section narrators ----------

def narrate_overview(meta: Dict, target: pd.Series, currency: str) -> List:
    """Opening paragraph for the region — describes the dataset and recent move."""
    out = []
    region_title = meta["region"].title()
    n_obs = meta["n_observations"]
    start, end = meta["date_start"], meta["date_end"]
    n_drivers = meta["n_drivers"]

    # Recent dynamics
    last_value = float(target.iloc[-1])
    one_year_ago = target.shift(12).iloc[-1] if len(target) >= 13 else None
    yoy_change = ((last_value - one_year_ago) / one_year_ago * 100) \
                  if one_year_ago and one_year_ago != 0 else None

    full_min = float(target.min()); full_max = float(target.max())
    full_max_date = target.idxmax().strftime("%b %Y")
    full_min_date = target.idxmin().strftime("%b %Y")

    p1 = (
        f"This section presents the {region_title} HRC market analysis, drawing on "
        f"<b>{n_obs} monthly observations</b> from <b>{start}</b> to <b>{end}</b>, "
        f"with <b>{n_drivers} driver variables</b> auto-detected from the source "
        f"spreadsheet. The target series is <i>{meta['target']}</i> denominated in "
        f"{currency}/t."
    )
    out.append(p1)

    p2 = (
        f"Across the full sample, prices have ranged from a low of "
        f"<b>{fmt_money(full_min, currency)}</b> ({full_min_date}) to a peak of "
        f"<b>{fmt_money(full_max, currency)}</b> ({full_max_date}) — a "
        f"<b>{(full_max/full_min - 1)*100:.0f}% peak-to-trough range</b> reflecting "
        f"the structural volatility characteristic of HRC markets. The latest "
        f"observation prints at <b>{fmt_money(last_value, currency)}</b>"
    )
    if yoy_change is not None:
        direction = "above" if yoy_change >= 0 else "below"
        p2 += (f", placing it {fmt_pct(abs(yoy_change))} {direction} the level "
               f"of one year ago.")
    else:
        p2 += "."
    out.append(p2)
    return out


def narrate_spread(spread_result, currency: str) -> List:
    """Prose for the spread analysis section."""
    if spread_result is None or isinstance(spread_result, dict):
        return [
            "<i>Spread analysis is unavailable for this region — the iron ore or "
            "HCC columns specified in config.yaml were not found in the data.</i>"
        ]

    out = []
    cur = spread_result.current_snapshot
    p = spread_result.percentiles
    formula_iron = spread_result.spread_config.get("iron_ore_weight", 1.6)
    formula_hcc = spread_result.spread_config.get("hcc_weight", 0.9)

    # Para 1: methodology
    p1 = (
        f"The spread metric quantifies the gross margin between HRC selling price "
        f"and the cost of its two major raw material inputs. Following Tata's BPM "
        f"convention, it is calculated as: "
        f"<b>Spread = HRC − ({formula_iron} × Iron Ore + {formula_hcc} × HCC)</b>. "
        f"This is the headline real-time signal for steel-mill margin health and the "
        f"primary input to capacity-utilisation decisions."
    )
    out.append(p1)

    # Para 2: current snapshot
    regime = cur["current_regime"]
    pct = cur["spread_percentile"]
    regime_phrase = {
        "compressed": "in the bottom quartile of historical experience",
        "normal": "within the typical range of historical experience",
        "expanded": "in the top quartile of historical experience",
    }.get(regime, "")

    p2 = (
        f"As of <b>{cur['as_of']}</b>, the spread stands at "
        f"<b>{fmt_money(cur['spread'], currency)}</b> — the <b>P{pct:.0f}</b> "
        f"percentile of its full-sample distribution, classifying the regime as "
        f"<b>{regime.upper()}</b>, {regime_phrase}. Raw material cost accounts for "
        f"<b>{cur['rm_cost']/cur['hrc']*100:.0f}%</b> of the HRC selling price, "
        f"leaving the remainder as gross margin available to mills before accounting "
        f"for energy, labour, and capital costs."
    )
    out.append(p2)

    # Para 3: vs median + 5y average
    vs_p50 = cur["vs_p50"]
    direction_p50 = "above" if vs_p50 >= 0 else "below"
    p3 = (
        f"The current spread is <b>{fmt_money(abs(vs_p50), currency)}</b> "
        f"{direction_p50} the historical median of {fmt_money(p['p50'], currency)}"
    )
    if not np.isnan(cur.get("vs_5y_avg", np.nan)):
        vs_5y = cur["vs_5y_avg"]
        direction_5y = "above" if vs_5y >= 0 else "below"
        p3 += (f" and {fmt_money(abs(vs_5y), currency)} {direction_5y} the "
               f"five-year rolling average")
    p3 += "."
    if regime == "compressed":
        p3 += (
            " Spreads at this level historically coincide with mill capacity "
            "rationalisation, deferred capex, and inventory destocking — providing "
            "a structural cost floor that limits further downside but signals weak "
            "current profitability."
        )
    elif regime == "expanded":
        p3 += (
            " Spreads at this level historically reflect either a supply shock in "
            "raw materials cleared by demand-pass-through, or genuine demand surge "
            "outpacing input cost inflation — both conditions tend to be transient "
            "and revert toward the median within 6–12 months."
        )
    else:
        p3 += (
            " This is the typical operating regime for the steel cycle — neither "
            "stressed nor euphoric, with input costs and selling prices in "
            "approximate equilibrium."
        )
    out.append(p3)

    # Key interpretation box
    out.append(_key_box(
        "Key Interpretation",
        f"The {spread_result.region.title()} HRC spread is currently "
        f"<b>{regime.upper()}</b> at <b>P{pct:.0f}</b> of its historical "
        f"distribution. Raw material costs represent "
        f"{cur['rm_cost']/cur['hrc']*100:.0f}% of selling price. "
        f"Mill margins are {('under structural pressure' if regime == 'compressed' else 'unusually strong' if regime == 'expanded' else 'within their normal operating range')}."
    ))
    return out


def narrate_correlation(corr_df: pd.DataFrame, target: str,
                          currency: str) -> List:
    """Prose for the correlation matrix."""
    if corr_df is None or isinstance(corr_df, dict) or target not in corr_df.columns:
        return ["<i>Correlation analysis unavailable.</i>"]
    out = []
    target_corr = corr_df[target].drop(target).sort_values(
        key=lambda s: s.abs(), ascending=False)

    # Identify drivers + relationships
    strongest = target_corr.iloc[0]
    strongest_var = target_corr.index[0]
    weakest_strong = target_corr[target_corr.abs() >= 0.5]
    negative_strong = target_corr[(target_corr <= -0.30)]

    p1 = (
        f"The correlation matrix shows the linear pairwise relationships between "
        f"<i>{target}</i> and the {len(target_corr)} driver variables. "
        f"Correlations are computed on price levels, not returns — therefore they "
        f"capture co-movement of trends rather than co-movement of innovations. The "
        f"strongest relationship is with <b>{strongest_var}</b> "
        f"(r = {strongest:+.2f}, {_classify_correlation(strongest)} "
        f"{'positive' if strongest >= 0 else 'negative'}), "
    )
    if abs(strongest) >= 0.70:
        p1 += (
            f"indicating that this driver is a primary candidate for short-run "
            f"price-formation analysis."
        )
    else:
        p1 += (
            f"suggesting that no single driver dominates HRC price formation in "
            f"this dataset."
        )
    out.append(p1)

    # Multicollinearity warning
    high_pairs = []
    cols = [c for c in corr_df.columns if c != target]
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr_df.loc[c1, c2]
            if abs(r) > 0.85:
                high_pairs.append((c1, c2, r))
    if high_pairs:
        p2 = (
            f"<b>Multicollinearity flag:</b> {len(high_pairs)} driver pair(s) "
            f"exhibit correlations above |0.85| — namely "
        )
        p2 += "; ".join(f"<b>{a}</b> ↔ <b>{b}</b> (r={r:+.2f})"
                          for a, b, r in high_pairs[:3])
        p2 += (
            ". When two drivers are nearly perfectly collinear, a multivariate "
            "regression cannot reliably attribute price moves between them — the "
            "individual coefficients become unstable. Consider dropping one of "
            "each pair from the modelling specification."
        )
        out.append(p2)

    # Strong drivers
    if len(weakest_strong) > 0:
        names = [f"<b>{v}</b> (r={target_corr[v]:+.2f})"
                 for v in weakest_strong.index]
        p3 = (
            f"Drivers with at least moderate co-movement with HRC: "
            + ", ".join(names) + "."
        )
        if len(negative_strong) > 0:
            neg_names = ", ".join(f"<b>{v}</b>"
                                    for v in negative_strong.index)
            p3 += (
                f" The negative correlations of {neg_names} merit attention: in "
                f"commodity markets, structurally negative correlation typically "
                f"reflects either a financial-conditions channel (interest rates, "
                f"currency strength) or a cross-asset substitution effect."
            )
        out.append(p3)

    out.append(_key_box(
        "Key Interpretation",
        f"<b>{strongest_var}</b> exhibits the strongest co-movement with HRC "
        f"(r = {strongest:+.2f}), positioning it as the primary tracking variable. "
        f"{'Multicollinearity is present and should be addressed in modelling. ' if high_pairs else ''}"
        f"{len(weakest_strong)} of {len(target_corr)} drivers cross the moderate-correlation threshold (|r| ≥ 0.5)."
    ))
    return out


def narrate_lead_lag(ll_df: pd.DataFrame, target: str) -> List:
    """Prose for the lead-lag analysis."""
    if ll_df is None or isinstance(ll_df, dict) or len(ll_df) == 0:
        return ["<i>Lead-lag analysis unavailable.</i>"]
    out = []

    # Top by absolute correlation
    top = ll_df.iloc[0]
    leading = ll_df[ll_df["best_lag_months"] > 0].sort_values(
        "ccf_at_best_lag", key=lambda s: s.abs(), ascending=False)
    coincident = ll_df[ll_df["best_lag_months"] == 0]
    lagging = ll_df[ll_df["best_lag_months"] < 0].sort_values(
        "ccf_at_best_lag", key=lambda s: s.abs(), ascending=False)

    granger_sig = ll_df[ll_df["granger_x_causes_y"] == True]

    p1 = (
        f"Lead-lag analysis decomposes each driver's co-movement with HRC across "
        f"lags from −12 to +12 months. A <b>positive lag</b> means the driver "
        f"moves first and HRC follows (a leading indicator); a <b>negative lag</b> "
        f"means HRC moves first and the driver follows (a lagging indicator); "
        f"<b>lag zero</b> indicates a coincident relationship. The complementary "
        f"Granger causality test asks the stricter question: does the driver's "
        f"history actually improve HRC forecasts beyond HRC's own lags?"
    )
    out.append(p1)

    summary_parts = []
    if len(leading) > 0:
        names = ", ".join(
            f"<b>{r['driver']}</b> ({r['best_lag_months']:+.0f}m, "
            f"r={r['ccf_at_best_lag']:+.2f})"
            for _, r in leading.head(3).iterrows()
        )
        summary_parts.append(f"<b>Leading indicators</b> — {names}")
    if len(coincident) > 0:
        names = ", ".join(
            f"<b>{r['driver']}</b> (r={r['ccf_at_best_lag']:+.2f})"
            for _, r in coincident.head(3).iterrows()
        )
        summary_parts.append(f"<b>Coincident</b> — {names}")
    if len(lagging) > 0:
        names = ", ".join(
            f"<b>{r['driver']}</b> ({r['best_lag_months']:+.0f}m, "
            f"r={r['ccf_at_best_lag']:+.2f})"
            for _, r in lagging.head(3).iterrows()
        )
        summary_parts.append(f"<b>Lagging</b> — {names}")
    if summary_parts:
        out.append("Classified by best-lag direction: " + ". ".join(summary_parts) + ".")

    # Granger commentary
    if len(granger_sig) > 0:
        names = ", ".join(f"<b>{n}</b>" for n in granger_sig["driver"].head(5))
        p3 = (
            f"<b>Granger-significant drivers</b> — {len(granger_sig)} of "
            f"{len(ll_df)} drivers genuinely improve HRC forecasts at conventional "
            f"significance levels: {names}. These are the variables most worth "
            f"watching for short-run price prediction. The remainder show "
            f"correlation but no incremental forecast value."
        )
        out.append(p3)
    else:
        out.append(
            "<b>No driver passes the Granger causality test</b> at the 5% "
            "significance level. While correlations exist, no driver provides "
            "incremental predictive content for HRC beyond HRC's own lagged values "
            "in this sample. This is a cautionary finding for any forecasting "
            "model that relies on driver-based prediction."
        )

    # Lagging insight
    if len(lagging) > 0:
        lag_drivers = lagging["driver"].tolist()
        if len(lag_drivers) > 0:
            out.append(
                f"<b>Counter-intuitive lagging signals:</b> {', '.join(lag_drivers)} "
                f"appear to <i>follow</i> HRC rather than lead it. In commodity "
                f"markets this typically reflects either (1) HRC responding faster "
                f"to a common macro shock that takes longer to transmit to the "
                f"trailing variable, or (2) a feedback channel where steel demand "
                f"itself drives the variable in question."
            )

    # Key box
    direction_word = "leading" if top["best_lag_months"] > 0 else "coincident" \
                     if top["best_lag_months"] == 0 else "lagging"
    out.append(_key_box(
        "Key Interpretation",
        f"<b>{top['driver']}</b> is the dominant relationship with HRC "
        f"(r = {top['ccf_at_best_lag']:+.2f} at lag {top['best_lag_months']:+.0f} "
        f"months — {direction_word}). "
        f"{len(granger_sig)} of {len(ll_df)} drivers are Granger-significant. "
        f"Real-time monitoring should prioritise {('leading' if len(leading) > 0 else 'coincident')} "
        f"drivers: {(leading['driver'].iloc[0] if len(leading) > 0 else top['driver'])}."
    ))
    return out


def narrate_diagnostics(adf_df: pd.DataFrame, vif_df: pd.DataFrame,
                         target: str) -> List:
    """Prose for the diagnostics section (stationarity + VIF)."""
    out = []

    if adf_df is not None and not isinstance(adf_df, dict) and len(adf_df) > 0:
        target_lvl = adf_df[(adf_df["variable"] == target) &
                              (adf_df["transform"] == "levels")]
        target_diff = adf_df[(adf_df["variable"] == target) &
                              (adf_df["transform"] == "first_diff")]
        target_lvl_p = float(target_lvl["p_value"].iloc[0]) if len(target_lvl) > 0 else None
        target_diff_p = float(target_diff["p_value"].iloc[0]) if len(target_diff) > 0 else None

        # Count I(0) and I(1)
        levels = adf_df[adf_df["transform"] == "levels"]
        diffs = adf_df[adf_df["transform"] == "first_diff"]
        n_i0 = int((levels["p_value"] < 0.05).sum())
        n_i1 = int((levels["p_value"] >= 0.05).sum() &
                    (diffs["p_value"] < 0.05).sum() if len(diffs) > 0 else 0)
        n_total = int(len(levels))

        p1 = (
            f"The Augmented Dickey-Fuller test checks whether each series is "
            f"stationary (I(0)) or contains a unit root (I(1) or higher). "
            f"Non-stationary series cannot be used in level OLS regressions without "
            f"producing spurious results. "
        )
        if target_lvl_p is not None:
            target_status = "stationary in levels" if target_lvl_p < 0.05 \
                             else "non-stationary in levels"
            p1 += (
                f"<i>{target}</i> is <b>{target_status}</b> "
                f"(ADF p = {target_lvl_p:.3f})"
            )
            if target_lvl_p >= 0.05 and target_diff_p is not None:
                if target_diff_p < 0.05:
                    p1 += (
                        f", but its first difference is stationary "
                        f"(ADF p = {target_diff_p:.3f}), classifying the series as "
                        f"<b>I(1)</b>. ARIMA-class models with d=1 differencing are "
                        f"appropriate."
                    )
                else:
                    p1 += (
                        f", and even its first difference fails the test "
                        f"(p = {target_diff_p:.3f}). The series may be I(2) — "
                        f"second-differencing should be considered."
                    )
            else:
                p1 += "."
        out.append(p1)

        # Mixed-order finding
        if n_i0 > 0 and n_i0 < n_total:
            out.append(
                f"<b>Mixed integration order detected:</b> {n_i0} of {n_total} "
                f"variables are I(0) in levels, while the rest require differencing. "
                f"This invalidates the Engle-Granger cointegration test — which "
                f"assumes all variables are I(1) — and motivates the ARDL Bounds "
                f"Testing framework (Pesaran, Shin & Smith 2001) which accommodates "
                f"any combination of I(0) and I(1) regressors."
            )

    if vif_df is not None and not isinstance(vif_df, dict) and len(vif_df) > 0:
        severe = vif_df[vif_df["vif"] > 10]
        moderate = vif_df[(vif_df["vif"] > 5) & (vif_df["vif"] <= 10)]
        p3 = (
            f"The Variance Inflation Factor (VIF) measures multicollinearity — how "
            f"much each driver's variance is inflated by its correlation with the "
            f"other drivers. A VIF > 10 is severe; VIF 5–10 is moderate; below 5 "
            f"is acceptable. "
        )
        if len(severe) > 0:
            names = ", ".join(f"<b>{r['variable']}</b> (VIF={r['vif']:.1f})"
                                for _, r in severe.iterrows())
            p3 += (
                f"<b>Severe multicollinearity</b> detected for {len(severe)} "
                f"variable(s): {names}. These variables are nearly redundant for "
                f"regression purposes and should typically be reduced to a single "
                f"representative."
            )
        elif len(moderate) > 0:
            names = ", ".join(f"<b>{r['variable']}</b> (VIF={r['vif']:.1f})"
                                for _, r in moderate.iterrows())
            p3 += (
                f"No severe multicollinearity. {len(moderate)} variable(s) show "
                f"moderate VIF — acceptable but worth monitoring: {names}."
            )
        else:
            p3 += "All variables have VIF below 5 — no multicollinearity concerns."
        out.append(p3)

    # Key box
    if vif_df is not None and not isinstance(vif_df, dict) and len(vif_df) > 0:
        max_vif_row = vif_df.iloc[0]
        max_vif = float(max_vif_row["vif"])
        out.append(_key_box(
            "Key Interpretation",
            f"Most series require first-differencing for stationarity (I(1) "
            f"behaviour). Highest VIF: <b>{max_vif_row['variable']}</b> at {max_vif:.1f} "
            f"({'severe' if max_vif > 10 else 'moderate' if max_vif > 5 else 'acceptable'}). "
            f"ARDL is the appropriate framework given mixed integration orders."
        ))
    return out


def narrate_cyclicity(cyc_result, currency: str) -> List:
    """Prose for the cyclicity section — regimes, cycles, transitions."""
    if cyc_result is None:
        return ["<i>Cyclicity analysis unavailable.</i>"]

    out = []
    n_regimes = cyc_result.n_regimes
    profiles = cyc_result.regime_profiles
    cycle = cyc_result.cycle_stats
    cur_r = cyc_result.current_regime
    cur_profile = profiles[cur_r]

    # Para 1: methodology
    p1 = (
        f"This section identifies the cyclical structure of the HRC market using "
        f"a multi-feature Gaussian Mixture Model (GMM) on five derived features: "
        f"log-level, 1-month log-return, 3-month cumulative return, 3-month "
        f"rolling volatility, and 6-month momentum. The GMM identifies "
        f"<b>{n_regimes} distinct regimes</b>, ranked here from lowest to highest "
        f"by mean target price. Each regime is characterised by both its level "
        f"and its dynamic behaviour, allowing the same price to belong to "
        f"different regimes depending on whether the market is trending, "
        f"oscillating, or stagnant."
    )
    out.append(p1)

    # Para 2: regime summary
    summary_lines = []
    for p in profiles:
        if p.n_months > 0:
            summary_lines.append(
                f"<b>R{p.regime_id} ({p.label})</b> — {p.n_months} months across "
                f"{p.n_spells} spell(s), avg duration {p.avg_spell_duration:.1f}m, "
                f"mean price {fmt_money(p.mean_target, currency)}, "
                f"avg return {fmt_pct(p.avg_monthly_return_pct, with_sign=True)}/month"
            )
    p2 = "Regime profile: " + "; ".join(summary_lines) + "."
    out.append(p2)

    # Para 3: cycle stats
    if cycle.n_peaks > 0:
        p3_parts = [
            f"Across the full sample, <b>{cycle.n_peaks} peaks</b> and "
            f"<b>{cycle.n_troughs} troughs</b> are identified using prominence-"
            f"based detection"
        ]
        if cycle.avg_peak_to_peak_months:
            p3_parts.append(
                f"average peak-to-peak spacing is "
                f"<b>{cycle.avg_peak_to_peak_months:.1f} months</b>"
            )
        if cycle.avg_amplitude_pct:
            p3_parts.append(
                f"average peak-to-trough amplitude of "
                f"<b>{cycle.avg_amplitude_pct:.1f}%</b>"
            )
        p3 = "; ".join(p3_parts) + "."
        if cycle.dominant_spectral_period_months:
            p3 += (
                f" Spectral analysis (DFT periodogram on log-returns) identifies a "
                f"dominant cycle period of <b>{cycle.dominant_spectral_period_months:.1f} "
                f"months</b> — consistent with the inventory-restocking rhythm of "
                f"steel service centres."
            )
        out.append(p3)

    # Para 4: per-regime cycles
    per_reg_lines = []
    for r in range(n_regimes):
        c = cyc_result.per_regime_cycles.get(r)
        if c is None or c.n_peaks == 0:
            continue
        line = f"<b>R{r}</b>"
        if c.dominant_spectral_period_months:
            line += f" cycles at ~{c.dominant_spectral_period_months:.0f}m"
        if c.avg_amplitude_pct:
            line += f" with {c.avg_amplitude_pct:.0f}% avg amplitude"
        gp = cyc_result.garch_persistence.get(r)
        if gp is not None:
            line += f", GARCH persistence {gp:.2f}"
        per_reg_lines.append(line)
    if per_reg_lines:
        out.append(
            "Per-regime cycle frequencies and volatility persistence: "
            + "; ".join(per_reg_lines) + "."
        )

    # Para 5: Markov transitions
    sp = cyc_result.self_persistence
    most_persistent = max(sp.items(), key=lambda kv: kv[1])
    least_persistent = min(sp.items(), key=lambda kv: kv[1])
    p5 = (
        f"The Markov transition matrix exhibits diagonal dominance, confirming "
        f"that regimes persist once entered. <b>R{most_persistent[0]}</b> is the "
        f"stickiest state with self-transition probability "
        f"{most_persistent[1]*100:.0f}%, while <b>R{least_persistent[0]}</b> is "
        f"the most transient ({least_persistent[1]*100:.0f}% self-transition). "
    )
    # Look for skipped transitions — physical-enforcement signal
    tm = cyc_result.transition_matrix.values
    skipped = []
    for i in range(n_regimes):
        for j in range(n_regimes):
            if abs(i - j) >= 2 and tm[i, j] == 0 and i != 0:
                skipped.append((i, j))
    if any(tm[0, n_regimes - 1] < 0.05 for _ in [0]):
        p5 += (
            f"Notably, direct R0 → R{n_regimes-1} transitions are essentially "
            f"absent — the market never jumps from stagnation to super-cycle "
            f"without first passing through intermediate regimes. This is not a "
            f"statistical artefact but a physical constraint: steel mills require "
            f"sustained capacity utilisation increases before supply genuinely "
            f"becomes inelastic."
        )
    out.append(p5)

    # Para 6: macro cycles
    if len(cyc_result.macro_cycles) > 0:
        n_mc = len(cyc_result.macro_cycles)
        cycle_descs = []
        for _, mc in cyc_result.macro_cycles.iterrows():
            cycle_descs.append(
                f"<b>{mc['start']} → {mc['end']}</b> "
                f"({mc['duration_months']}m, peak {fmt_money(mc['peak_target'], currency)})"
            )
        out.append(
            f"<b>Macro-cycle identification:</b> {n_mc} complete macro cycle(s) "
            f"detected — " + "; ".join(cycle_descs) + ". A macro cycle is defined "
            f"as a contiguous traversal of regimes that touches the highest-"
            f"intensity regime. This is the cycle relevant for capex decisions, "
            f"long-term contract pricing, and strategic capacity planning — "
            f"considerably longer than the 8–12 month tactical inventory cycle."
        )
    else:
        out.append(
            "<b>No complete macro cycle</b> has yet been observed in this region's "
            "history. The market has not traversed the full sequence "
            f"{' → '.join(f'R{i}' for i in range(n_regimes))} → R0 within the "
            "available sample. This may reflect insufficient history, structurally "
            "lower volatility relative to the China benchmark, or simply that the "
            "current macro cycle is still in progress."
        )

    # Key box
    cur_label = profiles[cur_r].label
    msg = (
        f"The market currently sits in <b>R{cur_r} ({cur_label})</b> at "
        f"{fmt_money(cur_profile.mean_target, currency)} mean. "
    )
    if cycle.dominant_spectral_period_months:
        msg += f"Dominant cycle frequency: ~{cycle.dominant_spectral_period_months:.0f} months. "
    msg += f"{cycle.n_peaks} historical peaks, average amplitude {cycle.avg_amplitude_pct:.0f}%." \
           if cycle.avg_amplitude_pct else f"{cycle.n_peaks} historical peaks identified."
    out.append(_key_box("Key Interpretation", msg))
    return out


def narrate_cross_region(cross_data: Dict) -> List:
    """Prose for the cross-region (China vs India) comparison."""
    if not cross_data or not cross_data.get("available"):
        return ["<i>Cross-region comparison unavailable — both China and India regions "
                "must be enabled with sufficient overlapping data.</i>"]
    out = []
    cp = cross_data["china_percentile_now"]
    ip = cross_data["india_percentile_now"]
    div = cross_data["divergence_now"]
    corr = cross_data["overall_correlation"]

    p1 = (
        f"This section compares HRC spread dynamics in China (USD/t) and India "
        f"(INR/t). Because the two regions price in different currencies, absolute "
        f"comparison is meaningless. We instead rank each region's spread within "
        f"its own historical distribution and use z-scores to measure divergence. "
        f"Across the {cross_data['overlap_months']} months of overlapping data "
        f"({cross_data['overlap_start']} to {cross_data['overlap_end']}), the two "
        f"spreads exhibit a <b>{_classify_correlation(corr)} {'positive' if corr >= 0 else 'negative'} "
        f"correlation of r = {corr:+.2f}</b>, indicating "
        f"{'similar cyclical behaviour' if corr >= 0.5 else 'partially independent dynamics' if corr >= 0 else 'inverse cyclical behaviour'} "
        f"despite the regions' different macro environments."
    )
    out.append(p1)

    p2 = (
        f"At the latest observation: <b>China's spread is at P{cp:.0f}</b> of "
        f"its own history, while <b>India's spread is at P{ip:.0f}</b>. "
        f"The divergence (India minus China z-score) stands at {div:+.2f}, "
    )
    if div > 0.5:
        p2 += (
            f"a meaningfully positive value indicating that <b>Indian mills currently "
            f"enjoy relatively healthier margin conditions than their Chinese "
            f"counterparts</b>. This kind of divergence typically reflects either "
            f"localised demand strength in India, structurally different cost "
            f"bases, or import-parity-pricing that insulates Indian domestic prices "
            f"from international weakness."
        )
    elif div < -0.5:
        p2 += (
            f"a meaningfully negative value indicating that <b>Chinese mills "
            f"currently enjoy relatively healthier margin conditions than Indian "
            f"counterparts</b> — an unusual configuration that warrants investigation "
            f"into India-specific cost or demand pressures."
        )
    else:
        p2 += (
            f"a near-neutral value indicating that both regions are operating in "
            f"approximately the same position within their respective historical "
            f"distributions."
        )
    out.append(p2)

    p3 = (
        f"<b>Trade-flow implication:</b> when China's spread is materially below "
        f"India's (Indian relative healthier), Chinese export economics "
        f"deteriorate while Indian domestic margins are protected. Conversely, "
        f"when China's spread expands sharply above India's, Chinese exporters "
        f"can price aggressively into India, pressuring local domestic prices. "
        f"The current configuration "
    )
    if div > 0.3:
        p3 += "favours India domestic producers."
    elif div < -0.3:
        p3 += "favours Chinese exporters and pressures Indian mills."
    else:
        p3 += "is broadly neutral for cross-region trade flows."
    out.append(p3)

    out.append(_key_box(
        "Key Interpretation",
        cross_data["interpretation"]
    ))
    return out


def narrate_models(models_dict: Dict, currency: str, target: str) -> List:
    """Prose for the modelling section."""
    out = []
    valid_models = {
        name: result for name, result in models_dict.items()
        if not (isinstance(result, dict) and "error" in result)
        and not getattr(result, "error", None)
    }
    failed = {
        name: (result.get("error") if isinstance(result, dict)
               else getattr(result, "error", "?"))
        for name, result in models_dict.items()
        if (isinstance(result, dict) and "error" in result)
        or getattr(result, "error", None)
    }

    if not valid_models and not failed:
        return ["<i>No models were configured for this region.</i>"]

    p1 = (
        f"This section reports forecasts and volatility models. The pipeline runs "
        f"every model enabled in <code>config.yaml</code>; new models can be added "
        f"by dropping a Python file into the <code>models/</code> directory. "
    )
    if failed:
        p1 += (
            f"<b>{len(failed)} model(s) failed to fit</b>: "
            + ", ".join(f"<b>{n}</b> ({err[:60]}...)" for n, err in failed.items())
            + ". "
        )
    if valid_models:
        p1 += (
            f"<b>{len(valid_models)} model(s) successfully produced output</b>: "
            + ", ".join(f"<b>{n.upper()}</b>" for n in valid_models.keys())
            + "."
        )
    out.append(p1)

    # Find best forecasting model by RMSE
    best_model = None; best_rmse = float("inf")
    for name, result in valid_models.items():
        metrics = getattr(result, "metrics", {}) or {}
        rmse = metrics.get("rmse")
        if rmse is not None and not np.isnan(rmse) and rmse < best_rmse:
            best_rmse = rmse; best_model = name

    if best_model:
        m = valid_models[best_model]
        metrics = m.metrics
        out.append(
            f"<b>Forecasting performance.</b> Out-of-sample test: the best-performing "
            f"model on the held-out window is <b>{best_model.upper()}</b> with "
            f"RMSE = {fmt_money(best_rmse, currency)}, "
            f"MAPE = {metrics.get('mape', float('nan')):.2f}%, "
            f"and R² = {metrics.get('r2', float('nan')):.3f}. "
            f"This model's 12-month forecast is shown alongside its in-sample fit "
            f"and 95% confidence interval below."
        )

    # GARCH commentary if present
    if "garch" in valid_models:
        g = valid_models["garch"]
        if hasattr(g, "diagnostics"):
            params = g.diagnostics.get("params", {})
            alpha = params.get("alpha[1]", 0)
            beta = params.get("beta[1]", 0)
            persist = alpha + beta
            half_life = (np.log(0.5) / np.log(persist)) if 0 < persist < 1 else None
            out.append(
                f"<b>GARCH(1,1) volatility model.</b> Persistence (α+β) = "
                f"<b>{persist:.3f}</b>"
                + (f" — implying a volatility half-life of "
                   f"<b>{half_life:.1f} months</b>." if half_life else ".")
                + " High persistence means that volatility shocks decay slowly, "
                "compounding tail risk during stress periods. Risk models calibrated "
                "in calm periods will materially underestimate tail risk during "
                "high-volatility regimes."
            )

    return out


def narrate_attribution(attr_result, currency: str) -> List:
    """Prose for the rolling attribution section."""
    if attr_result is None or len(attr_result.rolling_betas) == 0:
        return ["<i>Attribution analysis unavailable.</i>"]

    out = []
    cur = attr_result.current_attribution.dropna().sort_values(ascending=False)
    fs = attr_result.full_sample_attribution.dropna().sort_values(ascending=False)

    p1 = (
        f"Rolling attribution decomposes the variance of HRC into the relative "
        f"contributions of each driver, using a 24-month rolling regression. The "
        f"shares are normalised to sum to 100% and reflect each driver's share of "
        f"explanatory power within that window. Tracking how these shares evolve "
        f"over time reveals when the price-formation process structurally shifts."
    )
    out.append(p1)

    # Compare current vs full-sample
    if len(cur) >= 2 and len(fs) >= 2:
        cur_top = cur.index[0]
        fs_top = fs.index[0]
        if cur_top == fs_top:
            p2 = (
                f"The current dominant driver, <b>{cur_top}</b> "
                f"({cur.iloc[0]:.1f}% of explanatory power), matches the full-sample "
                f"leader — indicating a stable price-formation regime."
            )
        else:
            p2 = (
                f"<b>Driver rotation observed.</b> Over the full sample, "
                f"<b>{fs_top}</b> ({fs.iloc[0]:.1f}%) is the dominant driver, but "
                f"in the most recent rolling window, <b>{cur_top}</b> "
                f"({cur.iloc[0]:.1f}%) has overtaken it. This kind of structural "
                f"shift in driver hierarchy typically signals a regime transition "
                f"and warrants a re-examination of forecasting assumptions."
            )
        out.append(p2)

        # Top 3 currently
        top3_names = ", ".join(f"<b>{v}</b> ({cur[v]:.0f}%)"
                                  for v in cur.head(3).index)
        out.append(
            f"<b>Top 3 current drivers</b> (most recent window): {top3_names}. "
            f"These three variables together explain "
            f"<b>{cur.head(3).sum():.0f}%</b> of HRC's variance — they are the "
            f"primary monitoring targets for short-run price prediction."
        )

    return out


# ---------- MACRO CALENDAR narration ----------

def _channel_label(channel_id: str) -> str:
    """Convert primary_channel id to a human-readable label."""
    return {
        "dxy_to_iron_ore": "USD → Iron Ore",
        "china_demand_direct": "China Demand → Iron Ore + HRC",
        "china_export_intent": "China Export Pressure → India Import Parity",
        "us_demand_to_china_exports": "US Demand → Chinese Steel Exports",
        "european_steel_demand": "EU Demand → Chinese Steel Exports",
        "oil_to_steel_input": "Oil → Steel Energy + Freight",
    }.get(channel_id, channel_id.replace("_", " ").title())


def _impact_word(impact: str) -> str:
    return {"HIGH": "high-impact", "MED": "medium-impact",
            "LOW": "low-impact"}.get(impact, impact.lower())


def narrate_macro_calendar_intro(calendar_result) -> List:
    """Opening paragraphs for the macro calendar section."""
    cal = calendar_result
    out = []
    p1 = (
        f"This section forward-projects HRC steel exposure to the "
        f"<b>{len(cal.events)} HRC-relevant macro events</b> scheduled across "
        f"<b>{cal.window_start}</b> to <b>{cal.window_end}</b> — the next ~45 trading days. "
        f"Of these, <b>{cal.n_high} are HIGH-impact</b> events expected to produce "
        f"material price moves within days of release, and <b>{cal.n_med} are MED-impact</b> "
        f"events that may transmit second-order effects through the iron ore or "
        f"INR channels. Each event is presented with its transmission mechanism "
        f"to HRC, expected reaction by region, and historical analogues from the "
        f"data record."
    )
    out.append(p1)

    p2 = (
        f"The methodology applies the trading-desk framework from the reference "
        f"macro fluency reports — translated for HRC analysis. Each event affects "
        f"HRC through one or more of six identified transmission channels: "
        f"<b>USD → Iron Ore</b> (dollar strength inversely moves USD-priced iron ore), "
        f"<b>China Demand → Iron Ore + HRC</b> (direct Chinese consumption signal), "
        f"<b>China Export Pressure → India Import Parity</b> (Chinese surplus exports "
        f"pressure Indian domestic prices), "
        f"<b>US Demand → Chinese Steel Exports</b> (consumer durables drive global "
        f"steel demand), <b>EU Demand → Chinese Steel Exports</b>, and "
        f"<b>Oil → Steel Energy + Freight</b> (oil flows into mill operating cost "
        f"and shipping)."
    )
    out.append(p2)

    # Current setup summary
    if cal.events:
        e = cal.events[0]  # all events share the same 'current setup'
        regime_phrase = {
            "compressed": "in the bottom quartile",
            "normal": "in the typical range",
            "expanded": "in the top quartile",
        }.get(e.spread_regime, "in unknown territory")
        p3 = (
            f"<b>Current market setup:</b> Iron ore is at the "
            f"<b>P{e.iron_ore_percentile:.0f}</b> percentile of its historical "
            f"distribution. The China spread is currently classified as "
            f"<b>{e.spread_regime.upper()}</b> ({regime_phrase} of historical "
            f"experience). The historical analogues shown for each event are drawn "
            f"from past months matching this setup — providing an empirical anchor "
            f"for the expected HRC trajectory if conditions hold."
        )
        out.append(p3)

    out.append(_key_box(
        "Key Interpretation",
        f"<b>{cal.n_high} HIGH-impact events</b> in the next 45 days. Highest-priority "
        f"window: <b>April 28 – May 2 ('Mega Week')</b> — FOMC, ECB, US Q1 GDP, and "
        f"NFP all clustered in 5 trading days. China Q1 GDP (Apr 21) is the single "
        f"most consequential data point for HRC. With the market currently in "
        f"a <b>{cal.events[0].spread_regime if cal.events else 'unknown'}</b> spread regime, "
        f"asymmetric downside risk on hawkish surprises is the primary positioning "
        f"consideration."
    ))
    return out


def narrate_event(event, currency: str = "USD") -> dict:
    """
    Generate the prose block for a single event.
    Returns a dict the report builder converts to an event card.
    """
    e = event

    # Build the opening line with timing context
    if e.days_until > 0:
        timing = f"in <b>{e.days_until} days</b>"
    elif e.days_until == 0:
        timing = "<b>today</b>"
    elif e.days_until > -7:
        timing = f"<b>{abs(e.days_until)} days ago</b>"
    else:
        timing = f"on {e.event_date} (past)"

    intro = (
        f"<b>{e.name}</b> is a {_impact_word(e.impact)} {e.category.replace('_', ' ')} "
        f"event scheduled {timing}. Consensus expectation: {e.consensus}. "
        f"The primary HRC transmission channel is <b>{_channel_label(e.primary_channel)}</b>, "
        f"affecting "
        f"{', '.join(r.title() for r in e.affects_regions) if e.affects_regions else 'no specific region'}."
    )

    mechanism_para = e.mechanism

    reaction_para = (
        f"<b>Expected HRC reaction:</b> {e.expected_hrc_reaction}"
    )

    # Historical analogue paragraph
    analogue_para = ""
    if e.analogue_confidence in ("high", "medium") and e.cond_n >= 3:
        analogue_para = (
            f"<b>Historical analogues</b> ({e.analogue_confidence} confidence, "
            f"{e.cond_n} matching past months with similar setup — iron ore at "
            f"P{e.iron_ore_percentile:.0f} ±20 and {e.spread_regime} spread regime): "
            f"in those analogous setups, HRC moved on average "
            f"<b>{e.cond_avg_30d:+.1f}%</b> over the following 30 days, "
            f"<b>{e.cond_avg_60d:+.1f}%</b> over 60 days, and "
            f"<b>{e.cond_avg_90d:+.1f}%</b> over 90 days. This is the empirical "
            f"baseline against which the expected reaction above should be read — "
            f"the mechanism predicts <i>direction</i>, the analogues calibrate "
            f"<i>magnitude</i>."
        )
    elif e.analogue_confidence == "low" and e.simple_n >= 6:
        analogue_para = (
            f"<b>Historical analogues</b> (low confidence — only {e.cond_n} months "
            f"match the precise current setup; falling back to broader sample of "
            f"{e.simple_n} recent months): average HRC move "
            f"<b>{e.simple_avg_30d:+.1f}%</b> over 30 days, "
            f"<b>{e.simple_avg_60d:+.1f}%</b> over 60 days. These figures reflect "
            f"general HRC drift rather than setup-conditional behaviour and should "
            f"be read with caution."
        )
    else:
        analogue_para = (
            f"<b>Historical analogues:</b> insufficient matching past setups to "
            f"compute a reliable analogue (current setup is unusual). Rely on the "
            f"transmission mechanism above for directional guidance."
        )

    return {
        "_kind": "event_card",
        "date": e.event_date,
        "name": e.name,
        "country": e.country,
        "impact": e.impact,
        "category": e.category,
        "intro": intro,
        "mechanism": mechanism_para,
        "reaction": reaction_para,
        "analogue": analogue_para,
        "confidence": e.analogue_confidence,
        "cond_n": e.cond_n,
    }


def narrate_macro_calendar(calendar_result) -> List:
    """
    Top-level narration: intro paragraphs + per-event cards.
    Returns a list of mixed prose strings, key_interpretation dicts,
    and event_card dicts. Report builder + dashboard render each accordingly.
    """
    out = list(narrate_macro_calendar_intro(calendar_result))
    for ev in calendar_result.events:
        out.append(narrate_event(ev))
    return out
