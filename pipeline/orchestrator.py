"""
Orchestrator — runs the full pipeline end-to-end for all enabled regions.

Returns a single dict with everything needed to build the report or feed
the dashboard. The dashboard reads the cached results.json; running the
pipeline regenerates it.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from .data_loader import load_data, Dataset, RegionData
from .diagnostics import adf_table, vif_table, correlation_matrix, summary_stats
from .lead_lag import lead_lag_summary, rolling_correlations, cross_correlation, lag_matrix
from .spread import analyse_region as analyse_spread, cross_region_comparison
from .attribution import rolling_attribution
from .events import run_event_analysis
from .cyclicity import analyse_cyclicity
from .event_deep_dive import (build_event_candidates, analyse_event_deep_dive,
                              load_event_context)
from .macro_calendar import analyse_macro_calendar
from models.registry import build_models, list_available


def _step(label, fn, *args, **kwargs):
    """Run a pipeline step with timing + friendly logging."""
    print(f" • {label}...", end="", flush=True)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        print(f" (ok) ({time.time() - t0:.1f}s)")
        return result
    except Exception as e:
        print(f" ✗ ({type(e).__name__}: {e})")
        return {"error": str(e), "type": type(e).__name__}


def _run_models_for_region(region_data: RegionData, config: dict) -> Dict[str, Any]:
    """Fit every enabled model on this region."""
    instances = build_models(config, region=region_data.name)
    results = {}
    for model_name, model in instances.items():
        try:
            print(f" – {model_name}...", end="", flush=True)
            t0 = time.time()
            if model.kind == "volatility":
                model.fit(region_data.y)
                fr = model.forecast(region=region_data.name)
            else:
                model.fit(region_data.y, region_data.X)
                fr = model.forecast(region=region_data.name)
            elapsed = time.time() - t0
            results[model_name] = fr
            if getattr(fr, "error", None):
                print(f" ✗ ({fr.error})")
            else:
                print(f" (ok) ({elapsed:.1f}s)")
        except ImportError as e:
            print(f" ✗ Missing package: {e}. Run: pip install -r requirements.txt")
            results[model_name] = {"error": f"Missing package: {e}"}
        except Exception as e:
            print(f" ✗ ({type(e).__name__}: {e})")
            results[model_name] = {"error": f"{type(e).__name__}: {e}"}
    return results


def _analyse_region(region_data: RegionData, config: dict) -> Dict[str, Any]:
    """Run the full analytical pipeline for one region."""
    print(f"\n→ Region: {region_data.name.upper()} "
          f"({region_data.n_obs} obs, {len(region_data.drivers)} drivers, "
          f"{region_data.currency})")

    out = {"meta": region_data.summary()}

    # Expose the full region dataframe + liquidity column list to downstream
    # consumers (specifically the HTML report builder, which needs to render
    # the India Liquidity section). These keys are deliberately prefixed
    # with an underscore to mark them as "internal pass-through" rather
    # than analytical results — they should NOT be JSON-serialised verbatim.
    out["_region_df"] = region_data.df
    out["_liquidity_cols"] = list(region_data.liquidity_cols)

    df = pd.concat([region_data.y, region_data.X], axis=1)

    out["summary_stats"] = _step("summary stats", summary_stats, df)
    out["adf"] = _step("ADF stationarity", adf_table, df,
                                    config["analysis"]["stationarity"]["significance"])
    out["vif"] = _step("VIF multicollinearity", vif_table, region_data.X)
    out["correlation"] = _step("correlation matrix", correlation_matrix, df)
    out["lead_lag"] = _step("lead-lag (CCF + Granger)",
                                    lead_lag_summary, region_data.y, region_data.X,
                                    config["analysis"]["lead_lag"]["max_lag_months"],
                                    config["analysis"]["lead_lag"]["granger_significance"])
    out["lag_matrix"] = _step("lag matrix (heatmap source)",
                                    lag_matrix, region_data.y, region_data.X,
                                    config["analysis"]["lead_lag"]["max_lag_months"])
    out["rolling_corr"] = _step("rolling correlations",
                                    rolling_correlations, region_data.y, region_data.X,
                                    config["analysis"]["lead_lag"]["rolling_window"])
    out["spread"] = _step("spread analysis",
                                    analyse_spread, region_data)

    # Cyclicity — the behavioural regime engine. CRITICAL: this must be called
    # with KEYWORD arguments. The engine signature is
    #   analyse_cyclicity(target, region, currency, drivers, n_regimes, ...)
    # so `drivers` is the 4th positional slot. The previous code passed
    # n_regimes positionally into that slot, meaning the drivers never
    # reached the engine and every region silently ran the price-only 'lite'
    # five-feature version — which is why the HTML report's regimes did not
    # match the dashboard (the dashboard passes drivers correctly). Passing
    # drivers here is what makes the report and dashboard share one engine.
    cyc_cfg = config["analysis"].get("cyclicity", {})
    _drivers = region_data.X if len(region_data.drivers) > 0 else None
    out["cyclicity"] = _step("cyclicity (GMM + spectral + Markov)",
                                    analyse_cyclicity, region_data.y,
                                    region=region_data.name,
                                    currency=region_data.currency,
                                    drivers=_drivers,
                                    n_regimes=cyc_cfg.get("n_regimes", 4),
                                    random_state=cyc_cfg.get("random_state", 42),
                                    prominence_pct=cyc_cfg.get("peak_prominence_pct", 5.0),
                                    min_distance_months=cyc_cfg.get("min_distance_months", 4))
    out["attribution"] = _step("driver attribution",
                                    rolling_attribution, region_data.y, region_data.X,
                                    config["analysis"]["attribution"]["rolling_window"],
                                    region_data.name)
    out["events"] = _step("event windows",
                                    run_event_analysis, region_data.y, region_data.X,
                                    config["analysis"]["events"]["episodes"],
                                    config["analysis"]["events"]["windows_months"],
                                    region_data.name)

    print(f" • Models:")
    out["models"] = _run_models_for_region(region_data, config)

    return out


def run_pipeline(config: dict) -> Dict[str, Any]:
    """The main entry point. Returns a complete results dict."""
    print("=" * 70)
    print("HRC STEEL PIPELINE")
    print("=" * 70)
    t_total = time.time()

    print("\n[1/3] Loading data")
    dataset = load_data(config)
    for name, region in dataset.regions.items():
        print(f" (ok) {name}: {region.n_obs} obs, {len(region.drivers)} drivers")
        if region.skipped_columns:
            print(f" skipped (non-numeric): {region.skipped_columns}")

    print(f"\n[2/3] Available models: {list_available()}")

    print(f"\n[3/3] Running analyses + models")
    results = {
        "config": config,
        "dataset_meta": dataset.summary(),
        "regions": {},
    }
    for name, region in dataset.regions.items():
        results["regions"][name] = _analyse_region(region, config)

    # Event Deep-Dive — Phase 4. Runs AFTER per-region analysis because it is
    # cross-regional: each region's deep-dive uses every region's cyclicity
    # result for the 'where did it change' panel. We produce one deep-dive
    # per config episode, per region, so the HTML report can render the same
    # curated-context + driver-tracing content the dashboard shows.
    print("\n→ Event Deep-Dive analysis")
    try:
        from .covid_filter import covid_window_from_config
        _ctx = load_event_context()
        _episodes = config["analysis"]["events"]["episodes"]
        _covid = covid_window_from_config(config)
        # Assemble cross-region inputs (price + cyclicity per region)
        _all_regions = {}
        for rname, rres in results["regions"].items():
            rcyc = rres.get("cyclicity")
            rdata = dataset.regions.get(rname)
            if rdata is not None:
                _all_regions[rname] = {
                    "y": rdata.y, "currency": rdata.currency,
                    "cyc": rcyc if not isinstance(rcyc, dict) else None,
                }
        for rname, rres in results["regions"].items():
            rdata = dataset.regions.get(rname)
            rcyc = rres.get("cyclicity")
            if rdata is None or isinstance(rcyc, dict):
                rres["event_deep_dive"] = {"available": False,
                                            "reason": "No cyclicity result."}
                continue
            _drv = rdata.X if len(rdata.drivers) > 0 else None
            cands = build_event_candidates(rcyc, _episodes, rdata.y)
            # Only deep-dive the named config episodes for the report (the
            # detected turns are an interactive-dashboard feature).
            cfg_cands = [c for c in cands if c.source == "config"]
            deep = []
            for cand in cfg_cands:
                try:
                    edd = analyse_event_deep_dive(
                        rdata.y, _drv, rcyc, cand,
                        region=rname, currency=rdata.currency,
                        all_regions=_all_regions, context=_ctx,
                        covid=_covid)
                    deep.append(edd)
                except Exception as e:
                    print(f"   ✗ {cand.label}: {type(e).__name__}: {e}")
            rres["event_deep_dive"] = deep
            print(f" • {rname}: {len(deep)} event deep-dives (ok)")
    except Exception as e:
        print(f" ✗ Event Deep-Dive step failed: {type(e).__name__}: {e}")

    # Cross-region (only if both China + India spread available)
    print("\n→ Cross-region analysis")
    china_spread = results["regions"].get("china", {}).get("spread")
    india_spread = results["regions"].get("india", {}).get("spread")
    if china_spread and india_spread and not isinstance(china_spread, dict) \
                                       and not isinstance(india_spread, dict):
        results["cross_region"] = _step("China vs India comparison",
                                         cross_region_comparison,
                                         china_spread, india_spread)
    else:
        results["cross_region"] = {"available": False,
                                    "reason": "Need both regions for comparison"}

    # Macro calendar (uses both regions; degrades to china-only if india missing)
    print("\n→ Macro calendar analysis")
    india_data = dataset.regions.get("india")
    china_data = dataset.regions.get("china")
    if china_data is not None:
        results["macro_calendar"] = _step(
            "macro events with historical analogues",
            analyse_macro_calendar, china_data, india_data, config,
        )
    else:
        results["macro_calendar"] = {"error": "China region required for macro calendar"}

    print(f"\n(ok) Pipeline complete in {time.time() - t_total:.1f}s")
    return results


def save_results_json(results: dict, path: str) -> None:
    """Save a JSON-friendly version of results for the dashboard to consume."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return {"_type": "DataFrame",
                    "data": obj.reset_index().to_dict(orient="list")}
        if isinstance(obj, pd.Series):
            return {"_type": "Series", "name": obj.name,
                    "index": [str(i) for i in obj.index],
                    "values": [None if pd.isna(v) else
                               float(v) if isinstance(v, (int, float, np.number))
                               else str(v) for v in obj.values]}
        if isinstance(obj, (np.integer, np.floating)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj.date())
        if isinstance(obj, dict):
            # Skip underscore-prefixed pass-through keys (e.g. "_region_df",
            # "_liquidity_cols"). These are internal handoffs to the report
            # builder, not analytical results meant for JSON consumers.
            return {k: make_serializable(v) for k, v in obj.items()
                    if not (isinstance(k, str) and k.startswith("_"))}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return make_serializable(obj.__dict__)
        return obj

    serializable = make_serializable(results)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f" (ok) Results saved: {path}")
