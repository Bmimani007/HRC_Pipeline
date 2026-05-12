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
from .lead_lag import lead_lag_summary, rolling_correlations, cross_correlation
from .spread import analyse_region as analyse_spread, cross_region_comparison
from .regimes import classify_regimes
from .attribution import rolling_attribution
from .events import run_event_analysis
from .cyclicity import analyse_cyclicity
from .macro_calendar import analyse_macro_calendar
from models.registry import build_models, list_available


def _step(label, fn, *args, **kwargs):
    """Run a pipeline step with timing + friendly logging."""
    print(f"  • {label}...", end="", flush=True)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        print(f" ✓ ({time.time() - t0:.1f}s)")
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
            print(f"    – {model_name}...", end="", flush=True)
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
                print(f" ✓ ({elapsed:.1f}s)")
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

    out["summary_stats"]   = _step("summary stats",     summary_stats, df)
    out["adf"]             = _step("ADF stationarity",  adf_table, df,
                                    config["analysis"]["stationarity"]["significance"])
    out["vif"]             = _step("VIF multicollinearity", vif_table, region_data.X)
    out["correlation"]     = _step("correlation matrix", correlation_matrix, df)
    out["lead_lag"]        = _step("lead-lag (CCF + Granger)",
                                    lead_lag_summary, region_data.y, region_data.X,
                                    config["analysis"]["lead_lag"]["max_lag_months"],
                                    config["analysis"]["lead_lag"]["granger_significance"])
    out["rolling_corr"]    = _step("rolling correlations",
                                    rolling_correlations, region_data.y, region_data.X,
                                    config["analysis"]["lead_lag"]["rolling_window"])
    out["spread"]          = _step("spread analysis",
                                    analyse_spread, region_data)
    out["regimes"]         = _step("regime classification",
                                    classify_regimes, region_data.y, region_data.X,
                                    config["analysis"]["regimes"]["n_regimes"],
                                    region_data.name,
                                    config["analysis"]["regimes"]["random_state"])
    cyc_cfg = config["analysis"].get("cyclicity", {})
    out["cyclicity"]       = _step("cyclicity (GMM + spectral + Markov)",
                                    analyse_cyclicity, region_data.y,
                                    region_data.name,
                                    region_data.currency,
                                    cyc_cfg.get("n_regimes", 4),
                                    cyc_cfg.get("random_state", 42),
                                    cyc_cfg.get("peak_prominence_pct", 5.0),
                                    cyc_cfg.get("min_distance_months", 4))
    out["attribution"]     = _step("driver attribution",
                                    rolling_attribution, region_data.y, region_data.X,
                                    config["analysis"]["attribution"]["rolling_window"],
                                    region_data.name)
    out["events"]          = _step("event windows",
                                    run_event_analysis, region_data.y, region_data.X,
                                    config["analysis"]["events"]["episodes"],
                                    config["analysis"]["events"]["windows_months"],
                                    region_data.name)

    print(f"  • Models:")
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
        print(f"  ✓ {name}: {region.n_obs} obs, {len(region.drivers)} drivers")
        if region.skipped_columns:
            print(f"    skipped (non-numeric): {region.skipped_columns}")

    print(f"\n[2/3] Available models: {list_available()}")

    print(f"\n[3/3] Running analyses + models")
    results = {
        "config": config,
        "dataset_meta": dataset.summary(),
        "regions": {},
    }
    for name, region in dataset.regions.items():
        results["regions"][name] = _analyse_region(region, config)

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

    print(f"\n✓ Pipeline complete in {time.time() - t_total:.1f}s")
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
    print(f"  ✓ Results saved: {path}")
