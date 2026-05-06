"""
Event window analysis — examines how target + drivers behaved before and
after specific event dates (e.g., supercycle peak, Ukraine war).

For each event:
    • Pre-event window (e.g., 6 months before)
    • Post-event window (e.g., 6 months after)
    • Compute mean, max, min, % change for each variable
    • Identify which drivers moved most around the event
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd


@dataclass
class EventWindowResult:
    region: str
    episodes: List[Dict]                # one dict per event with full stats


def analyse_event(y: pd.Series, X: pd.DataFrame,
                   event_date: pd.Timestamp,
                   window_months: List[int] = [3, 6]) -> dict:
    """For one event, compute pre/post windows of various sizes."""
    df = pd.concat([y.rename("target"), X], axis=1)
    out = {"event_date": str(event_date.date()), "windows": {}}

    for w in window_months:
        pre_start = event_date - pd.DateOffset(months=w)
        post_end = event_date + pd.DateOffset(months=w)
        pre = df.loc[pre_start:event_date - pd.DateOffset(days=1)]
        post = df.loc[event_date:post_end]

        var_stats = {}
        for col in df.columns:
            pre_vals = pre[col].dropna()
            post_vals = post[col].dropna()
            if len(pre_vals) == 0 or len(post_vals) == 0:
                continue
            pre_avg = float(pre_vals.mean())
            post_avg = float(post_vals.mean())
            pct_change = ((post_avg - pre_avg) / pre_avg * 100) \
                          if pre_avg != 0 else float("nan")
            var_stats[col] = {
                "pre_avg": pre_avg,
                "post_avg": post_avg,
                "pct_change": float(pct_change),
                "pre_n": int(len(pre_vals)),
                "post_n": int(len(post_vals)),
            }
        out["windows"][f"{w}m"] = var_stats
    return out


def run_event_analysis(y: pd.Series, X: pd.DataFrame,
                        episodes_cfg: List[dict],
                        window_months: List[int] = [3, 6],
                        region: str = "") -> EventWindowResult:
    """Run the full event analysis from config-defined episodes."""
    episodes = []
    for ep in episodes_cfg:
        d = pd.to_datetime(ep["date"])
        result = analyse_event(y, X, d, window_months=window_months)
        result["name"] = ep["name"]
        result["type"] = ep.get("type", "event")
        # Only include if we have data around the event
        first_w = result["windows"].get(f"{window_months[0]}m", {})
        if first_w:
            episodes.append(result)
    return EventWindowResult(region=region, episodes=episodes)
