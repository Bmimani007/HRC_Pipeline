"""
Regime classification — K-means clustering on standardised target + drivers.

Identifies distinct market states (e.g., 'supercycle peak', 'correction',
'recovery') based on the joint behaviour of price + drivers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeResult:
    region: str
    n_regimes: int
    labels: pd.Series                          # regime label per month
    centers: pd.DataFrame                      # cluster centres in original units
    regime_means: pd.DataFrame                 # mean of each variable per regime
    transition_dates: pd.DataFrame             # when regimes changed
    current_regime: int
    regime_stats: pd.DataFrame                 # duration, frequency per regime


def classify_regimes(y: pd.Series, X: pd.DataFrame,
                     n_regimes: int = 3, region: str = "",
                     random_state: int = 42) -> RegimeResult:
    """K-means on standardised [target, drivers]."""
    df = pd.concat([y.rename("target"), X], axis=1).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df.values)

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    # Order regimes by mean target (regime 0 = lowest target, regime n-1 = highest)
    label_target_means = pd.DataFrame({
        "label": labels, "target": df["target"].values
    }).groupby("label")["target"].mean().sort_values()
    relabel_map = {old: new for new, old in enumerate(label_target_means.index)}
    new_labels = pd.Series([relabel_map[l] for l in labels], index=df.index, name="regime")

    # Cluster centres in original units
    centers_scaled = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_orig, columns=df.columns)
    centers_df.index = [relabel_map[i] for i in range(n_regimes)]
    centers_df = centers_df.sort_index()

    # Mean of each variable per regime
    df_with_label = df.copy()
    df_with_label["regime"] = new_labels
    regime_means = df_with_label.groupby("regime").mean()

    # Transitions
    changes = new_labels.ne(new_labels.shift())
    trans_dates = new_labels[changes].reset_index()
    trans_dates.columns = ["date", "new_regime"]

    # Stats per regime
    stats_rows = []
    for r in range(n_regimes):
        mask = new_labels == r
        stats_rows.append({
            "regime": r,
            "n_months": int(mask.sum()),
            "frequency_pct": float(mask.mean() * 100),
            "first_seen": str(new_labels[mask].index.min().date()) if mask.sum() else None,
            "last_seen": str(new_labels[mask].index.max().date()) if mask.sum() else None,
            "avg_target": float(df.loc[mask, "target"].mean()),
        })
    regime_stats = pd.DataFrame(stats_rows)

    return RegimeResult(
        region=region,
        n_regimes=n_regimes,
        labels=new_labels,
        centers=centers_df,
        regime_means=regime_means,
        transition_dates=trans_dates,
        current_regime=int(new_labels.iloc[-1]),
        regime_stats=regime_stats,
    )
