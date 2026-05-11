"""
Data loader — reads the multi-sheet Raw_data.xlsx and auto-detects columns.

Key feature: when `drivers: auto` is set in config, ANY numeric column
that isn't the date or the target is treated as a driver. Add or remove
columns in the xlsx — the pipeline adapts automatically. No code changes.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class RegionData:
    """All data + metadata for one region (China, India, US, etc.)."""
    name: str                                    # 'china', 'india', 'us'
    df: pd.DataFrame                             # date-indexed dataframe
    target: str                                  # name of the target column
    drivers: List[str]                           # driver column names
    currency: str                                # 'USD' or 'INR'
    spread_config: Dict                          # iron_ore + hcc cols & weights
    skipped_columns: List[str] = field(default_factory=list)
    overview_only: bool = False                  # True = only Overview tab in dashboard
    liquidity_cols: List[str] = field(default_factory=list)  # liquidity columns (base + derived), present only if region has liquidity block

    @property
    def y(self) -> pd.Series:
        return self.df[self.target]

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.drivers]

    @property
    def n_obs(self) -> int:
        return len(self.df)

    def summary(self) -> dict:
        return {
            "region": self.name,
            "currency": self.currency,
            "n_observations": self.n_obs,
            "n_drivers": len(self.drivers),
            "target": self.target,
            "drivers": list(self.drivers),
            "skipped_columns": self.skipped_columns,
            "date_start": str(self.df.index.min().date()),
            "date_end": str(self.df.index.max().date()),
            "missing_values": int(self.df.isna().sum().sum()),
            "overview_only": self.overview_only,
            "liquidity_cols": list(self.liquidity_cols),
        }


@dataclass
class Dataset:
    """Container for all loaded regions + file metadata."""
    regions: Dict[str, RegionData]
    source_path: str
    file_hash: str

    def __getitem__(self, region_name: str) -> RegionData:
        return self.regions[region_name]

    def __contains__(self, region_name: str) -> bool:
        return region_name in self.regions

    def summary(self) -> dict:
        return {
            "source": self.source_path,
            "file_hash": self.file_hash[:12],
            "regions": {name: r.summary() for name, r in self.regions.items()},
        }


def _file_hash(path: str) -> str:
    """MD5 of file contents — used by the cache to detect data changes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _friendly_error(msg: str, available: list = None) -> ValueError:
    """Plain-English error messages instead of raw stack traces."""
    if available:
        msg += f"\n\n  Available options: {available}"
    msg += "\n\n  → Edit config.yaml to fix this, then run again."
    return ValueError(msg)


def _load_one_region(name: str, region_cfg: dict, file_path: str) -> RegionData:
    """Load one region (one sheet) from the xlsx."""
    sheet = region_cfg["sheet"]

    # Read the sheet
    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
    except ValueError as e:
        # Sheet doesn't exist
        all_sheets = pd.ExcelFile(file_path).sheet_names
        raise _friendly_error(
            f"Sheet '{sheet}' (region '{name}') not found in {file_path}.",
            available=all_sheets,
        )

    # Date column
    date_col = region_cfg["date_column"]
    if date_col not in df.columns:
        raise _friendly_error(
            f"Date column '{date_col}' not found in sheet '{sheet}'.",
            available=list(df.columns),
        )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Target
    target = region_cfg["target"]
    if target not in df.columns:
        raise _friendly_error(
            f"Target '{target}' not found in sheet '{sheet}'.",
            available=list(df.columns),
        )

    # --- LIQUIDITY MODULE -----------------------------------------------------
    # If a `liquidity:` block exists for this region, compute derived series
    # (WACR_Spread, Stress_Index, regime labels, etc.) and stamp them onto the
    # dataframe as additional columns. They can then be selected as drivers via
    # the standard `drivers:` config or referenced by the dashboard's Liquidity
    # tab. Derived columns are skipped silently if the required raw inputs are
    # missing — non-liquidity regions are unaffected.
    liquidity_cols: List[str] = []
    liq_cfg = region_cfg.get("liquidity")
    if liq_cfg:
        try:
            from pipeline.liquidity import (
                compute_derived_series,
                classify_liquidity_regime,
                compute_stress_index,
                detect_policy_regime,
            )
        except Exception as exc:
            raise _friendly_error(
                f"Failed to import pipeline.liquidity module: {exc}"
            )

        base_cols = liq_cfg.get("base_columns", [])
        present = [c for c in base_cols if c in df.columns]
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise _friendly_error(
                f"Liquidity base columns not in sheet '{sheet}': {missing}",
                available=list(df.columns),
            )

        df = compute_derived_series(df)
        if "WACR_Spread" in df.columns:
            df["Liquidity_Regime"] = classify_liquidity_regime(
                df["WACR_Spread"],
                method=liq_cfg.get("regime_method", "std_band"),
                band_width=float(liq_cfg.get("regime_band_width", 0.5)),
            )
            df["Stress_Index"] = compute_stress_index(df)
        if "Repo_Rate" in df.columns:
            df["Policy_Regime"] = detect_policy_regime(
                df["Repo_Rate"],
                lookback_months=int(liq_cfg.get("policy_lookback_months", 6)),
            )

        # All columns that are part of the liquidity universe
        derived = ["WACR_Spread", "GSec_Repo_Spread", "Bank_Credit_YoY",
                   "Repo_6M_Change", "Stress_Index", "Liquidity_Regime",
                   "Policy_Regime"]
        liquidity_cols = present + [c for c in derived if c in df.columns]

    # Drivers
    driver_cfg = region_cfg.get("drivers", "auto")
    skipped = []
    if driver_cfg == "auto":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drivers = [c for c in numeric_cols if c != target]
        # Identify non-numeric columns we skipped (e.g., notes columns)
        skipped = [c for c in df.columns if c != target and c not in drivers]

        # If a liquidity block is configured, EXCLUDE all liquidity-related
        # columns from the auto driver list and re-add only those nominated in
        # liquidity.driver_columns. This avoids the multicollinearity that
        # would otherwise arise from including WACR + Repo_Rate + WACR_Spread
        # (which is just WACR − Repo_Rate) all at once.
        if liq_cfg:
            liq_universe = set(liquidity_cols)
            drivers = [c for c in drivers if c not in liq_universe]
            nominated = liq_cfg.get("driver_columns", [])
            for c in nominated:
                if c not in df.columns:
                    raise _friendly_error(
                        f"Liquidity driver_column '{c}' not present after "
                        f"derived-series computation for region '{name}'.",
                        available=list(df.columns),
                    )
                if c not in drivers:
                    drivers.append(c)
    elif isinstance(driver_cfg, list):
        missing = [c for c in driver_cfg if c not in df.columns]
        if missing:
            raise _friendly_error(
                f"Drivers not found in sheet '{sheet}': {missing}",
                available=list(df.columns),
            )
        drivers = driver_cfg
    else:
        raise _friendly_error(
            f"data.regions.{name}.drivers must be 'auto' or a list, got: {driver_cfg}"
        )

    overview_only = region_cfg.get("overview_only", False)

    if len(drivers) == 0 and not overview_only:
        raise _friendly_error(
            f"No usable drivers in sheet '{sheet}'. "
            f"Add at least one numeric column besides the target, "
            f"or set 'overview_only: true' to enable overview-only mode."
        )

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    if len(df) < 24:
        raise _friendly_error(
            f"Only {len(df)} observations in sheet '{sheet}' — need at least 24 "
            f"for meaningful time-series modelling."
        )

    return RegionData(
        name=name,
        df=df,
        target=target,
        drivers=drivers,
        currency=region_cfg.get("currency", "USD"),
        spread_config=region_cfg.get("spread", {}),
        skipped_columns=skipped,
        overview_only=overview_only,
        liquidity_cols=liquidity_cols,
    )


def load_data(config: dict) -> Dataset:
    """Load all enabled regions from the configured xlsx file."""
    file_path = config["data"]["file"]

    # Path resilience: try the path as given (relative to CWD), then relative
    # to the project root (parent of pipeline/). Streamlit Cloud may run from
    # an unexpected working directory.
    candidates = [Path(file_path)]
    project_root = Path(__file__).parent.parent.resolve()
    candidates.append(project_root / file_path)

    found_path = None
    for p in candidates:
        if p.exists():
            found_path = p
            break

    if found_path is None:
        raise _friendly_error(
            f"Data file not found. Tried:\n" +
            "\n".join(f"  • {p}" for p in candidates) +
            f"\n\nCheck 'data.file' in config.yaml, or place the file at one of these paths."
        )
    file_path = str(found_path)

    regions_cfg = config["data"]["regions"]
    regions = {}
    for name, rcfg in regions_cfg.items():
        if not rcfg.get("enabled", True):
            continue
        regions[name] = _load_one_region(name, rcfg, file_path)

    if not regions:
        raise _friendly_error("No regions enabled. Enable at least one in config.yaml.")

    return Dataset(
        regions=regions,
        source_path=str(file_path),
        file_hash=_file_hash(file_path),
    )


if __name__ == "__main__":
    # Sanity check: python3 -m pipeline.data_loader
    import yaml, json
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ds = load_data(cfg)
    print(json.dumps(ds.summary(), indent=2, default=str))
