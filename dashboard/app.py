"""
HRC Steel Pipeline — Live Dashboard

Run with: streamlit run dashboard/app.py

This dashboard:
    • Reads the same xlsx + config the pipeline reads
    • Re-runs the pipeline automatically when the xlsx file changes (cached by file hash)
    • Lets you toggle drivers and date ranges on the fly
    • Shows everything from the report, but interactive
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
from datetime import date

# Make project root importable
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline.data_loader import load_data
from pipeline.diagnostics import adf_table, vif_table, correlation_matrix
from pipeline.lead_lag import lead_lag_summary, rolling_correlations
from pipeline import narrator
from pipeline.spread import analyse_region as analyse_spread, cross_region_comparison
from pipeline.regimes import classify_regimes
from pipeline.attribution import rolling_attribution


# ---------- Page setup ----------
st.set_page_config(
    page_title="HRC Steel Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle additive styling — relies on Streamlit's light theme (forced via
# .streamlit/config.toml) for colors, only tweaks spacing and a few details.
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    h1 { font-weight: 700; font-size: 1.6rem; margin-bottom: 0.5rem; }
    h2 { font-weight: 600; font-size: 1.15rem; margin-top: 1.2rem; }
    h3 { font-weight: 600; font-size: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.78rem; }
    section[data-testid="stSidebar"] { border-right: 1px solid #E5E9F0; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    button[kind="secondary"] { width: 100%; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 18px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ---------- Password gate (active only when a password is configured) ----------
def _check_password() -> bool:
    """
    Returns True if the user has authenticated, False otherwise.

    Reads the password from st.secrets["app_password"] when deployed on
    Streamlit Cloud. When running locally without secrets configured, the gate
    is bypassed (no password prompt) so local development isn't disrupted.

    To set the password:
      • Local: create .streamlit/secrets.toml with: app_password = "your-pw"
      • Cloud: add app_password under "Secrets" in the Streamlit Cloud
                app settings → no code changes needed
    """
    # If no password is configured, skip the gate entirely (local dev mode)
    try:
        configured_pw = st.secrets["app_password"]
    except (KeyError, FileNotFoundError, Exception):
        return True

    if not configured_pw:
        return True

    # Already authenticated this session?
    if st.session_state.get("authenticated"):
        return True

    # Show the login form
    st.markdown("<div style='max-width: 380px; margin: 80px auto; "
                "padding: 32px; background: white; border-radius: 12px; "
                "border: 1px solid #E5E9F0;'>", unsafe_allow_html=True)
    st.markdown("### HRC Steel Intelligence")
    st.caption("Enter the access password to continue.")
    pw_input = st.text_input("Password", type="password", label_visibility="collapsed",
                              placeholder="Password")
    if st.button("Sign in", use_container_width=True, type="primary"):
        if pw_input == configured_pw:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown("</div>", unsafe_allow_html=True)
    return False


if not _check_password():
    st.stop()


# ---------- Cache: load + analyse only if file changed ----------
@st.cache_data(show_spinner=False)
def cached_load(config_str: str, file_mtime: float):
    """Cache key is config + file mtime. Edit the xlsx → cache invalidates."""
    cfg = yaml.safe_load(config_str)
    return load_data(cfg)


@st.cache_data(show_spinner="Computing spread analysis...")
def cached_spread(_region_data, cache_key: str, file_mtime: float):
    return analyse_spread(_region_data)


@st.cache_data(show_spinner="Computing diagnostics...")
def cached_diagnostics(_region_data, cache_key: str, file_mtime: float):
    df = pd.concat([_region_data.y, _region_data.X], axis=1)
    return {
        "adf": adf_table(df),
        "vif": vif_table(_region_data.X),
        "corr": correlation_matrix(df),
    }


@st.cache_data(show_spinner="Computing lead/lag...")
def cached_lead_lag(_region_data, cache_key: str, max_lag: int, file_mtime: float):
    return lead_lag_summary(_region_data.y, _region_data.X, max_lag=max_lag)


@st.cache_data(show_spinner="Classifying regimes...")
def cached_regimes(_region_data, cache_key: str, n_regimes: int, file_mtime: float):
    return classify_regimes(_region_data.y, _region_data.X, n_regimes=n_regimes,
                              region=_region_data.name)


@st.cache_data(show_spinner="Computing attribution...")
def cached_attribution(_region_data, cache_key: str, window: int, file_mtime: float):
    return rolling_attribution(_region_data.y, _region_data.X, window=window,
                                 region=_region_data.name)


@st.cache_data(show_spinner="Running cyclicity analysis...")
def cached_cyclicity(_region_data, cache_key: str, n_regimes: int, file_mtime: float):
    from pipeline.cyclicity import analyse_cyclicity
    return analyse_cyclicity(_region_data.y, region=_region_data.name,
                              currency=_region_data.currency,
                              n_regimes=n_regimes)


@st.cache_data(show_spinner="Loading macro calendar...")
def cached_macro_calendar(_dataset, config_yaml: str, window_days: int,
                            show_past: bool, past_days: int, file_mtime: float):
    from pipeline.macro_calendar import analyse_macro_calendar
    china = _dataset.regions.get("china")
    india = _dataset.regions.get("india")
    if china is None:
        return None
    return analyse_macro_calendar(china, india, yaml.safe_load(config_yaml),
                                    window_days=window_days,
                                    show_past=show_past, past_days=past_days)


# ---------- FORECASTING caches ----------
@st.cache_data(show_spinner="Fitting ARIMAX...")
def cached_arimax(_region, region_key: str, ar: int, d: int, ma: int,
                    horizon: int, drivers_tuple: tuple, file_mtime: float):
    """Cache key includes region_key + all params + drivers tuple + file mtime."""
    from pipeline.forecasting import fit_arimax
    drivers = list(drivers_tuple) if drivers_tuple else None
    return fit_arimax(_region.y, _region.X, ar=ar, d=d, ma=ma,
                        horizon=horizon, drivers=drivers)


@st.cache_data(show_spinner="Fitting ARDL...")
def cached_ardl(_region, region_key: str, ar: int, dl: int,
                  horizon: int, drivers_tuple: tuple, file_mtime: float):
    from pipeline.forecasting import fit_ardl
    drivers = list(drivers_tuple) if drivers_tuple else None
    return fit_ardl(_region.y, _region.X, ar=ar, dl=dl,
                     horizon=horizon, drivers=drivers)


@st.cache_data(show_spinner="Running out-of-sample test...")
def cached_oos_test(_region, region_key: str, model_type: str,
                     ar: int, d: int, ma: int, dl: int,
                     test_size: int, drivers_tuple: tuple, file_mtime: float):
    """
    Single train/test split OOS evaluation — uses the SAME ARIMAXModel /
    ARDLModel classes the report uses, so dashboard and report numbers
    agree exactly when the orders match.

    Holds out the last `test_size` months. Fits the chosen model on data
    BEFORE that window, forecasts forward, then compares predictions to
    the held-out actuals using the actual driver values for that window
    (not held-constant), which mirrors what report/builder.py renders
    under "Out-of-sample test metrics".
    """
    from types import SimpleNamespace
    drivers = list(drivers_tuple) if drivers_tuple else None

    y = _region.y
    X = _region.X
    if drivers and X is not None and len(X) > 0:
        usable = [c for c in drivers if c in X.columns]
        X = X[usable] if usable else X

    if len(y) <= test_size + 12:
        return SimpleNamespace(
            success=False,
            error_msg=f"Test size {test_size} too large — model needs at least "
                       f"{test_size + 12} obs, region has {len(y)}.",
        )

    try:
        if model_type.lower() == "arimax":
            from models.arimax import ARIMAXModel
            cfg = {"order": [ar, d, ma],
                   "seasonal_order": [0, 0, 0, 0],
                   "forecast_horizon": 12,
                   "test_size": test_size}
            model = ARIMAXModel(cfg)
            config_summary = f"ARIMAX({ar},{d},{ma})"
        else:
            from models.ardl import ARDLModel
            cfg = {"ar_order": ar, "dl_order": dl,
                   "use_order_selection": False,
                   "forecast_horizon": 12,
                   "test_size": test_size}
            model = ARDLModel(cfg)
            config_summary = f"ARDL(ar={ar}, dl={dl})"

        model.fit(y, X)
        result = model.forecast(steps=12, region=_region.name)
    except Exception as e:
        return SimpleNamespace(
            success=False,
            error_msg=f"Model fit failed: {type(e).__name__}: {str(e)[:200]}",
        )

    if result.error:
        return SimpleNamespace(success=False, error_msg=result.error)

    if result.oos_predictions is None or result.oos_actuals is None:
        return SimpleNamespace(
            success=False,
            error_msg=(f"OOS test could not run — typically because the training "
                        f"window after holding out {test_size} months is too short. "
                        f"Try a smaller test window."),
        )

    actual = result.oos_actuals
    pred = result.oos_predictions
    metrics = result.metrics or {}

    # Hit rate — directional accuracy. Anchor first comparison at the last
    # in-sample value so each test month produces one pred-vs-actual direction
    # comparison.
    last_train_val = float(y.iloc[-(test_size + 1)])
    pred_full = np.concatenate([[last_train_val], pred.values])
    actual_full = np.concatenate([[last_train_val], actual.values])
    pred_dir = np.sign(np.diff(pred_full))
    actual_dir = np.sign(np.diff(actual_full))
    matches = (pred_dir == actual_dir) | (pred_dir == 0) | (actual_dir == 0)
    hit_rate = float(np.mean(matches) * 100) if len(matches) > 0 else float("nan")

    # Ljung-Box on training residuals (transparency about training fit)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        if result.residuals is not None and len(result.residuals.dropna()) > 12:
            lb = acorr_ljungbox(result.residuals.dropna(), lags=[10],
                                  return_df=True)
            ljung_p = float(lb["lb_pvalue"].iloc[0])
        else:
            ljung_p = None
    except Exception:
        ljung_p = None

    def _fmt_date(d):
        return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

    return SimpleNamespace(
        success=True, error_msg=None,
        model_type=model_type.upper(),
        config_summary=config_summary,
        # Series for charting
        fitted_in_sample=result.fitted,
        oos_predictions=pred,
        oos_actuals=actual,
        # Model classes don't compute CI for the OOS window — match the report,
        # which also doesn't show CI on its OOS overlay
        oos_lower=None, oos_upper=None,
        # Window meta
        n_train=len(y) - test_size,
        n_test=len(actual),
        train_start=_fmt_date(y.index[0]),
        train_end=_fmt_date(y.index[-(test_size + 1)]),
        test_start=_fmt_date(actual.index[0]),
        test_end=_fmt_date(actual.index[-1]),
        # Metrics — straight from the model class (matches report exactly)
        rmse=metrics.get("rmse", float("nan")),
        mae=metrics.get("mae", float("nan")),
        mape=metrics.get("mape", float("nan")),
        r2=metrics.get("r2", float("nan")),
        hit_rate=hit_rate,
        # Coefficients / fit details for transparency
        coefficients=result.coefficients,
        ljung_box_p=ljung_p,
        diagnostics=result.diagnostics or {},
    )


@st.cache_data(show_spinner="Fitting GARCH + risk metrics...")
def cached_garch(_region, region_key: str, horizon: int, file_mtime: float):
    from pipeline.forecasting import fit_garch_with_risk
    return fit_garch_with_risk(_region.y, horizon=horizon)


@st.cache_data(show_spinner="Running scenario...")
def cached_scenario(_region, region_key: str, model_type: str,
                      ar: int, d: int, ma: int, dl: int,
                      horizon: int, drivers_tuple: tuple,
                      shocks_tuple: tuple, file_mtime: float):
    """Wrapper around run_arimax_scenario / run_ardl_scenario.
    shocks_tuple is a sorted tuple of (driver_name, pct) for cache keying."""
    from pipeline.forecasting import run_arimax_scenario, run_ardl_scenario
    drivers = list(drivers_tuple) if drivers_tuple else None
    shocks = dict(shocks_tuple) if shocks_tuple else {}
    if model_type == "arimax":
        return run_arimax_scenario(_region.y, _region.X, shocks=shocks,
                                      ar=ar, d=d, ma=ma, horizon=horizon, drivers=drivers)
    else:
        return run_ardl_scenario(_region.y, _region.X, shocks=shocks,
                                    ar=ar, dl=dl, horizon=horizon, drivers=drivers)


# ---------- Render helper: convert narrator blocks to Streamlit HTML ----------
def render_interpretation(blocks, label: str = "Interpretation",
                            settings_summary: str = None):
    """
    Render narrator output as a collapsible expander. Default collapsed.
    blocks: list of strings (paragraphs) and dicts ({'_kind': 'key_interpretation'} etc.)
    settings_summary: optional one-line summary of current control values.
    """
    with st.expander(label, expanded=False):
        if settings_summary:
            st.caption(f"{settings_summary}")
        for b in blocks:
            if isinstance(b, dict) and b.get("_kind") == "key_interpretation":
                st.markdown(
                    f'<div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);'
                    f' border-left: 4px solid #1F4E79; padding: 12px 18px; margin: 12px 0;'
                    f' border-radius: 6px;">'
                    f'<div style="font-size: 0.7rem; font-weight: 700; color: #1F4E79;'
                    f' text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">'
                    f'{b["title"]}</div>'
                    f'<div style="font-size: 0.92rem; color: #1A1F2E; line-height: 1.55;">'
                    f'{b["body"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif isinstance(b, str):
                st.markdown(
                    f'<p style="font-size: 0.93rem; line-height: 1.65; color: #2D3748;'
                    f' margin: 8px 0;">{b}</p>',
                    unsafe_allow_html=True,
                )
            # event_card blocks: not used in dashboard interpretations


# ---------- GLOSSARY system ----------
@st.cache_data
def load_glossary():
    """Load glossary YAML and return as dict keyed by lowercase term."""
    candidates = [
        Path("data/glossary.yaml"),
        Path(__file__).parent.parent / "data" / "glossary.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                g = yaml.safe_load(f)
            entries = g.get("entries", [])
            return {e["term"].lower(): e for e in entries}
    return {}


GLOSSARY_CATEGORIES = {
    "statistical": ("", "Statistical concepts"),
    "model": ("", "Models"),
    "hrc-domain": ("", "HRC domain"),
    "macro": ("", "Macro"),
    "dashboard": ("", "Dashboard-specific"),
}


def _render_glossary_entry(entry: dict, glossary: dict, key_prefix: str = ""):
    """Render a single glossary entry as expandable HTML."""
    term = entry["term"]
    full = entry.get("full_name", "")
    cat = entry.get("category", "")
    icon = GLOSSARY_CATEGORIES.get(cat, ("●", cat))[0]
    short = entry.get("short", "")
    detail = entry.get("detail", "").strip()
    see_also = entry.get("see_also", [])

    header = f"{icon} **{term}**"
    if full:
        header += f" — *{full}*"

    st.markdown(header)
    st.markdown(
        f'<p style="font-size: 0.92rem; color: #2D3748; margin: 4px 0 8px 0;">'
        f'<i>{short}</i></p>',
        unsafe_allow_html=True,
    )
    if detail:
        st.markdown(
            f'<div style="font-size: 0.92rem; color: #1A1F2E; line-height: 1.6;">'
            f'{detail}</div>',
            unsafe_allow_html=True,
        )
    if see_also:
        valid_refs = [t for t in see_also if t.lower() in glossary]
        if valid_refs:
            tags = " · ".join(f"`{t}`" for t in valid_refs)
            st.markdown(
                f'<p style="font-size: 0.82rem; color: #5C6B7F; margin-top: 8px;">'
                f'See also: {tags}</p>',
                unsafe_allow_html=True,
            )



# ---------- Color palette ----------
COLORS = {"accent": "#1F4E79", "accent2": "#2D6A4F", "warning": "#C9540F",
          "danger": "#A4161A", "ink": "#1A1F2E", "muted": "#5C6B7F",
          "rule": "#E5E9F0"}
CHART_PALETTE = ["#1F4E79", "#2D6A4F", "#C9540F", "#7B5EA7",
                 "#A4161A", "#0F8B8D", "#5C6B7F", "#B8860B"]

# Plotly layout — PLOT_BASE intentionally contains NO axis or margin keys, so
# downstream update_layout calls can freely add xaxis/yaxis/margin without
# colliding. Axis grid styling is applied via update_xaxes/update_yaxes helpers.
PLOT_BASE = dict(
    template="plotly_white",
    colorway=CHART_PALETTE,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif",
              color=COLORS["ink"], size=12),
)
DEFAULT_MARGIN = dict(l=50, r=20, t=50, b=40)
COMPACT_MARGIN = dict(l=40, r=10, t=40, b=30)


def style_axes(fig):
    """Apply consistent axis styling — call AFTER update_layout."""
    fig.update_xaxes(gridcolor=COLORS["rule"], linecolor=COLORS["rule"])
    fig.update_yaxes(gridcolor=COLORS["rule"], linecolor=COLORS["rule"])
    return fig


# ---------- Sidebar: config + global filters ----------
st.sidebar.markdown("### HRC Steel Intelligence")
st.sidebar.caption("Live analytical dashboard")

# Load config
config_path = PROJECT_ROOT / "config.yaml"
if not config_path.exists():
    st.error(f"config.yaml not found at {config_path}")
    st.stop()

config_str = config_path.read_text()
config = yaml.safe_load(config_str)
data_file = PROJECT_ROOT / config["data"]["file"]
if not data_file.exists():
    st.error(f"Data file not found: {data_file}")
    st.stop()

file_mtime = data_file.stat().st_mtime

# Load data
try:
    dataset = cached_load(config_str, file_mtime)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Region selector
region_names = list(dataset.regions.keys())
options = [r.title() for r in region_names]
if len(region_names) > 1:
    options.append("Cross-Region")
region_pick_label = st.sidebar.radio("Region", options)
# Map back to internal name
region_pick = region_pick_label.lower() if region_pick_label != "Cross-Region" else "Cross-Region"

st.sidebar.markdown("---")
st.sidebar.caption(f"File: `{data_file.name}` · {len(region_names)} regions")
st.sidebar.caption(f"Last modified: {pd.Timestamp(file_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption(f"Hash: `{dataset.file_hash[:10]}`")

if st.sidebar.button("Refresh data (clear cache)"):
    st.cache_data.clear()
    st.rerun()


# ---------- HTML report download ----------
@st.cache_data(show_spinner=False)
def _build_report_html(file_hash: str, config_str: str) -> bytes:
    """
    Build the full HTML report on demand and return the bytes for download.
    Cached on (file_hash, config_str) so the report is rebuilt only when the
    underlying data or config changes — subsequent downloads are instant.
    """
    from pipeline.orchestrator import run_pipeline
    from report.builder import build_report
    import tempfile, os

    cfg = yaml.safe_load(config_str)
    results = run_pipeline(cfg)

    # Write the report to a temp path, then read its bytes back
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                      encoding="utf-8") as tmp:
        tmp_path = tmp.name
    try:
        build_report(results, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


st.sidebar.markdown("---")
st.sidebar.markdown("**HTML Report**")
st.sidebar.caption(
    "Generates the full prose-heavy HTML report (all regions, all sections). "
    "First click takes 30-60s; subsequent downloads are instant."
)

# Two-step button: 'Generate' → then 'Download'. Keeps the heavy work explicit.
if st.sidebar.button("Generate report", use_container_width=True):
    try:
        with st.spinner("Building report — running full pipeline..."):
            html_bytes = _build_report_html(dataset.file_hash, config_str)
        st.session_state["report_bytes"] = html_bytes
        st.session_state["report_built_at"] = pd.Timestamp.now().strftime("%H:%M:%S")
        st.sidebar.success(f"Report ready ({len(html_bytes)/1024:.0f} KB)")
    except Exception as e:
        st.sidebar.error(f"Build failed: {type(e).__name__}: {e}")

if "report_bytes" in st.session_state:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    st.sidebar.download_button(
        label="Download HTML report",
        data=st.session_state["report_bytes"],
        file_name=f"HRC_Steel_Report_{timestamp}.html",
        mime="text/html",
        use_container_width=True,
    )
    st.sidebar.caption(f"Built at {st.session_state.get('report_built_at', '?')}")


# ---------- Sidebar Glossary Search ----------
st.sidebar.markdown("---")
st.sidebar.markdown("**Glossary search**")
_glossary = load_glossary()
if _glossary:
    search_query = st.sidebar.text_input(
        "Search any term",
        placeholder="e.g., VIF, ARIMAX, spread...",
        key="sidebar_glossary_search",
        label_visibility="collapsed",
    )
    if search_query:
        q = search_query.strip().lower()
        # Direct match
        matches = []
        if q in _glossary:
            matches.append(_glossary[q])
        # Partial matches in term or full_name or short
        for key, e in _glossary.items():
            if e in matches:
                continue
            if (q in key
                or q in e.get("full_name", "").lower()
                or q in e.get("short", "").lower()):
                matches.append(e)

        if matches:
            st.sidebar.caption(f"{len(matches)} match{'es' if len(matches) != 1 else ''}")
            for m in matches[:5]:
                with st.sidebar.expander(f"{m['term']}", expanded=(len(matches) == 1)):
                    if m.get("full_name"):
                        st.markdown(f"**{m['full_name']}**")
                    st.markdown(f"*{m.get('short', '')}*")
                    if m.get("detail"):
                        st.markdown(m["detail"].strip())
                    if m.get("see_also"):
                        valid = [t for t in m["see_also"] if t.lower() in _glossary]
                        if valid:
                            st.caption(f"See also: {' · '.join(valid)}")
            if len(matches) > 5:
                st.sidebar.caption(f"...and {len(matches) - 5} more — see Glossary tab.")
        else:
            st.sidebar.caption("No matches. Try a different keyword or browse the Glossary tab.")
    else:
        st.sidebar.caption(f"{len(_glossary)} terms available · full browser in Glossary tab")
else:
    st.sidebar.caption("Glossary not loaded — check data/glossary.yaml exists.")


# ---------- CROSS-REGION VIEW ----------
if region_pick == "Cross-Region":
    st.title("Cross-Region Analysis")
    if "china" not in dataset.regions or "india" not in dataset.regions:
        st.warning("Need both China and India regions enabled in config.yaml.")
        st.stop()

    # ===== SECTION 1: Spread comparison (China vs India) =====
    st.markdown("### Section 1 · Spread Regime Comparison (China vs India)")
    st.caption("Compares mill margin proxies across regions, normalised by historical percentiles.")

    cs = cached_spread(dataset["china"], "china_full", file_mtime)
    ins = cached_spread(dataset["india"], "india_full", file_mtime)
    cmp = cross_region_comparison(cs, ins)

    if cmp.get("available"):
        st.markdown(f"**{cmp['interpretation']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("China spread now", f"USD {cs.current_snapshot['spread']:,.0f}/t",
                   f"P{cmp['china_percentile_now']:.0f} of history")
        c2.metric("India spread now", f"INR {ins.current_snapshot['spread']:,.0f}/t",
                   f"P{cmp['india_percentile_now']:.0f} of history")
        c3.metric("Divergence (z-diff)", f"{cmp['divergence_now']:+.2f}",
                   "India minus China")
        c4.metric("Overall correlation", f"{cmp['overall_correlation']:.2f}",
                   f"{cmp['overlap_months']} months overlap")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Spread Percentile Rank (within own history)",
                                              "Divergence: India z − China z"),
                              vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=cmp["china_percentile_series"].index,
                                   y=cmp["china_percentile_series"].values,
                                   mode="lines", name="China",
                                   line=dict(color=COLORS["danger"], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cmp["india_percentile_series"].index,
                                   y=cmp["india_percentile_series"].values,
                                   mode="lines", name="India",
                                   line=dict(color=COLORS["accent2"], width=2)), row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=COLORS["muted"], row=1, col=1)
        fig.add_trace(go.Scatter(x=cmp["divergence_series"].index,
                                   y=cmp["divergence_series"].values,
                                   mode="lines", name="Divergence",
                                   line=dict(color=COLORS["accent"], width=2),
                                   fill="tozeroy", fillcolor="rgba(31,78,121,0.1)"),
                       row=2, col=1)
        fig.add_hline(y=0, line_color=COLORS["muted"], row=2, col=1)
        fig.update_yaxes(title_text="Percentile", row=1, col=1)
        fig.update_yaxes(title_text="z-score difference", row=2, col=1)
        fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=520, showlegend=True)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Spread comparison not available: {cmp.get('reason')}")


    # ===== SECTION 2: Lead-Lag Analysis on Returns =====
    st.markdown("---")
    st.markdown("### Section 2 · Pairwise Lead-Lag Analysis")
    st.caption(
        "All metrics computed on monthly **returns** (log price changes), making the "
        "analysis currency-neutral and unit-free. Cointegration uses **levels** because "
        "it concerns long-run equilibrium. Pairs are formed across all available regions."
    )

    @st.cache_data(show_spinner="Running cross-region lead-lag analysis...")
    def cached_cross_region(_dataset, max_lag: int, granger_lag: int, file_mtime: float):
        from pipeline.cross_region import analyse_cross_region
        return analyse_cross_region(_dataset, max_lag=max_lag, granger_lag=granger_lag)

    crc1, crc2 = st.columns([1, 2])
    with crc1:
        cr_max_lag = st.slider("Max CCF lag (months)", 6, 36, 24, step=6,
                                  help="Range of lags to scan in cross-correlation.")
    with crc2:
        cr_granger_lag = st.slider("Granger max lag (months)", 1, 12, 6,
                                      help="Lag horizon for Granger causality test.")

    cross = cached_cross_region(dataset, cr_max_lag, cr_granger_lag, file_mtime)

    if not cross.success:
        st.warning(f"Cross-region analysis unavailable: {cross.error_msg}")
    else:
        # Live interpretation panel
        try:
            render_interpretation(
                narrator.narrate_cross_region_extended(cross),
                label="Cross-region interpretation",
                settings_summary=f"Regions: {', '.join(cross.regions_analysed).upper()} · "
                                  f"Common overlap: {cross.overlap_start} → {cross.overlap_end} "
                                  f"({cross.n_overlap}m) · Max lag: {cr_max_lag}m",
            )
        except Exception:
            pass

        st.caption(
            f"**Common overlap window:** {cross.overlap_start} → {cross.overlap_end} "
            f"({cross.n_overlap} months across all regions). "
            f"Pairwise overlaps may differ — see each row below."
        )

        # ----- Summary table -----
        st.markdown("##### Pairwise summary")
        rows = []
        for p in cross.pairs:
            # Format leader description
            if p.leader == "a":
                lead_desc = f"**{p.region_a.upper()}** leads {p.region_b.upper()}"
            elif p.leader == "b":
                lead_desc = f"**{p.region_b.upper()}** leads {p.region_a.upper()}"
            elif p.leader == "both":
                lead_desc = "bidirectional"
            else:
                lead_desc = "no clear leader"

            # Granger results
            ga_b = (f"{p.granger_a_to_b_pvalue:.3f}{' (ok)' if p.granger_a_to_b_pvalue and p.granger_a_to_b_pvalue < 0.05 else ''}"
                    if p.granger_a_to_b_pvalue is not None else "—")
            gb_a = (f"{p.granger_b_to_a_pvalue:.3f}{' (ok)' if p.granger_b_to_a_pvalue and p.granger_b_to_a_pvalue < 0.05 else ''}"
                    if p.granger_b_to_a_pvalue is not None else "—")

            # Cointegration result
            if p.cointegration_pvalue is not None:
                coint_str = f"{p.cointegration_pvalue:.3f}{' (ok) cointegrated' if p.cointegrated else ''}"
            else:
                coint_str = "—"

            rows.append({
                "Pair": f"{p.region_a.upper()} ↔ {p.region_b.upper()}",
                "n": p.n_obs,
                "Contemp corr": f"{p.contemporaneous_corr:.3f}",
                "Best lag (m)": p.best_lag,
                "Best |corr|": f"{abs(p.best_corr):.3f}",
                "Lead direction": lead_desc,
                f"Granger {p.region_a}→{p.region_b}": ga_b,
                f"Granger {p.region_b}→{p.region_a}": gb_a,
                "Cointegration p": coint_str,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "**Best lag** sign convention: positive = first region (A) leads, "
            "negative = second region (B) leads. **Granger significant** = p < 0.05 "
            "(statistical predictive precedence). **Cointegrated** = p < 0.05 "
            "(long-run equilibrium between price levels)."
        )

        # ----- CCF plots, one per pair -----
        st.markdown("##### Cross-correlation function (CCF) by pair")
        st.caption(
            "Each plot shows correlation across lags. Asymmetric peaks (one side higher than the other) "
            "indicate clearer leader-follower relationships. Symmetric peaks around zero suggest "
            "contemporaneous co-movement without clear lead."
        )

        n_pairs = len(cross.pairs)
        if n_pairs > 0:
            ccf_fig = make_subplots(
                rows=n_pairs, cols=1,
                subplot_titles=[f"{p.region_a.upper()} (A) ↔ {p.region_b.upper()} (B)"
                                for p in cross.pairs],
                vertical_spacing=0.12,
            )

            for i, p in enumerate(cross.pairs):
                row = i + 1
                # Bar chart of CCF
                ccf_fig.add_trace(
                    go.Bar(x=p.ccf_lags, y=p.ccf_values,
                              marker_color=[COLORS["accent"] if abs(v) >= 0.3 else COLORS["muted"]
                                              for v in p.ccf_values],
                              showlegend=False,
                              hovertemplate=f"Lag: %{{x}}m<br>Corr: %{{y:.3f}}<extra></extra>"),
                    row=row, col=1
                )
                # Vertical line at best lag
                if p.best_lag is not None:
                    ccf_fig.add_vline(x=p.best_lag, line=dict(color=COLORS["danger"], width=2, dash="dot"),
                                       row=row, col=1,
                                       annotation_text=f"Best: lag {p.best_lag}m",
                                       annotation_position="top right")
                # Zero line
                ccf_fig.add_hline(y=0, line=dict(color=COLORS["muted"], width=1), row=row, col=1)
                # 0.3 reference lines
                ccf_fig.add_hline(y=0.3, line=dict(color=COLORS["accent2"], width=1, dash="dot"),
                                    row=row, col=1)
                ccf_fig.add_hline(y=-0.3, line=dict(color=COLORS["accent2"], width=1, dash="dot"),
                                    row=row, col=1)

            # Update axis labels
            for i in range(n_pairs):
                ccf_fig.update_yaxes(title_text="Correlation", row=i + 1, col=1)
                if i == n_pairs - 1:
                    ccf_fig.update_xaxes(
                        title_text="Lag (months) — positive = first region leads, negative = second leads",
                        row=i + 1, col=1
                    )

            ccf_fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN,
                                    height=300 * n_pairs, showlegend=False)
            style_axes(ccf_fig)
            st.plotly_chart(ccf_fig, use_container_width=True)

        # Methodology note
        with st.expander("Methodology notes"):
            st.markdown("""
**Returns**: log differences of monthly prices. Currency-neutral and unit-free,
removing confounds from differing currencies (USD vs INR) and units
(metric tonne vs short ton).

**Best lag**: the lag (between -24 and +24 months) at which |correlation| is maximised.
- Positive lag = the first region (A) leads the second (B)
- Negative lag = the second region (B) leads the first (A)
- Lag 0 = contemporaneous co-movement, no lead-lag relationship

**Granger causality**: tests whether past values of one series help predict the other,
beyond what the other's own history predicts. p < 0.05 = significant predictive precedence.
This is statistical leadership, not true causation.

**Engle-Granger cointegration**: tests whether two non-stationary price series have a stable
long-run relationship. If cointegrated, deviations from equilibrium predict reversion.
Important for trading-strategy considerations across regions.
""")
    st.stop()


# ---------- SINGLE-REGION VIEW ----------
region = dataset[region_pick]
currency = region.currency

st.title(f"{region_pick.title()} — HRC Steel Intelligence")
st.caption(f"{region.n_obs} monthly observations · {region.df.index.min().strftime('%b %Y')} – "
           f"{region.df.index.max().strftime('%b %Y')} · {len(region.drivers)} drivers · {currency}")

# Sidebar: per-region controls
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {region_pick.title()} controls")

# Date range slicer
min_d = region.df.index.min().date()
max_d = region.df.index.max().date()
date_range = st.sidebar.slider("Date range", min_value=min_d, max_value=max_d,
                                 value=(min_d, max_d), format="MMM YYYY")

# Driver toggles
selected_drivers = st.sidebar.multiselect(
    "Drivers to include",
    options=region.drivers,
    default=region.drivers,
    help="Untick to exclude from analyses below.",
)

# Apply filters
mask = (region.df.index.date >= date_range[0]) & (region.df.index.date <= date_range[1])
filt_df = region.df.loc[mask]
filt_y = filt_df[region.target]
filt_X = filt_df[selected_drivers] if selected_drivers else pd.DataFrame(index=filt_df.index)

# Build a transient region-data object for filtered analyses
class _FilteredRegion:
    def __init__(self, original, df, X, y):
        self.name = original.name; self.currency = original.currency
        self.target = original.target; self.drivers = list(X.columns)
        self.df = df; self.spread_config = original.spread_config
        self.y = y; self.X = X
        self.n_obs = len(df)

filt_region = _FilteredRegion(region, filt_df, filt_X, filt_y)

# Cache key — captures everything that should invalidate downstream caches.
# Region name + filtered date range + sorted driver list. Switching region OR
# changing date OR toggling drivers all bust the cache, as expected.
filter_key = (
    f"{region_pick}|{date_range[0]}|{date_range[1]}|"
    f"{','.join(sorted(selected_drivers))}"
)


# ---------- TABS ----------
# Show all tabs always; for overview-only regions (e.g., US without drivers),
# the tab body checks _overview_only and shows a friendly placeholder.
_overview_only = getattr(region, "overview_only", False)
_has_liquidity = bool(getattr(region, "liquidity_cols", []))

# The Liquidity tab is region-specific: it appears only when the region has a
# liquidity block configured in config.yaml (currently India only). This keeps
# the tab strip uncluttered for regions where it wouldn't apply.
if _has_liquidity:
    tab_overview, tab_spread, tab_diag, tab_lead_lag, tab_regimes, tab_cyclicity, tab_attribution, tab_forecast, tab_macro, tab_liquidity = st.tabs([
        "Overview", "Spread", "Diagnostics", "Lead/Lag", "Regimes", "Cyclicity", "Attribution", "Forecasts", "Macro Calendar", "Liquidity"
    ])
else:
    tab_overview, tab_spread, tab_diag, tab_lead_lag, tab_regimes, tab_cyclicity, tab_attribution, tab_forecast, tab_macro = st.tabs([
        "Overview", "Spread", "Diagnostics", "Lead/Lag", "Regimes", "Cyclicity", "Attribution", "Forecasts", "Macro Calendar"
    ])
    tab_liquidity = None # sentinel; tab body guards on this
tab_glossary = None # glossary is sidebar-only; tab body guarded


def _render_overview_only_placeholder(tab_name: str):
    """Render a friendly notice inside a tab when the region is overview-only."""
    st.info(
        f"**{tab_name} analysis is not available for {region_pick.title()}** because "
        f"this region currently has no driver data configured. Only the Overview tab "
        f"is active for this region."
    )
    st.caption(
        f"To enable {tab_name.lower()} analysis: add driver columns to the {region_pick.upper()} "
        f"sheet in `Raw_data.xlsx`, then set `overview_only: false` in `config.yaml`."
    )


# ===== OVERVIEW =====
with tab_overview:
    st.subheader(f"{region.target}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filt_y.index, y=filt_y.values,
                              mode="lines", line=dict(color=COLORS["accent"], width=2),
                              fill="tozeroy", fillcolor="rgba(31,78,121,0.06)",
                              hovertemplate=f"%{{x|%b %Y}}: {currency} %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=440, yaxis_title=f"{currency}/t")
    style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ===== WHAT CHANGED — the Monday-morning glance panel =====
    # Always-on. Tells the user in 30 seconds what's actually new since last
    # month. Degrades cleanly for overview-only regions (no driver delta or
    # spread percentile shown).
    st.markdown("---")
    st.subheader("What Changed")
    _full_y = region.df[region.target].dropna()
    _latest_date = _full_y.index[-1]
    st.caption(f"Snapshot anchored on the latest observation. Data as of: "
                 f"**{_latest_date.strftime('%b %Y')}** "
                 f"({len(_full_y)} months of history).")

    if len(_full_y) < 13:
        st.info("Need at least 13 months of history to render the What Changed panel.")
    else:
        _last_px = float(_full_y.iloc[-1])
        _prev_px = float(_full_y.iloc[-2])
        _mom_abs = _last_px - _prev_px
        _mom_pct = (_mom_abs / _prev_px * 100.0) if _prev_px else 0.0

        _avg_3m = float(_full_y.iloc[-3:].mean())
        _avg_12m = float(_full_y.iloc[-12:].mean())
        _vs_3m_pct = (_last_px / _avg_3m - 1) * 100.0 if _avg_3m else 0.0
        _vs_12m_pct = (_last_px / _avg_12m - 1) * 100.0 if _avg_12m else 0.0

        # Row 1: price tiles
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Latest price", f"{currency} {_last_px:,.0f}/t",
                     f"{_mom_abs:+,.0f} MoM")
        wc2.metric("MoM change", f"{_mom_pct:+.1f}%",
                     help="Month-on-month % change in HRC price.")
        wc3.metric("vs 3M avg", f"{_vs_3m_pct:+.1f}%",
                     help=f"Latest price vs trailing 3-month average ({currency} {_avg_3m:,.0f}/t).")
        wc4.metric("vs 12M avg", f"{_vs_12m_pct:+.1f}%",
                     help=f"Latest price vs trailing 12-month average ({currency} {_avg_12m:,.0f}/t).")

        # Row 2: drivers + spread (only when available)
        _row2_items = []

        # Biggest driver mover (skip if overview-only / no drivers).
        # Uses z-score of MoM change vs that driver's own history — works
        # cleanly for all drivers regardless of whether they're prices,
        # rates, indices, or spreads that can flip sign. Falls back to
        # % change for display where the driver is strictly positive
        # (prices, indices) and to absolute change for drivers that can
        # cross zero (rates, spreads).
        _biggest_drv = None
        _biggest_drv_label = ""
        _biggest_drv_z = 0.0
        if not _overview_only and len(region.drivers) > 0:
            _drv_movers = []
            for d_ in region.drivers:
                _s = region.df[d_].dropna()
                if len(_s) < 13:
                    continue
                _diff = _s.diff().dropna()
                _std = float(_diff.std())
                if _std <= 0 or not np.isfinite(_std):
                    continue
                _last_diff = float(_diff.iloc[-1])
                _z = _last_diff / _std
                _v0, _v1 = float(_s.iloc[-2]), float(_s.iloc[-1])
                # Decide display format: % only if series is strictly positive
                _all_positive = (_s.min() > 0)
                if _all_positive and _v0 != 0:
                    _label = f"{(_v1/_v0 - 1)*100.0:+.1f}% MoM"
                else:
                    _label = f"{_last_diff:+,.2f} MoM (abs)"
                _drv_movers.append((d_, _z, _label, _v0, _v1))
            if _drv_movers:
                _biggest_drv = max(_drv_movers, key=lambda r: abs(r[1]))
                _biggest_drv_label = _biggest_drv[2]
                _biggest_drv_z = _biggest_drv[1]

        # Spread percentile (China/India only — need spread_config)
        _spread_pct_now = None
        _spread_pct_prev = None
        _spread_now_value = None
        try:
            if not _overview_only and region.spread_config:
                _sp_result = cached_spread(region, f"{region_pick}_full", file_mtime)
                if _sp_result is not None:
                    _sp = _sp_result.spread_series.dropna()
                    if len(_sp) >= 13:
                        _sp_now = float(_sp.iloc[-1])
                        _sp_prev = float(_sp.iloc[-2])
                        _sp_arr = _sp.values
                        _spread_pct_now = float((_sp_arr <= _sp_now).mean() * 100.0)
                        _spread_pct_prev = float((_sp_arr <= _sp_prev).mean() * 100.0)
                        _spread_now_value = _sp_now
        except Exception:
            pass

        if _biggest_drv is not None or _spread_pct_now is not None:
            wc5, wc6, wc7 = st.columns(3)
            if _biggest_drv is not None:
                _drv_name, _z, _label_text, _drv_prev, _drv_now = _biggest_drv
                _drv_display = _drv_name if len(_drv_name) <= 24 else _drv_name[:22] + "…"
                wc5.metric(f"Biggest driver mover", _drv_display,
                             _label_text,
                             help=f"Driver whose MoM move was largest in z-score terms "
                                  f"(how unusual vs that driver's own monthly volatility). "
                                  f"Z-score this month: {_z:+.2f}σ. "
                                  f"Full name: {_drv_name}. "
                                  f"Latest: {_drv_now:,.2f}, previous: {_drv_prev:,.2f}.")
            else:
                wc5.metric("Biggest driver mover", "—", help="No driver data available for this region.")

            if _spread_pct_now is not None:
                _pct_delta = _spread_pct_now - _spread_pct_prev
                wc6.metric("Spread percentile", f"P{_spread_pct_now:.0f}",
                             f"{_pct_delta:+.0f} pts MoM",
                             help="Where current Tata BPM spread sits in its own history "
                                  "(P50 = median, P90 = top decile). Higher = stronger margin.")
                wc7.metric("Spread level", f"{currency} {_spread_now_value:,.0f}/t",
                             help="Latest HRC − 1.6×IO − 0.9×HCC spread value.")
            else:
                wc6.metric("Spread percentile", "—", help="Spread analysis not configured for this region.")

        # The one-line "what flipped" callout — generated dynamically
        _flips = []
        # 1. Direction flip vs trend
        if _mom_pct > 0 and _vs_12m_pct < -2:
            _flips.append(f"Price ticked up {_mom_pct:+.1f}% MoM but is still {_vs_12m_pct:.1f}% below 12M avg — early reversal or noise?")
        elif _mom_pct < 0 and _vs_12m_pct > 2:
            _flips.append(f"Price pulled back {_mom_pct:.1f}% MoM but remains {_vs_12m_pct:+.1f}% above 12M avg — pause in uptrend.")
        elif abs(_mom_pct) > 5:
            _flips.append(f"Notable {_mom_pct:+.1f}% MoM move — outside typical monthly range.")

        # 2. Spread percentile shifts
        if _spread_pct_now is not None and _spread_pct_prev is not None:
            _pdelta = _spread_pct_now - _spread_pct_prev
            if _spread_pct_now < 25 and _pdelta < -10:
                _flips.append(f"Spread dropped to P{_spread_pct_now:.0f} (down {abs(_pdelta):.0f} pts) — margin compression accelerating.")
            elif _spread_pct_now > 75 and _pdelta > 10:
                _flips.append(f"Spread climbed to P{_spread_pct_now:.0f} (up {_pdelta:.0f} pts) — margin expansion accelerating.")
            elif _spread_pct_now < 25:
                _flips.append(f"Spread sitting at P{_spread_pct_now:.0f} of history — bottom-quartile margin environment.")
            elif _spread_pct_now > 75:
                _flips.append(f"Spread sitting at P{_spread_pct_now:.0f} of history — top-quartile margin environment.")

        # 3. Driver outsized move — z-score above 2σ flags a genuinely unusual move
        if _biggest_drv is not None and abs(_biggest_drv_z) >= 2.0:
            _drv_name = _biggest_drv[0]
            _flips.append(f"{_drv_name} moved {_biggest_drv_label} ({_biggest_drv_z:+.1f}σ event) — large input shock, watch for HRC pass-through over next 1-3 months.")

        if _flips:
            st.markdown("**Headline reads:**")
            for _f in _flips:
                st.markdown(f"- {_f}")
        else:
            st.caption("No major flips this month — prices, drivers and spread broadly in line with recent trend.")

    st.markdown("---")

    st.subheader("Drivers")
    n_cols = 2
    for i in range(0, len(selected_drivers), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j >= len(selected_drivers):
                continue
            d = selected_drivers[i + j]
            with col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filt_df.index, y=filt_df[d].values,
                                          mode="lines",
                                          line=dict(color=CHART_PALETTE[(i + j) % len(CHART_PALETTE)], width=1.6)))
                fig.update_layout(**PLOT_BASE, margin=COMPACT_MARGIN, height=240, title=d)
                style_axes(fig)
                st.plotly_chart(fig, use_container_width=True)


# ===== SPREAD =====
with tab_spread:
    if _overview_only:
        _render_overview_only_placeholder("Spread")
    else:
        spread = cached_spread(region, f"{region_pick}_full", file_mtime)
        if spread is None:
            st.warning(f"No spread config for {region_pick}. Add `spread:` block to "
                       f"config.yaml under data.regions.{region_pick}.")
        else:
            # Live interpretation panel (collapsed by default)
            render_interpretation(
                narrator.narrate_spread(spread, currency),
                label="Spread interpretation",
                settings_summary=f"Region: {region_pick.title()}, Currency: {currency}",
            )

            cur = spread.current_snapshot
            st.markdown(f"**Spread = HRC − ({spread.spread_config.get('iron_ore_weight', 1.6)} × Iron Ore + "
                        f"{spread.spread_config.get('hcc_weight', 0.9)} × HCC)** in {currency}/t")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"HRC ({currency}/t)", f"{cur['hrc']:,.0f}", cur['as_of'])
            c2.metric("Raw material cost", f"{cur['rm_cost']:,.0f}",
                      f"{cur['rm_cost']/cur['hrc']*100:.0f}% of HRC")
            c3.metric("Spread", f"{cur['spread']:,.0f}",
                      f"{'+' if cur['vs_p50']>=0 else ''}{cur['vs_p50']:,.0f} vs median")
            c4.metric("Percentile", f"P{cur['spread_percentile']:.0f}",
                      cur['current_regime'].upper())

            # Spread time series
            s = spread.spread_series; p = spread.percentiles
            fig = go.Figure()
            fig.add_hrect(y0=p["p10"], y1=p["p25"], fillcolor=COLORS["danger"],
                          opacity=0.06, line_width=0)
            fig.add_hrect(y0=p["p75"], y1=p["p90"], fillcolor=COLORS["accent2"],
                          opacity=0.06, line_width=0)
            fig.add_hline(y=p["p50"], line_dash="dot", line_color=COLORS["muted"],
                          annotation_text="Median")
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                      line=dict(color=COLORS["accent"], width=2.5)))
            fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=400, yaxis_title=f"Spread ({currency}/t)")
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Decomposition
            st.subheader("HRC Price Decomposition")
            d = spread.decomposition
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d.index, y=d["IronOre_contribution"],
                                      mode="lines", line=dict(width=0), stackgroup="one",
                                      name="Iron Ore (1.6×)", fillcolor="rgba(31,78,121,0.7)"))
            fig.add_trace(go.Scatter(x=d.index, y=d["HCC_contribution"],
                                      mode="lines", line=dict(width=0), stackgroup="one",
                                      name="HCC (0.9×)", fillcolor="rgba(45,106,79,0.7)"))
            fig.add_trace(go.Scatter(x=d.index, y=d["Spread"],
                                      mode="lines", line=dict(width=0), stackgroup="one",
                                      name="Spread", fillcolor="rgba(201,84,15,0.6)"))
            fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=400, hovermode="x unified",
                               yaxis_title=f"{currency}/t")
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("FY Average Spread")
            st.dataframe(spread.fy_table, use_container_width=True, hide_index=True)


# ===== DIAGNOSTICS =====
with tab_diag:
    if _overview_only:
        _render_overview_only_placeholder("Diagnostics")
    else:
        if len(selected_drivers) == 0:
            st.warning("Select at least one driver in the sidebar.")
        else:
            diag = cached_diagnostics(filt_region, filter_key, file_mtime)
            st.subheader("Correlation Matrix")
            corr = diag["corr"]
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0, COLORS["danger"]], [0.5, "white"], [1, COLORS["accent2"]]],
                zmin=-1, zmax=1, text=np.round(corr.values, 2),
                texttemplate="%{text}", textfont=dict(size=10)))
            fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=500)
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Multicollinearity (VIF)")
            st.dataframe(diag["vif"], use_container_width=True, hide_index=True)
            st.caption("VIF > 10 = severe multicollinearity. Consider removing one of the affected drivers.")

            st.subheader("Stationarity (ADF)")
            st.dataframe(diag["adf"], use_container_width=True, hide_index=True)


# ===== LEAD/LAG =====
with tab_lead_lag:
    if _overview_only:
        _render_overview_only_placeholder("Lead/Lag")
    else:
        if len(selected_drivers) == 0:
            st.warning("Select at least one driver in the sidebar.")
        else:
            max_lag = st.slider("Max lag (months)", 3, 24, 12)
            ll = cached_lead_lag(filt_region, filter_key, max_lag, file_mtime)

            # Live interpretation
            try:
                target_name = filt_region.y.name if hasattr(filt_region.y, "name") else "HRC"
                render_interpretation(
                    narrator.narrate_lead_lag(ll, target_name),
                    label="Lead/Lag interpretation",
                    settings_summary=f"Region: {region_pick.title()}, Max lag: ±{max_lag}m, "
                                      f"Drivers selected: {len(selected_drivers)}",
                )
            except Exception as _e:
                pass

            st.subheader("Best Lead/Lag per Driver")
            fig = go.Figure()
            colors_list = [COLORS["accent"] if g else COLORS["muted"]
                            for g in ll["granger_x_causes_y"]]
            fig.add_trace(go.Bar(y=ll["driver"], x=ll["best_lag_months"],
                                  orientation="h", marker_color=colors_list,
                                  text=[f"r={c:.2f} (lag {l:+d})" for c, l in
                                        zip(ll["ccf_at_best_lag"], ll["best_lag_months"])],
                                  textposition="outside"))
            fig.add_vline(x=0, line_color=COLORS["muted"], line_dash="dot")
            fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=max(380, len(ll) * 30),
                               xaxis_title="Best lag (months)",
                               yaxis=dict(autorange="reversed"))
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Filled bars = Granger-significant. Positive lag = driver leads price.")
            st.dataframe(ll, use_container_width=True, hide_index=True)


# ===== REGIMES =====
with tab_regimes:
    if _overview_only:
        _render_overview_only_placeholder("Regimes")
    else:
        if len(selected_drivers) == 0:
            st.warning("Select at least one driver in the sidebar.")
        else:
            n_reg = st.slider("Number of regimes", 2, 5, 3)
            regimes = cached_regimes(filt_region, filter_key, n_reg, file_mtime)

            # Live interpretation
            try:
                render_interpretation(
                    narrator.narrate_regimes(regimes, currency),
                    label="Regimes interpretation",
                    settings_summary=f"Region: {region_pick.title()}, Regimes: {n_reg}, "
                                      f"Drivers: {len(selected_drivers)}",
                )
            except Exception:
                pass

            st.metric("Current regime", f"Regime {regimes.current_regime}",
                       f"of {regimes.n_regimes}")
            # Plot
            target = filt_y
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=target.index, y=target.values, mode="lines",
                                      line=dict(color=COLORS["ink"], width=1.5),
                                      name="Price"))
            for r in range(regimes.n_regimes):
                mask = regimes.labels == r
                fig.add_trace(go.Scatter(x=target.index[mask], y=target.values[mask],
                                          mode="markers", name=f"Regime {r}",
                                          marker=dict(size=8, opacity=0.7,
                                                       color=CHART_PALETTE[r % len(CHART_PALETTE)])))
            fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=440, yaxis_title=f"{currency}/t")
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(regimes.regime_stats, use_container_width=True, hide_index=True)


# ===== CYCLICITY =====
with tab_cyclicity:
    if _overview_only:
        _render_overview_only_placeholder("Cyclicity")
    else:
        n_cyc = st.slider("Number of GMM regimes", 2, 5, 4, key="cyc_nreg")
        try:
            cyc = cached_cyclicity(region, f"{region_pick}_full", n_cyc, file_mtime)
        except Exception as e:
            st.error(f"Cyclicity analysis failed: {type(e).__name__}: {e}")
            st.stop()

        # Live interpretation
        try:
            render_interpretation(
                narrator.narrate_cyclicity(cyc, currency),
                label="Cyclicity interpretation",
                settings_summary=f"Region: {region_pick.title()}, GMM regimes: {n_cyc}",
            )
        except Exception:
            pass

        # Top-line metrics
        cur_p = cyc.regime_profiles[cyc.current_regime]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current regime", f"R{cyc.current_regime}",
                   cur_p.label[:28] + "..." if len(cur_p.label) > 28 else cur_p.label)
        c2.metric("Peaks identified", cyc.cycle_stats.n_peaks,
                   f"{cyc.cycle_stats.n_troughs} troughs")
        if cyc.cycle_stats.avg_peak_to_peak_months:
            c3.metric("Avg peak-to-peak", f"{cyc.cycle_stats.avg_peak_to_peak_months:.1f}m")
        if cyc.cycle_stats.avg_amplitude_pct:
            c4.metric("Avg amplitude", f"{cyc.cycle_stats.avg_amplitude_pct:.1f}%")

        # Regime-coloured price chart
        st.subheader("Regime classification (Multi-feature GMM)")
        target = region.y
        palette = [COLORS["danger"], COLORS["muted"], COLORS["warning"], COLORS["accent2"]]
        if cyc.n_regimes > 4:
            palette = CHART_PALETTE[:cyc.n_regimes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=target.index, y=target.values, mode="lines",
                                  line=dict(color=COLORS["ink"], width=1.5), name="Price"))
        for r in range(cyc.n_regimes):
            mask = cyc.regime_labels == r
            if mask.sum() == 0: continue
            idx = cyc.regime_labels.index[mask]
            vals = target.reindex(idx).values
            label = cyc.regime_profiles[r].label
            fig.add_trace(go.Scatter(x=idx, y=vals, mode="markers",
                                       name=f"R{r}: {label}",
                                       marker=dict(size=8, opacity=0.75,
                                                    color=palette[r % len(palette)])))
        fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=440,
                           yaxis_title=f"{currency}/t",
                           legend=dict(orientation="h", y=-0.15))
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Per-regime stats
        st.subheader("Per-regime statistics")
        rows = []
        for p in cyc.regime_profiles:
            c = cyc.per_regime_cycles.get(p.regime_id)
            gp = cyc.garch_persistence.get(p.regime_id)
            rows.append({
                "Regime": f"R{p.regime_id}",
                "Label": p.label,
                "Months": p.n_months,
                f"Mean ({currency})": f"{p.mean_target:,.0f}" if not np.isnan(p.mean_target) else "—",
                "Avg Return %/mo": f"{p.avg_monthly_return_pct:+.2f}" if not np.isnan(p.avg_monthly_return_pct) else "—",
                "# Spells": p.n_spells,
                "Avg Spell (m)": f"{p.avg_spell_duration:.1f}",
                "Cycle (m)": f"{c.dominant_spectral_period_months:.1f}" if c and c.dominant_spectral_period_months else "—",
                "GARCH Persist": f"{gp:.3f}" if gp is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Peaks & troughs chart
        st.subheader("Peak/trough detection")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=target.index, y=target.values, mode="lines",
                                  line=dict(color=COLORS["accent"], width=1.8), name="Price"))
        if len(cyc.peaks) > 0:
            fig.add_trace(go.Scatter(x=cyc.peaks.index, y=cyc.peaks.values,
                                       mode="markers", name=f"Peaks (n={len(cyc.peaks)})",
                                       marker=dict(symbol="triangle-up", size=11,
                                                    color=COLORS["accent2"])))
        if len(cyc.troughs) > 0:
            fig.add_trace(go.Scatter(x=cyc.troughs.index, y=cyc.troughs.values,
                                       mode="markers", name=f"Troughs (n={len(cyc.troughs)})",
                                       marker=dict(symbol="triangle-down", size=11,
                                                    color=COLORS["danger"])))
        fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=420,
                           yaxis_title=f"{currency}/t")
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Markov transition matrix
        st.subheader("Markov transition matrix")
        tm = cyc.transition_matrix
        labels = [f"R{i}" for i in range(cyc.n_regimes)]
        fig = go.Figure(data=go.Heatmap(
            z=tm.values, x=labels, y=labels,
            colorscale=[[0, "white"], [0.5, COLORS["accent"]], [1, COLORS["ink"]]],
            zmin=0, zmax=1,
            text=np.round(tm.values * 100, 1), texttemplate="%{text}%",
            textfont=dict(size=11)))
        fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=400,
                           xaxis_title="To", yaxis_title="From",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Macro cycles
        if len(cyc.macro_cycles) > 0:
            st.subheader("Macro cycles identified")
            st.dataframe(cyc.macro_cycles, use_container_width=True, hide_index=True)


# ===== ATTRIBUTION =====
with tab_attribution:
    if _overview_only:
        _render_overview_only_placeholder("Attribution")
    else:
        if len(selected_drivers) < 2:
            st.warning("Select at least two drivers in the sidebar.")
        else:
            window = st.slider("Rolling window (months)", 12, 48, 24)
            attr = cached_attribution(filt_region, filter_key, window, file_mtime)
            if len(attr.rolling_betas) == 0:
                st.warning(f"Not enough data for {window}-month window.")
            else:
                # Live interpretation
                try:
                    render_interpretation(
                        narrator.narrate_attribution(attr, currency),
                        label="Attribution interpretation",
                        settings_summary=f"Region: {region_pick.title()}, "
                                          f"Rolling window: {window}m, "
                                          f"Drivers: {len(selected_drivers)}",
                    )
                except Exception:
                    pass

                st.subheader("Current attribution")
                cur_df = attr.current_attribution.reset_index()
                cur_df.columns = ["Driver", "Attribution (%)"]
                cur_df = cur_df.sort_values("Attribution (%)", ascending=False)
                cols = st.columns(min(len(cur_df), 5))
                for i, (_, row) in enumerate(cur_df.head(5).iterrows()):
                    cols[i].metric(row["Driver"], f"{row['Attribution (%)']:.1f}%")

                st.subheader("Rolling attribution over time")
                abs_betas = attr.rolling_betas.abs()
                shares = abs_betas.div(abs_betas.sum(axis=1), axis=0) * 100
                fig = go.Figure()
                for col in shares.columns:
                    fig.add_trace(go.Scatter(x=shares.index, y=shares[col], mode="lines",
                                              stackgroup="one", name=col))
                fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=440,
                                   yaxis_title="Share of explanatory power (%)",
                                   hovermode="x unified")
                style_axes(fig)
                st.plotly_chart(fig, use_container_width=True)


# ===== FORECASTS — Model Laboratory + OOS Test + GARCH =====
with tab_forecast:
    if _overview_only:
        _render_overview_only_placeholder("Forecasts")
    else:
        st.markdown("### Forecast Laboratory")
        st.caption(
            "Configure ARIMAX/ARDL parameters and inspect forecasts, out-of-sample "
            "tests, and GARCH volatility/risk metrics. All operations are cached — "
            "first run takes 5-30 seconds, subsequent views are instant."
        )

        # Reset button
        rc1, rc2 = st.columns([1, 5])
        with rc1:
            if st.button("Reset to defaults", help="Clear forecasting cache and reset all controls"):
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith("fc_")]
                for k in keys_to_clear:
                    del st.session_state[k]
                st.rerun()

        # ===== SECTION A: Model Laboratory =====
        st.markdown("---")
        st.markdown("#### Section A · Model Laboratory")
        st.caption("Fit ARIMAX or ARDL with full parameter control. Compare models side-by-side.")

        # Controls row 1: model selection + horizon
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            model_choice = st.radio(
                "Model", ["ARIMAX", "ARDL", "Both (compare)"],
                horizontal=True, key="fc_model_choice",
                help="ARIMAX: AutoRegressive Integrated Moving Average with eXogenous drivers. "
                     "ARDL: AutoRegressive Distributed Lag — uses lags of drivers. "
                     "Both: fit and compare side-by-side."
            )
        with col2:
            forecast_horizon = st.slider(
                "Forecast horizon (months)", 3, 24, 12, step=3, key="fc_horizon",
                help="How many months forward to forecast."
            )
        with col3:
            # Info display only — region from main selector
            st.metric("Region", region_pick.title(),
                       f"{region.n_obs} obs · {len(region.drivers)} drivers")

        # Controls row 2: ARIMAX/ARDL orders
        if model_choice in ("ARIMAX", "Both (compare)"):
            st.markdown("**ARIMAX parameters**")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                ar_arx = st.slider("AR order (p)", 0, 4, 2, key="fc_ar_arx",
                                     help="AutoRegressive order: number of lagged HRC values used.")
            with ac2:
                d_arx = st.slider("Differencing (d)", 0, 2, 1, key="fc_d_arx",
                                    help="Differencing order: 0 = levels, 1 = first difference (typical), 2 = second difference.")
            with ac3:
                ma_arx = st.slider("MA order (q)", 0, 4, 1, key="fc_ma_arx",
                                     help="Moving Average order: number of lagged residuals used.")
        else:
            ar_arx, d_arx, ma_arx = 2, 1, 1

        if model_choice in ("ARDL", "Both (compare)"):
            st.markdown("**ARDL parameters**")
            bc1, bc2, _ = st.columns(3)
            with bc1:
                ar_ardl = st.slider("AR order (lags of HRC)", 0, 4, 2, key="fc_ar_ardl",
                                      help="Number of lagged HRC values used in ARDL.")
            with bc2:
                dl_ardl = st.slider("DL order (lags of drivers)", 0, 4, 2, key="fc_dl_ardl",
                                      help="Number of lagged driver values used in ARDL.")
        else:
            ar_ardl, dl_ardl = 2, 2

        # Driver selection
        all_drivers = list(region.drivers)
        selected_drivers = st.multiselect(
            "Drivers to include in model",
            options=all_drivers,
            default=all_drivers,
            key=f"fc_drivers_{region.name}",
            help="Uncheck a driver to fit the model without it. Useful for testing each driver's contribution."
        )

        _forecast_can_run = len(selected_drivers) > 0
        if not _forecast_can_run:
            st.warning("Select at least one driver to fit the model.")

        if _forecast_can_run:
            # Fit + display the chosen model(s)
            drivers_tuple = tuple(sorted(selected_drivers))

            def _render_fit(fit_result, label, color):
                """Render one model fit's outputs."""
                if not fit_result.success:
                    st.error(f"(fail) **{label} failed**: {fit_result.error_msg}")
                    st.caption(f"Configuration: {fit_result.config_summary}")
                    return

                st.success(f"(ok) **{label}** — {fit_result.config_summary}")

                # Metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("RMSE", f"{currency} {fit_result.rmse:,.0f}/t")
                m2.metric("MAPE", f"{fit_result.mape:.2f}%")
                m3.metric("R²", f"{fit_result.r2:.3f}")
                m4.metric("AIC", f"{fit_result.aic:.0f}")
                m5.metric("BIC", f"{fit_result.bic:.0f}")

                # Ljung-Box residual diagnostic
                if fit_result.ljung_box_p is not None:
                    lb_status = "(ok) no autocorrelation" if fit_result.ljung_box_p > 0.05 else "(!) residual autocorrelation present"
                    st.caption(f"Ljung-Box test (10 lags): p = {fit_result.ljung_box_p:.3f} → {lb_status}")

                # Forecast chart
                target = region.y
                fig = go.Figure()
                # In-sample fit
                fig.add_trace(go.Scatter(
                    x=target.index, y=target.values, mode="lines",
                    line=dict(color=COLORS["ink"], width=1.5),
                    name="Actual"
                ))
                if fit_result.fitted_in_sample is not None:
                    fig.add_trace(go.Scatter(
                        x=fit_result.fitted_in_sample.index,
                        y=fit_result.fitted_in_sample.values,
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="dot"),
                        name="In-sample fit"
                    ))
                # Forecast + 95% CI
                if fit_result.forecast_mean is not None:
                    fig.add_trace(go.Scatter(
                        x=fit_result.forecast_upper_95.index,
                        y=fit_result.forecast_upper_95.values,
                        mode="lines", line=dict(width=0), showlegend=False,
                        hoverinfo="skip"
                    ))
                    fig.add_trace(go.Scatter(
                        x=fit_result.forecast_lower_95.index,
                        y=fit_result.forecast_lower_95.values,
                        mode="lines", line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba{tuple(list(_hex_to_rgb(color)) + [0.2])}",
                        name="95% CI"
                    ))
                    fig.add_trace(go.Scatter(
                        x=fit_result.forecast_mean.index,
                        y=fit_result.forecast_mean.values,
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        name=f"Forecast ({forecast_horizon}m)"
                    ))
                fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=420,
                                    yaxis_title=f"{currency}/t",
                                    title=dict(text=f"{label} forecast", font=dict(size=13)))
                style_axes(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Coefficient table
                with st.expander("Coefficient table"):
                    if fit_result.coefficients is not None:
                        cdf = fit_result.coefficients.copy()
                        cdf["coef"] = cdf["coef"].round(4)
                        cdf["std_err"] = cdf["std_err"].round(4)
                        cdf["p_value"] = cdf["p_value"].round(4)
                        cdf["sig"] = cdf["p_value"].apply(
                            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        )
                        st.dataframe(cdf, use_container_width=True, hide_index=True)
                        st.caption("Significance: * p<0.05, ** p<0.01, *** p<0.001")

            # Helper for color blending
            def _hex_to_rgb(hex_color):
                h = hex_color.lstrip("#")
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

            if model_choice == "ARIMAX":
                fit_arx = cached_arimax(region, f"{region_pick}", ar_arx, d_arx, ma_arx,
                                          forecast_horizon, drivers_tuple, file_mtime)
                _render_fit(fit_arx, "ARIMAX", COLORS["accent"])
                try:
                    render_interpretation(
                        narrator.narrate_model_fit(fit_arx, "ARIMAX", selected_drivers, currency),
                        label="ARIMAX interpretation",
                        settings_summary=f"AR={ar_arx}, d={d_arx}, MA={ma_arx}, "
                                          f"horizon={forecast_horizon}m, "
                                          f"{len(selected_drivers)} drivers",
                    )
                except Exception:
                    pass
            elif model_choice == "ARDL":
                fit_ardl = cached_ardl(region, f"{region_pick}", ar_ardl, dl_ardl,
                                         forecast_horizon, drivers_tuple, file_mtime)
                _render_fit(fit_ardl, "ARDL", COLORS["accent2"])
                try:
                    render_interpretation(
                        narrator.narrate_model_fit(fit_ardl, "ARDL", selected_drivers, currency),
                        label="ARDL interpretation",
                        settings_summary=f"AR={ar_ardl}, DL={dl_ardl}, "
                                          f"horizon={forecast_horizon}m, "
                                          f"{len(selected_drivers)} drivers",
                    )
                except Exception:
                    pass
            else: # Both (compare)
                fit_arx = cached_arimax(region, f"{region_pick}", ar_arx, d_arx, ma_arx,
                                          forecast_horizon, drivers_tuple, file_mtime)
                fit_ardl_r = cached_ardl(region, f"{region_pick}", ar_ardl, dl_ardl,
                                           forecast_horizon, drivers_tuple, file_mtime)

                # Two-column comparison
                st.markdown("##### Side-by-side comparison")
                comp_data = []
                for fit_res, name in [(fit_arx, "ARIMAX"), (fit_ardl_r, "ARDL")]:
                    if fit_res.success:
                        comp_data.append({
                            "Model": name,
                            "RMSE": f"{fit_res.rmse:,.0f}",
                            "MAPE %": f"{fit_res.mape:.2f}",
                            "R²": f"{fit_res.r2:.3f}",
                            "AIC": f"{fit_res.aic:.0f}",
                            "BIC": f"{fit_res.bic:.0f}",
                            "Status": "(ok)",
                        })
                    else:
                        comp_data.append({
                            "Model": name, "RMSE": "—", "MAPE %": "—", "R²": "—",
                            "AIC": "—", "BIC": "—", "Status": "(fail)",
                        })
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

                # Render each in collapsible expanders to keep things tidy
                with st.expander("ARIMAX details", expanded=True):
                    _render_fit(fit_arx, "ARIMAX", COLORS["accent"])
                with st.expander("ARDL details"):
                    _render_fit(fit_ardl_r, "ARDL", COLORS["accent2"])

                # Combined interpretation comparing both
                try:
                    both_blocks = []
                    both_blocks.extend(narrator.narrate_model_fit(fit_arx, "ARIMAX", selected_drivers, currency))
                    both_blocks.append("<hr style='border: 0; border-top: 1px dashed #E5E9F0; margin: 16px 0;'>")
                    both_blocks.extend(narrator.narrate_model_fit(fit_ardl_r, "ARDL", selected_drivers, currency))
                    render_interpretation(
                        both_blocks,
                        label="Combined ARIMAX + ARDL interpretation",
                        settings_summary=f"ARIMAX({ar_arx},{d_arx},{ma_arx}) vs "
                                          f"ARDL(ar={ar_ardl}, dl={dl_ardl}), "
                                          f"horizon={forecast_horizon}m, "
                                          f"{len(selected_drivers)} drivers",
                    )
                except Exception:
                    pass


            # ===== SECTION A.5: Scenarios — driver shock forecasts =====
            st.markdown("---")
            st.markdown("#### Scenarios · Driver Shock Forecasts")
            st.caption(
                "Apply a one-time % shift to selected drivers and project that shifted level "
                "forward across the forecast horizon. The model is refit on actual history "
                "(no contamination) and compared against the base case (drivers held flat at "
                "their latest values). This isolates the question: *what does the model say "
                "if iron ore drops 15%, or coking coal spikes 20%?*"
            )
            st.warning(
                "**Read the output as directional, not precise.** Scenarios apply isolated "
                "driver shocks. In reality, big moves usually come bundled — a 15% iron-ore "
                "drop typically arrives with a demand slowdown that also drags coking coal "
                "and HRC. The model can't know that. Use these outputs to size *which* "
                "direction matters and *how much* a single driver could push price, not to "
                "produce a tradable forecast.",
                icon=None,
            )

            # Pick the scenario model — default to ARIMAX since it tends to fit faster
            sc_c1, sc_c2 = st.columns([1, 1])
            with sc_c1:
                sc_model = st.selectbox(
                    "Scenario model", ["ARIMAX", "ARDL"], key="fc_sc_model",
                    help="Which model to use for scenario forecasts. Orders are taken from "
                         "the Section A sliders above. Run both to see how transmission "
                         "speed differs."
                )
            with sc_c2:
                st.metric("Forecast horizon",
                           f"{forecast_horizon} months",
                           help="Inherited from Section A's horizon slider.")

            # Driver shock sliders — one per selected driver
            st.markdown("**Driver shocks** (one-time % shift applied to latest value, then held)")
            _shocks: dict = {}
            # 2-column grid for slider layout
            _drv_for_shocks = list(selected_drivers)
            _shock_cols_per_row = 2
            for _i in range(0, len(_drv_for_shocks), _shock_cols_per_row):
                _scols = st.columns(_shock_cols_per_row)
                for _j, _sc in enumerate(_scols):
                    if _i + _j >= len(_drv_for_shocks):
                        continue
                    _d = _drv_for_shocks[_i + _j]
                    with _sc:
                        # Truncate display name
                        _disp = _d if len(_d) <= 36 else _d[:34] + "…"
                        _shock_val = st.slider(
                            _disp,
                            min_value=-30.0, max_value=30.0,
                            value=0.0, step=1.0,
                            key=f"fc_sc_shock_{region_pick}_{_d}",
                            format="%+.0f%%",
                            help=f"% shift applied to {_d}'s latest value. "
                                 f"Held constant at the shifted level across the horizon."
                        )
                        if abs(_shock_val) > 0.01:
                            _shocks[_d] = float(_shock_val)

            # Quick-reset
            _rc1, _rc2 = st.columns([1, 5])
            with _rc1:
                if st.button("Reset all shocks", key="fc_sc_reset",
                               help="Set every shock slider back to 0%."):
                    for _d in _drv_for_shocks:
                        _k = f"fc_sc_shock_{region_pick}_{_d}"
                        if _k in st.session_state:
                            st.session_state[_k] = 0.0
                    st.rerun()
            with _rc2:
                if not _shocks:
                    st.caption("No shocks applied — base forecast only will be shown. Move any slider to apply a shock.")
                else:
                    _summary = ", ".join(f"{k}: {v:+.0f}%" for k, v in _shocks.items())
                    st.caption(f"Applied: {_summary}")

            if st.button("Run scenario", key="fc_sc_run", type="primary",
                           help="Refit the model on actual history and produce base + "
                                "shocked forecasts."):
                _shocks_tuple = tuple(sorted(_shocks.items())) if _shocks else tuple()
                _sc_result = cached_scenario(
                    region, region_pick, sc_model.lower(),
                    ar_arx, d_arx, ma_arx, dl_ardl,
                    forecast_horizon, drivers_tuple, _shocks_tuple, file_mtime
                )
                st.session_state["fc_sc_result"] = _sc_result
                st.session_state["fc_sc_shocks_applied"] = dict(_shocks)
                st.session_state["fc_sc_model_used"] = sc_model

            if "fc_sc_result" in st.session_state:
                _sc = st.session_state["fc_sc_result"]
                _shocks_applied = st.session_state.get("fc_sc_shocks_applied", {})
                _model_used = st.session_state.get("fc_sc_model_used", sc_model)

                if not _sc.success:
                    st.error(f"Scenario run failed: {_sc.error_msg}")
                else:
                    # Headline metrics
                    _bf = _sc.base_forecast
                    _shf = _sc.shocked_forecast
                    _last_actual = float(_sc.recent_actuals.iloc[-1])

                    # 3M ahead delta (or use horizon-end if horizon < 3)
                    _idx_3m = min(2, len(_bf) - 1)
                    _idx_end = len(_bf) - 1

                    _base_3m = float(_bf.iloc[_idx_3m])
                    _sh_3m = float(_shf.iloc[_idx_3m])
                    _delta_3m = _sh_3m - _base_3m

                    _base_end = float(_bf.iloc[_idx_end])
                    _sh_end = float(_shf.iloc[_idx_end])
                    _delta_end = _sh_end - _base_end

                    st.markdown(f"##### Result ({_model_used})")

                    sm1, sm2, sm3, sm4 = st.columns(4)
                    sm1.metric("Latest actual",
                                 f"{currency} {_last_actual:,.0f}/t",
                                 help=f"Most recent observed HRC price ({_sc.recent_actuals.index[-1].strftime('%b %Y')}).")
                    sm2.metric(f"Base {min(3, forecast_horizon)}M",
                                 f"{currency} {_base_3m:,.0f}/t",
                                 f"{(_base_3m - _last_actual):+,.0f} vs actual",
                                 help=f"Forecast for {_bf.index[_idx_3m].strftime('%b %Y')} "
                                      f"with drivers held flat at their latest values.")
                    sm3.metric(f"Shocked {min(3, forecast_horizon)}M",
                                 f"{currency} {_sh_3m:,.0f}/t",
                                 f"{_delta_3m:+,.0f} vs base",
                                 help=f"Forecast for {_shf.index[_idx_3m].strftime('%b %Y')} "
                                      f"with shocks applied.")
                    sm4.metric(f"Delta at horizon ({forecast_horizon}M)",
                                 f"{_delta_end:+,.0f} {currency}/t",
                                 f"{(_delta_end / _base_end * 100):+.1f}%" if _base_end else "—",
                                 help=f"Difference between shocked and base case at the "
                                      f"end of the forecast horizon ({_bf.index[_idx_end].strftime('%b %Y')}).")

                    # Side-by-side comparison chart
                    st.markdown("##### Base vs Shocked forecast")
                    _fig = go.Figure()
                    # Recent actuals
                    _fig.add_trace(go.Scatter(
                        x=_sc.recent_actuals.index, y=_sc.recent_actuals.values,
                        mode="lines+markers", name="Actual (last 12M)",
                        line=dict(color="#1F4E79", width=2.2),
                    ))
                    # Base forecast
                    _fig.add_trace(go.Scatter(
                        x=_bf.index, y=_bf.values,
                        mode="lines+markers", name="Base forecast",
                        line=dict(color="#6B7280", width=2, dash="dot"),
                    ))
                    # Shocked forecast (only meaningful if shocks present)
                    if _shocks_applied:
                        _fig.add_trace(go.Scatter(
                            x=_shf.index, y=_shf.values,
                            mode="lines+markers", name="Shocked forecast",
                            line=dict(color="#B91C1C", width=2.4),
                        ))
                        # Fill between base and shocked to make delta visually obvious
                        _fig.add_trace(go.Scatter(
                            x=list(_bf.index) + list(_bf.index[::-1]),
                            y=list(_shf.values) + list(_bf.values[::-1]),
                            fill="toself", fillcolor="rgba(185,28,28,0.08)",
                            line=dict(color="rgba(0,0,0,0)"),
                            hoverinfo="skip", showlegend=False,
                        ))
                    _fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=420,
                                          yaxis_title=f"{currency}/t",
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    style_axes(_fig)
                    st.plotly_chart(_fig, use_container_width=True)

                    # Per-month table
                    st.markdown("##### Forecast table")
                    _tbl = pd.DataFrame({
                        "Month": [d.strftime("%b %Y") for d in _bf.index],
                        f"Base ({currency}/t)": [f"{v:,.0f}" for v in _bf.values],
                        f"Shocked ({currency}/t)": [f"{v:,.0f}" for v in _shf.values],
                        f"Δ ({currency}/t)": [f"{(s - b):+,.0f}" for s, b in zip(_shf.values, _bf.values)],
                        "Δ %": [f"{((s - b) / b * 100):+.1f}%" if b else "—" for s, b in zip(_shf.values, _bf.values)],
                    })
                    st.dataframe(_tbl, use_container_width=True, hide_index=True)

                    # Interpretation block
                    _interp_blocks = []
                    if not _shocks_applied:
                        _interp_blocks.append(
                            "No shocks applied — base and shocked forecasts are identical. "
                            "Move any driver slider away from 0% and re-run to see scenario impact."
                        )
                    else:
                        _shock_list = ", ".join(f"**{k}** {v:+.0f}%" for k, v in _shocks_applied.items())
                        _direction = "below" if _delta_end < 0 else "above"
                        _interp_blocks.append(
                            f"With {_shock_list} applied as one-time level shifts, the {_model_used} "
                            f"model places HRC at **{currency} {_sh_end:,.0f}/t** by "
                            f"{_shf.index[-1].strftime('%b %Y')} — "
                            f"**{abs(_delta_end):,.0f} {currency}/t {_direction}** the base case "
                            f"({(_delta_end / _base_end * 100):+.1f}%)."
                        )
                        # Magnitude framing
                        _last_actual_abs = abs(_last_actual) if _last_actual else 1.0
                        _pct_of_actual = abs(_delta_end) / _last_actual_abs * 100.0
                        if _pct_of_actual < 2.0:
                            _interp_blocks.append(
                                "**Interpretation: low sensitivity.** The model implies HRC barely "
                                "moves in response to this shock combination — coefficients on the "
                                "shocked drivers are small relative to the AR dynamics, or the "
                                "shocks partially offset each other."
                            )
                        elif _pct_of_actual < 8.0:
                            _interp_blocks.append(
                                "**Interpretation: moderate sensitivity.** The shock combination "
                                "moves HRC by single-digit percentage of the current price level. "
                                "This is within the band of typical monthly noise — the signal is "
                                "real but not dominating."
                            )
                        else:
                            _interp_blocks.append(
                                "**Interpretation: high sensitivity.** The shock combination moves "
                                "HRC by more than 8% of the current price level. If you believe "
                                "the shock is plausible, this is a material risk worth flagging "
                                "to commercial teams."
                            )
                        # Speed of transmission — compare 3M delta to horizon-end delta
                        if abs(_delta_end) > 1.0:
                            _speed = _delta_3m / _delta_end
                            if _speed > 0.7:
                                _interp_blocks.append(
                                    f"**Transmission speed: fast.** ~{_speed*100:.0f}% of the "
                                    f"horizon-end delta has already propagated by month 3, "
                                    f"meaning the model expects price to respond quickly."
                                )
                            elif _speed > 0.3:
                                _interp_blocks.append(
                                    f"**Transmission speed: gradual.** ~{_speed*100:.0f}% of the "
                                    f"horizon-end delta has propagated by month 3. The full "
                                    f"impact builds over several months."
                                )
                            else:
                                _interp_blocks.append(
                                    f"**Transmission speed: slow.** Only ~{_speed*100:.0f}% of "
                                    f"the horizon-end delta is visible by month 3. The bulk of "
                                    f"the impact materialises later in the horizon — useful "
                                    f"context for timing decisions."
                                )

                    render_interpretation(
                        _interp_blocks,
                        label="Scenario interpretation",
                        settings_summary=_sc.config_summary,
                    )


            # ===== SECTION B: Out-of-Sample Test =====
            st.markdown("---")
            st.markdown("#### Section B · Out-of-Sample Test")
            st.caption(
                "Hold out the most recent N months as a test set. Fit the model on data "
                "BEFORE that window, forecast forward through the test period using the "
                "actual driver values for those months, and compare predictions to the "
                "actuals you held out. Same code path the HTML report uses — numbers will "
                "match exactly when the orders match."
            )

            oos_c1, oos_c2, oos_c3 = st.columns([1, 1, 1])
            with oos_c1:
                oos_model = st.selectbox(
                    "Model", ["ARIMAX", "ARDL"], key="fc_oos_model",
                    help="Which model to evaluate. AR/d/MA/DL orders are taken from "
                         "the Section A sliders above so you can iterate quickly."
                )
            with oos_c2:
                # Cap test size sensibly — model classes need at least 12 obs of
                # training data on top of the test window
                oos_test_max = max(6, min(24, region.n_obs - 24))
                oos_test_size = st.slider(
                    "Test window (months)", 3, oos_test_max,
                    min(12, oos_test_max), step=3, key="fc_oos_test_size",
                    help="How many of the most recent months to hold out. "
                         "12 is the standard choice and matches the report's default."
                )
            with oos_c3:
                st.metric("Training months",
                           f"{region.n_obs - oos_test_size}",
                           f"of {region.n_obs} total")

            if st.button("Run out-of-sample test", key="fc_oos_run",
                           help="Click to fit on training data and evaluate on the "
                                "held-out window."):
                oos = cached_oos_test(region, region_pick, oos_model.lower(),
                                        ar_arx, d_arx, ma_arx, dl_ardl,
                                        oos_test_size, drivers_tuple, file_mtime)
                st.session_state["fc_oos_result"] = oos

            if "fc_oos_result" in st.session_state:
                oos = st.session_state["fc_oos_result"]
                if not oos.success:
                    st.error(f"Out-of-sample test failed: {oos.error_msg}")
                else:
                    # Frame the result before showing metrics
                    if oos.hit_rate >= 60:
                        skill_text = (f"A hit rate of **{oos.hit_rate:.0f}%** indicates "
                                       "the model has genuine directional skill on this "
                                       "window.")
                    elif oos.hit_rate >= 50:
                        skill_text = (f"A hit rate of **{oos.hit_rate:.0f}%** is only "
                                       "modestly above coin-flip — directional skill is weak.")
                    else:
                        skill_text = (f"A hit rate of **{oos.hit_rate:.0f}%** is below "
                                       "coin-flip — direction predictions failed on this window.")

                    st.info(
                        f"**Honest assessment:** Trained on {oos.n_train} months "
                        f"({oos.train_start} → {oos.train_end}), tested on {oos.n_test} "
                        f"months ({oos.test_start} → {oos.test_end}). "
                        f"Forecasts were on average **±{oos.mape:.1f}%** away from "
                        f"actual prices. {skill_text}"
                    )

                    # Metrics row
                    om1, om2, om3, om4, om5 = st.columns(5)
                    om1.metric("RMSE", f"{currency} {oos.rmse:,.0f}/t")
                    om2.metric("MAE", f"{currency} {oos.mae:,.0f}/t")
                    om3.metric("MAPE", f"{oos.mape:.2f}%")
                    om4.metric("R² (out-of-sample)", f"{oos.r2:.3f}",
                                help="Out-of-sample R². Can be negative if the forecast "
                                     "is worse than predicting the test-set mean — that "
                                     "indicates the model has no useful signal on this window.")
                    om5.metric("Hit rate (direction)", f"{oos.hit_rate:.0f}%",
                                help="% of test months where forecast direction matched "
                                     "actual direction (vs. previous value).")

                    # Ljung-Box on training residuals
                    if oos.ljung_box_p is not None:
                        lb_status = ("(ok) no residual autocorrelation" if oos.ljung_box_p > 0.05
                                      else "(!) residual autocorrelation present in training")
                        st.caption(f"Training-residual Ljung-Box (10 lags): "
                                    f"p = {oos.ljung_box_p:.3f} → {lb_status}")

                    # Chart: actual + in-sample fit + OOS predictions
                    st.markdown("##### Predicted vs actual on held-out window")
                    fig = go.Figure()
                    target = region.y

                    # Full actual series (black)
                    fig.add_trace(go.Scatter(
                        x=target.index, y=target.values, mode="lines",
                        line=dict(color=COLORS["ink"], width=1.5),
                        name="Actual",
                    ))

                    # In-sample fit (dotted accent — only training portion)
                    if oos.fitted_in_sample is not None:
                        fig.add_trace(go.Scatter(
                            x=oos.fitted_in_sample.index,
                            y=oos.fitted_in_sample.values,
                            mode="lines",
                            line=dict(color=COLORS["accent"], width=1.3, dash="dot"),
                            name="In-sample fit",
                            opacity=0.7,
                        ))

                    # OOS prediction line+markers (warning = orange, distinct)
                    fig.add_trace(go.Scatter(
                        x=oos.oos_predictions.index, y=oos.oos_predictions.values,
                        mode="lines+markers",
                        line=dict(color=COLORS["warning"], width=2.5),
                        marker=dict(size=7, symbol="diamond"),
                        name="OOS prediction",
                    ))

                    # Vertical divider at train/test split. Use add_shape +
                    # add_annotation separately rather than add_vline(annotation_text=...)
                    # because the latter crashes on a Timestamp x with current plotly
                    # versions ("Addition/subtraction of integers and Timestamp is no
                    # longer supported").
                    if oos.fitted_in_sample is not None and len(oos.fitted_in_sample) > 0:
                        split_x = oos.fitted_in_sample.index[-1]
                    else:
                        split_x = oos.oos_predictions.index[0]
                    split_x_str = (split_x.strftime("%Y-%m-%d")
                                    if hasattr(split_x, "strftime") else str(split_x))
                    fig.add_shape(
                        type="line", xref="x", yref="paper",
                        x0=split_x_str, x1=split_x_str, y0=0, y1=1,
                        line=dict(color=COLORS["muted"], width=1, dash="dash"),
                    )
                    fig.add_annotation(
                        x=split_x_str, y=1.02, xref="x", yref="paper",
                        text="train | test", showarrow=False,
                        font=dict(color=COLORS["muted"], size=11),
                    )

                    fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=440,
                                        yaxis_title=f"{currency}/t",
                                        hovermode="x unified",
                                        title=dict(
                                            text=f"{oos.model_type} — {oos.config_summary} · "
                                                  f"trained on {oos.n_train}m, tested on {oos.n_test}m",
                                            font=dict(size=13),
                                        ))
                    style_axes(fig)
                    st.plotly_chart(fig, use_container_width=True)

                    # Residuals table + chart
                    with st.expander("Test-period residuals (actual − predicted)"):
                        resid = oos.oos_actuals.values - oos.oos_predictions.values
                        resid_df = pd.DataFrame({
                            "Date": [d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d)
                                      for d in oos.oos_predictions.index],
                            f"Actual ({currency}/t)": oos.oos_actuals.values.round(0),
                            f"Predicted ({currency}/t)": oos.oos_predictions.values.round(0),
                            "Residual": resid.round(0),
                            "% error": (resid / oos.oos_actuals.values * 100).round(2),
                        })
                        st.dataframe(resid_df, use_container_width=True, hide_index=True)

                        rfig = go.Figure()
                        rfig.add_trace(go.Bar(
                            x=oos.oos_predictions.index, y=resid,
                            marker_color=[COLORS["accent2"] if r >= 0 else COLORS["danger"]
                                           for r in resid],
                            name="Residual",
                        ))
                        rfig.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
                        rfig.update_layout(**PLOT_BASE, margin=COMPACT_MARGIN, height=260,
                                            yaxis_title=f"Residual ({currency}/t)",
                                            title=dict(text="Residuals over test window",
                                                        font=dict(size=12)))
                        style_axes(rfig)
                        st.plotly_chart(rfig, use_container_width=True)

                    # Coefficient table from the training-period fit
                    if oos.coefficients is not None:
                        with st.expander("Training-period model coefficients"):
                            cdf = oos.coefficients.copy().reset_index()
                            # Round numeric columns
                            for col in cdf.columns:
                                if pd.api.types.is_numeric_dtype(cdf[col]):
                                    cdf[col] = cdf[col].round(4)
                            # Add significance stars if p_value column exists
                            if "p_value" in cdf.columns:
                                cdf["sig"] = cdf["p_value"].apply(
                                    lambda p: "***" if pd.notna(p) and p < 0.001 else
                                              "**" if pd.notna(p) and p < 0.01 else
                                              "*" if pd.notna(p) and p < 0.05 else ""
                                )
                            st.dataframe(cdf, use_container_width=True, hide_index=True)
                            st.caption("Significance: * p<0.05, ** p<0.01, *** p<0.001. "
                                        "These coefficients are estimated on training data only.")

                    # Plain-English interpretation expander
                    interp_blocks = []
                    interp_blocks.append(
                        f"**What this test shows.** The model was fit on the first "
                        f"{oos.n_train} months of data ({oos.train_start} through "
                        f"{oos.train_end}) and asked to forecast {oos.n_test} months "
                        f"forward. Those forecasts were then compared to the actual "
                        f"prices observed in {oos.test_start} – {oos.test_end}, which "
                        f"the model never saw during fitting."
                    )
                    interp_blocks.append(
                        f"**Magnitude error.** Average absolute miss was "
                        f"{currency} {oos.mae:,.0f}/t (MAE), or {oos.mape:.1f}% "
                        f"(MAPE). RMSE of {currency} {oos.rmse:,.0f}/t puts more weight "
                        f"on the worst months — when RMSE is meaningfully larger than MAE, "
                        f"a few forecasts missed badly while most were close."
                    )
                    if oos.r2 < 0:
                        r2_text = (f"Out-of-sample R² of **{oos.r2:.3f}** is negative — "
                                    "the model performed worse than simply predicting the "
                                    "test-set mean. This means the model's structure has "
                                    "no useful signal on this window; it's adding noise. "
                                    "Common causes: training period dynamics differ from "
                                    "the test period, or the test window is unusually flat.")
                    elif oos.r2 < 0.3:
                        r2_text = (f"Out-of-sample R² of **{oos.r2:.3f}** is low — "
                                    "the model captures only a small share of test-period "
                                    "variation. Treat point forecasts cautiously.")
                    elif oos.r2 < 0.7:
                        r2_text = (f"Out-of-sample R² of **{oos.r2:.3f}** is moderate — "
                                    "the model captures the broad shape of the test period "
                                    "but misses meaningful detail.")
                    else:
                        r2_text = (f"Out-of-sample R² of **{oos.r2:.3f}** is strong — "
                                    "the model tracks the test period well. Validate on "
                                    "additional windows before trusting it for live "
                                    "forecasting.")
                    interp_blocks.append(f"**Goodness of fit.** {r2_text}")
                    interp_blocks.append(
                        f"**Direction.** {skill_text} Note that with only {oos.n_test} "
                        f"comparisons, hit rate has wide confidence bands — repeat with "
                        f"different test windows or models before generalising."
                    )
                    interp_blocks.append(
                        "**Caveat.** This is a single train/test split. Performance on "
                        "this specific window may not reflect average performance — try a "
                        "different test size, swap models (ARIMAX vs ARDL), or remove "
                        "drivers in Section A and re-run to compare."
                    )
                    try:
                        render_interpretation(
                            interp_blocks,
                            label="Out-of-sample test interpretation",
                            settings_summary=f"{oos.model_type} {oos.config_summary}, "
                                              f"test window={oos.n_test}m, "
                                              f"{len(selected_drivers)} drivers",
                        )
                    except Exception:
                        pass


            # ===== SECTION C: GARCH Volatility + Risk =====
            st.markdown("---")
            st.markdown("#### Section C · GARCH Volatility & Risk Metrics")
            st.caption(
                "GARCH(1,1) model fitted on log-returns. Provides conditional volatility forecast, "
                "fan chart on point forecast, VaR/Expected Shortfall, and current volatility regime classification."
            )

            g = cached_garch(region, region_pick, forecast_horizon, file_mtime)

            if not g.success:
                st.error(f"GARCH fit failed: {g.error_msg}")
            else:
                # Live interpretation
                try:
                    render_interpretation(
                        narrator.narrate_garch_dashboard(g, currency,
                                                            latest_price=float(region.y.iloc[-1])),
                        label="GARCH / risk interpretation",
                        settings_summary=f"Region: {region_pick.title()}, "
                                          f"Forecast horizon: {forecast_horizon}m",
                    )
                except Exception:
                    pass

                # Top-line GARCH metrics
                gm1, gm2, gm3, gm4 = st.columns(4)
                gm1.metric("Persistence (α+β)", f"{g.persistence:.3f}",
                             help="Higher = volatility shocks decay slowly. Values near 1 indicate near-permanent shocks.")
                gm2.metric("Half-life", f"{g.half_life_months:.1f}m" if g.half_life_months and not np.isinf(g.half_life_months) else "∞",
                             help="Months until a vol shock decays to half its initial size.")
                gm3.metric("Current vol percentile", f"P{g.vol_percentile:.0f}")
                gm4.metric("Volatility regime", g.regime_label.upper())

                # ----- View 1: Fan chart -----
                st.markdown("##### View 1 — Volatility fan chart")
                st.caption("Forecast price (using ARIMAX point forecast as centre) with GARCH-derived uncertainty bands.")

                # Use ARIMAX forecast as centre (already cached)
                try:
                    arx_for_fan = cached_arimax(region, region_pick, ar_arx, d_arx, ma_arx,
                                                  forecast_horizon, drivers_tuple, file_mtime)
                except Exception:
                    arx_for_fan = None

                if arx_for_fan and arx_for_fan.success and arx_for_fan.forecast_mean is not None:
                    target = region.y
                    point_fc = arx_for_fan.forecast_mean

                    # Build fan: 1, 2, 3 sigma bands using GARCH forecast vol
                    fan_fig = go.Figure()
                    fan_fig.add_trace(go.Scatter(
                        x=target.index[-36:], y=target.values[-36:],
                        mode="lines",
                        line=dict(color=COLORS["ink"], width=1.5),
                        name="Actual"
                    ))
                    # Compute fan bands
                    for k_sigma, alpha_fill in [(3, 0.10), (2, 0.18), (1, 0.30)]:
                        # cumulative vol grows with sqrt(h) — multiply by sqrt(h)
                        horizon_steps = np.arange(1, len(point_fc) + 1)
                        vol_h = g.forecast_vol.values[: len(point_fc)] / 100.0 # to decimal
                        # The vol forecast is for log-returns; scale to price space
                        last_price = float(target.iloc[-1])
                        band_pct = k_sigma * vol_h * np.sqrt(horizon_steps)
                        upper = point_fc.values * np.exp(band_pct)
                        lower = point_fc.values * np.exp(-band_pct)
                        fan_fig.add_trace(go.Scatter(
                            x=point_fc.index, y=upper,
                            mode="lines", line=dict(width=0), showlegend=False,
                            hoverinfo="skip"
                        ))
                        fan_fig.add_trace(go.Scatter(
                            x=point_fc.index, y=lower,
                            mode="lines", line=dict(width=0),
                            fill="tonexty",
                            fillcolor=f"rgba(31, 78, 121, {alpha_fill})",
                            name=f"±{k_sigma}σ band",
                        ))
                    fan_fig.add_trace(go.Scatter(
                        x=point_fc.index, y=point_fc.values,
                        mode="lines",
                        line=dict(color=COLORS["accent"], width=2.5),
                        name="Point forecast"
                    ))
                    fan_fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=440,
                                            yaxis_title=f"{currency}/t",
                                            title=dict(text="Forecast with GARCH-derived uncertainty bands",
                                                         font=dict(size=13)))
                    style_axes(fan_fig)
                    st.plotly_chart(fan_fig, use_container_width=True)
                    st.caption(
                        "Fan widths: 1σ (~68% probability), 2σ (~95%), 3σ (~99.7%). "
                        "Bands grow with horizon because cumulative uncertainty compounds with √h."
                    )
                else:
                    st.warning("ARIMAX forecast required for fan chart but failed — try simpler ARIMAX parameters.")

                # ----- View 2: Risk metrics -----
                st.markdown("##### View 2 — Risk metrics (1-month forward)")
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("VaR 95%", f"{currency} {g.var_95:,.0f}/t",
                             help="5% probability of losing MORE than this in 1 month.")
                rm2.metric("VaR 99%", f"{currency} {g.var_99:,.0f}/t",
                             help="1% probability of losing MORE than this in 1 month.")
                rm3.metric("Expected Shortfall 95%", f"{currency} {g.expected_shortfall_95:,.0f}/t",
                             help="Average loss IF the 5% tail event occurs.")
                rm4.metric("Expected Shortfall 99%", f"{currency} {g.expected_shortfall_99:,.0f}/t",
                             help="Average loss IF the 1% tail event occurs.")
                st.caption(
                    "VaR (Value-at-Risk) and ES (Expected Shortfall) are computed from the GARCH-forecast "
                    "volatility under a normal distribution assumption. ES is generally more honest than "
                    "VaR because it captures the magnitude of tail losses, not just their threshold."
                )

                # ----- View 3: Volatility regime -----
                st.markdown("##### View 3 — Volatility regime indicator")

                # Build a visualization showing current vol vs full historical distribution
                vol_history = g.conditional_vol
                vol_levels = sorted(vol_history.values)
                n = len(vol_levels)

                # Plot histogram + current marker
                rg_fig = go.Figure()
                rg_fig.add_trace(go.Histogram(
                    x=vol_history.values,
                    nbinsx=40,
                    marker_color=COLORS["muted"],
                    opacity=0.7,
                    name="Historical vol"
                ))
                rg_fig.add_vline(
                    x=g.current_vol,
                    line=dict(color=COLORS["danger"], width=3, dash="dash"),
                    annotation_text=f"Current: P{g.vol_percentile:.0f}",
                    annotation_position="top",
                )
                # Regime band shading
                p25 = float(np.percentile(vol_history.values, 25))
                p65 = float(np.percentile(vol_history.values, 65))
                p90 = float(np.percentile(vol_history.values, 90))
                rg_fig.add_vrect(x0=0, x1=p25, fillcolor="green", opacity=0.05, line_width=0,
                                    annotation_text="LOW", annotation_position="top left")
                rg_fig.add_vrect(x0=p25, x1=p65, fillcolor="grey", opacity=0.05, line_width=0,
                                    annotation_text="NORMAL", annotation_position="top left")
                rg_fig.add_vrect(x0=p65, x1=p90, fillcolor="orange", opacity=0.10, line_width=0,
                                    annotation_text="ELEVATED", annotation_position="top left")
                rg_fig.add_vrect(x0=p90, x1=max(vol_history.values) * 1.05,
                                    fillcolor="red", opacity=0.15, line_width=0,
                                    annotation_text="EXTREME", annotation_position="top left")
                rg_fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=350,
                                        xaxis_title="Conditional volatility (% / period)",
                                        yaxis_title="Frequency",
                                        showlegend=False,
                                        title=dict(text="Current volatility regime in historical context",
                                                     font=dict(size=13)))
                style_axes(rg_fig)
                st.plotly_chart(rg_fig, use_container_width=True)

                regime_msg = {
                    "low": "Volatility is unusually subdued. Risk models calibrated here will materially under-estimate stress-period tail risk.",
                    "normal": "Volatility is in its typical operating range. Standard risk parameters apply.",
                    "elevated": "Volatility is elevated above norms. Position-sizing should be reduced; expect larger price swings.",
                    "extreme": "Volatility is in the top decile of historical experience. This is a stress regime — exercise maximum caution on directional positions."
                }.get(g.regime_label, "")
                if regime_msg:
                    st.info(f"**Regime interpretation:** {regime_msg}")


# ===== MACRO CALENDAR =====
with tab_macro:
    # Window controls
    cwl1, cwl2, cwl3 = st.columns([1, 1, 2])
    with cwl1:
        window_days = st.slider("Forward window (days)", 15, 90, 45, step=15,
                                  help="Window length from today.")
    with cwl2:
        show_past = st.toggle("Include past events", value=False,
                                help="Also show events from the recent past.")
        past_days = 30
        if show_past:
            past_days = st.slider("Past days", 7, 90, 30, step=7,
                                    label_visibility="collapsed")
    with cwl3:
        if show_past:
            st.caption(f"Showing **{past_days} past days** + **{window_days} forward days** "
                        f"from today ({pd.Timestamp.now().strftime('%b %d, %Y')}).")
        else:
            st.caption(f"Showing events from **{pd.Timestamp.now().strftime('%b %d, %Y')}** "
                        f"forward **{window_days} days**. Toggle 'Include past events' "
                        f"to see recent history.")

    try:
        cal = cached_macro_calendar(dataset, config_str, window_days,
                                       show_past, past_days, file_mtime)
    except Exception as e:
        st.error(f"Failed to load macro calendar: {type(e).__name__}: {e}")
        st.info("Make sure `data/macro_calendar.yaml` exists.")
        cal = None

    if cal is None:
        st.warning("Macro calendar unavailable — China region required.")
    elif len(cal.events) == 0:
        st.warning(f"No events found in the next {window_days} days. "
                    f"Add events to `data/macro_calendar.yaml`. "
                    f"Library currently has {cal.n_total_in_library} total events "
                    f"(all outside the current window).")
    else:
        st.markdown(f"#### {cal.window_description}")
        st.caption(f"Window: **{cal.window_start}** to **{cal.window_end}** ·  "
                    f"Library: {cal.n_total_in_library} total events ·  "
                    f"Filtered to window: {len(cal.events)}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events in window", len(cal.events),
                   f"of {cal.n_total_in_library} in library")
        c2.metric("HIGH impact", cal.n_high)
        c3.metric("MED impact", cal.n_med)
        c4.metric("Iron ore now",
                   f"P{cal.events[0].iron_ore_percentile:.0f}"
                   if cal.events[0].iron_ore_percentile else "—",
                   f"{cal.events[0].spread_regime}".upper())

        # Filter controls
        st.markdown("---")
        f1, f2 = st.columns([1, 2])
        with f1:
            filter_impact = st.multiselect(
                "Filter by impact",
                options=["HIGH", "MED", "LOW"],
                default=["HIGH", "MED", "LOW"],
            )
        with f2:
            filter_country = st.multiselect(
                "Filter by country",
                options=sorted(set(e.country for e in cal.events)),
                default=sorted(set(e.country for e in cal.events)),
            )

        filtered_events = [
            e for e in cal.events
            if e.impact in filter_impact and e.country in filter_country
        ]

        # Live interpretation — only intro paragraphs (per-event cards already render below)
        try:
            intro_blocks = list(narrator.narrate_macro_calendar_intro(cal))
            past_msg = f", incl. {past_days}d past" if show_past else ""
            render_interpretation(
                intro_blocks,
                label="Calendar overview interpretation",
                settings_summary=f"Window: {window_days}d forward{past_msg}, "
                                  f"{len(filtered_events)} events shown",
            )
        except Exception:
            pass

        st.markdown("---")
        st.markdown(f"**Showing {len(filtered_events)} of {len(cal.events)} events.** "
                    f"Each event card includes the transmission mechanism to HRC, "
                    f"expected reaction by region, and historical analogues from the data.")

        # Render each event as a styled container
        IMPACT_COLOR = {"HIGH": "#A4161A", "MED": "#C9540F", "LOW": "#5C6B7F"}
        IMPACT_BG = {"HIGH": "#FEE2E2", "MED": "#FED7AA", "LOW": "#F0F4F8"}
        CONF_COLOR = {"high": "#2D6A4F", "medium": "#C9540F", "low": "#A4161A", "none": "#5C6B7F"}

        for ev in filtered_events:
            impact_c = IMPACT_COLOR.get(ev.impact, "#5C6B7F")
            impact_bg = IMPACT_BG.get(ev.impact, "#F0F4F8")
            conf_c = CONF_COLOR.get(ev.analogue_confidence, "#5C6B7F")

            # Card container with colored left border
            st.markdown(
                f"""<div style="background:white; border:1px solid #E5E9F0;
                              border-left:4px solid {impact_c};
                              border-radius:6px; padding:14px 18px; margin:14px 0;">
                <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                    <span style="background:#1A1F2E; color:white; padding:3px 8px;
                                 border-radius:3px; font-size:0.75rem;
                                 font-weight:600; font-family:monospace;">{ev.event_date}</span>
                    <span style="background:{impact_bg}; color:{impact_c};
                                 padding:2px 8px; border-radius:3px;
                                 font-size:0.7rem; font-weight:700;">{ev.impact}</span>
                    <span style="color:#5C6B7F; font-size:0.75rem;
                                 background:#F0F4F8; padding:2px 8px;
                                 border-radius:3px;">{ev.country}</span>
                    <span style="color:{conf_c}; font-size:0.7rem; margin-left:auto;">
                        analogues: {ev.analogue_confidence} ({ev.cond_n} matches)
                    </span>
                </div>
                <div style="font-size:1.0rem; font-weight:600; margin:8px 0 4px;
                            color:#1A1F2E;">{ev.name}</div>
                <div style="font-size:0.8rem; color:#5C6B7F;">
                    {ev.days_until} days from today ·  {ev.consensus}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Use expander for the detail content
            with st.expander("Details: mechanism, reaction, analogues", expanded=False):
                st.markdown(f"**Affects:** {', '.join(r.title() for r in ev.affects_regions)} ·  "
                            f"**Channel:** {ev.primary_channel.replace('_', ' ').title()}")
                st.markdown(f"**Mechanism:**")
                st.write(ev.mechanism)
                st.markdown(f"**Expected HRC reaction:**")
                st.write(ev.expected_hrc_reaction)

                # Analogue summary
                st.markdown(f"**Historical analogues** ({ev.analogue_confidence} confidence)")
                if ev.cond_n >= 3 and ev.cond_avg_30d is not None:
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Avg 30d HRC move", f"{ev.cond_avg_30d:+.1f}%",
                                f"{ev.cond_n} matches")
                    a2.metric("Avg 60d HRC move", f"{ev.cond_avg_60d:+.1f}%")
                    a3.metric("Avg 90d HRC move", f"{ev.cond_avg_90d:+.1f}%")
                    st.caption(f"Based on past months with iron ore at "
                                f"P{ev.iron_ore_percentile:.0f} ±20 and "
                                f"{ev.spread_regime} spread regime.")
                elif ev.simple_n >= 6 and ev.simple_avg_30d is not None:
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Avg 30d HRC drift", f"{ev.simple_avg_30d:+.1f}%",
                                "broad sample")
                    a2.metric("Avg 60d HRC drift", f"{ev.simple_avg_60d:+.1f}%")
                    a3.metric("Avg 90d HRC drift", f"{ev.simple_avg_90d:+.1f}%")
                    st.caption(f"Limited setup-conditional matches; showing broader "
                                f"sample of {ev.simple_n} recent months.")
                else:
                    st.info("Insufficient analogues — rely on mechanism above for guidance.")


# ===== LIQUIDITY TAB (India only) =====
if tab_liquidity is not None:
    with tab_liquidity:
        try:
            from pipeline.liquidity import (
                regime_performance, liquidity_lead_lag,
                summarize_current_state, interpret_current_state, regime_periods,
            )

            rdf = region.df # convenience alias

            st.subheader("India Liquidity Monitor")
            st.markdown(
                "Banking-system liquidity conditions, RBI monetary policy stance, and how they "
                "transmit into Indian HRC prices. Liquidity tightness affects steel demand with "
                "a typical lag of **2–5 months** via working-capital financing for distributors "
                "and downstream consumers."
            )

            # ---- Defensive helpers ----
            def _safe_num(v, default=np.nan):
                try:
                    f = float(v)
                    return f if not np.isnan(f) else default
                except (TypeError, ValueError):
                    return default

            def _fmt_bps(spread_pct):
                """Convert a spread expressed in % to a +/- bps string. NaN-safe."""
                v = _safe_num(spread_pct)
                if np.isnan(v):
                    return "—"
                return f"{v*100:+.0f} bps"

            # ---- Current-state header (each metric guarded individually) ----
            state = summarize_current_state(rdf) or {}

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                regime = state.get('Liquidity_Regime', '—')
                if regime is None or (isinstance(regime, float) and np.isnan(regime)):
                    regime = '—'
                st.metric("Liquidity Regime", str(regime))

            with c2:
                spread = _safe_num(state.get('WACR_Spread'))
                spread_chg = _safe_num(state.get('WACR_Spread_3m_change'), default=0.0)
                if not np.isnan(spread):
                    st.metric(
                        "WACR Spread",
                        f"{spread*100:+.0f} bps",
                        delta=f"{spread_chg*100:+.0f} bps (3m)" if not np.isnan(spread_chg) else None,
                    )
                else:
                    st.metric("WACR Spread", "—")

            with c3:
                stress = _safe_num(state.get('Stress_Index'))
                stress_chg = _safe_num(state.get('Stress_Index_3m_change'), default=0.0)
                if not np.isnan(stress):
                    st.metric(
                        "Stress Index",
                        f"{stress:.0f} / 100",
                        delta=f"{stress_chg:+.1f} (3m)" if not np.isnan(stress_chg) else None,
                        delta_color="inverse",
                    )
                else:
                    st.metric("Stress Index", "—")

            with c4:
                policy = state.get('Policy_Regime', '—')
                if policy is None or (isinstance(policy, float) and np.isnan(policy)):
                    policy = '—'
                st.metric("RBI Policy", str(policy))

            # Interpretation paragraph
            try:
                interp_text = interpret_current_state(state)
                if interp_text:
                    st.info(interp_text)
            except Exception as e:
                st.caption(f"(interpretation unavailable: {type(e).__name__})")

            # Quick-read details card
            try:
                bc_yoy = _safe_num(state.get('Bank_Credit_YoY'))
                gs10 = _safe_num(state.get('GSec_10Y'))
                repo = _safe_num(state.get('Repo_Rate'))
                term_premium_bps = (gs10 - repo) * 100 if not (np.isnan(gs10) or np.isnan(repo)) else np.nan
                quick_parts = []
                if not np.isnan(bc_yoy):
                    quick_parts.append(f"Bank credit is growing at **{bc_yoy:.1f}% YoY**")
                if not np.isnan(gs10) and not np.isnan(repo):
                    quick_parts.append(
                        f"the 10Y G-Sec yield sits at **{gs10:.2f}%** versus a policy repo of "
                        f"**{repo:.2f}%**, giving a term premium of **{term_premium_bps:.0f} bps**"
                    )
                if quick_parts:
                    with st.expander("Quick read on India liquidity", expanded=False):
                        st.markdown("; ".join(quick_parts) + ".")
            except Exception as e:
                st.caption(f"(quick read unavailable: {type(e).__name__})")

            st.markdown("---")

            # ---- 1. STRESS GAUGE ----
            st.markdown("### Liquidity Stress Gauge")
            try:
                stress_val = float(state.get('Stress_Index', 50.0))
                if np.isnan(stress_val):
                    stress_val = 50.0
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=stress_val,
                    number={'suffix': " / 100", 'font': {'size': 28}},
                    delta={'reference': 50,
                           'increasing': {'color': COLORS["danger"]},
                           'decreasing': {'color': COLORS["accent2"]}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1,
                                 'tickcolor': COLORS["rule"]},
                        'bar': {'color': COLORS["ink"], 'thickness': 0.18},
                        'bgcolor': "white",
                        'borderwidth': 1, 'bordercolor': COLORS["rule"],
                        'steps': [
                            {'range': [0, 30], 'color': '#C7E8D2'},
                            {'range': [30, 50], 'color': '#E6F1E8'},
                            {'range': [50, 65], 'color': '#F5F1E5'},
                            {'range': [65, 80], 'color': '#F7DCC2'},
                            {'range': [80, 100], 'color': '#F2BFC0'},
                        ],
                        'threshold': {
                            'line': {'color': COLORS["danger"], 'width': 3},
                            'thickness': 0.85, 'value': stress_val,
                        },
                    },
                ))
                gauge.update_layout(**PLOT_BASE, height=280,
                                    margin=dict(l=20, r=20, t=20, b=10))
                st.plotly_chart(gauge, use_container_width=True)
                st.caption(
                    "**0–30** Abundant liquidity · **30–50** Accommodative · "
                    "**50–65** Mildly elevated · **65–80** Elevated stress · "
                    "**80–100** Severe stress. Composite of WACR spread, term premium, "
                    "credit-growth slowdown, and 6m repo change."
                )

                # Stress gauge interpretation expander
                with st.expander("Interpretation: what the stress score means today", expanded=False):
                    _s = stress_val
                    if _s < 30:
                        _band = "abundant liquidity"
                        _msg = ("Funding conditions are very loose. Distributors and steel consumers "
                                "have easy access to working capital. Historically, this band has "
                                "preceded strong industrial activity and firm HRC demand.")
                    elif _s < 50:
                        _band = "accommodative liquidity"
                        _msg = ("Conditions are below-average tight. The RBI is not actively draining "
                                "liquidity, and the call money market is functioning smoothly. Steel "
                                "demand should be supported, particularly in construction and auto.")
                    elif _s < 65:
                        _band = "mildly elevated stress"
                        _msg = ("Liquidity is slightly tighter than average. Not stressed, but worth "
                                "monitoring — if the index continues to rise, financing costs for "
                                "downstream steel buyers will begin to bite within 2–3 months.")
                    elif _s < 80:
                        _band = "elevated stress"
                        _msg = ("Banking-system liquidity is materially tight. Distributors face higher "
                                "carrying costs, which historically translates into inventory destocking "
                                "and price weakness in HRC within 2–5 months.")
                    else:
                        _band = "severe stress"
                        _msg = ("Liquidity is in the high-stress band. Historically this has coincided "
                                "with sharp drawdowns in Indian HRC as financing-constrained buyers "
                                "delay purchases and unwind inventory positions.")
                    st.markdown(
                        f"The current stress reading of **{_s:.0f}/100** falls in the **{_band}** band. "
                        f"{_msg}"
                    )

            except Exception as e:
                st.warning(f"Stress gauge could not render: {type(e).__name__}: {e}")

            st.markdown("---")

            # ---- 2. HRC vs LIQUIDITY REGIME CHART ----
            st.markdown(f"### {region.target} vs Liquidity Regime")
            try:
                hrc = rdf[region.target].dropna()
                spread_series = rdf['WACR_Spread'].dropna() if 'WACR_Spread' in rdf.columns else pd.Series(dtype=float)
                repo_series = rdf['Repo_Rate'].dropna() if 'Repo_Rate' in rdf.columns else pd.Series(dtype=float)
                regime_series = rdf['Liquidity_Regime'].dropna() if 'Liquidity_Regime' in rdf.columns else pd.Series(dtype=object)

                REGIME_COLOURS = {
                    'Surplus': 'rgba(45, 106, 79, 0.28)',
                    'Neutral': 'rgba(150, 150, 150, 0.18)',
                    'Tight':   'rgba(220, 50, 50, 0.22)',
                }
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Shaded regime backgrounds
                if not regime_series.empty:
                    blocks = regime_periods(regime_series)
                    for b in blocks:
                        fig.add_vrect(
                            x0=b['start'], x1=b['end'],
                            fillcolor=REGIME_COLOURS.get(b['regime'], 'rgba(0,0,0,0.05)'),
                            line_width=0, layer="below",
                        )

                # HRC line
                fig.add_trace(
                    go.Scatter(x=hrc.index, y=hrc.values, name=f"HRC ({currency}/t)",
                               mode="lines", line=dict(color=COLORS["accent"], width=2.2),
                               hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"),
                    secondary_y=False,
                )
                # WACR Spread in bps
                if not spread_series.empty:
                    fig.add_trace(
                        go.Scatter(x=spread_series.index, y=spread_series.values * 100,
                                   name="WACR − Repo (bps)",
                                   mode="lines",
                                   line=dict(color=COLORS["danger"], width=1.6, dash="dot"),
                                   hovertemplate="%{x|%b %Y}: %{y:+.0f} bps<extra></extra>"),
                        secondary_y=True,
                    )
                # Repo line (hidden by default)
                if not repo_series.empty:
                    fig.add_trace(
                        go.Scatter(x=repo_series.index, y=repo_series.values * 100,
                                   name="Repo Rate (bps)",
                                   mode="lines",
                                   line=dict(color=COLORS["muted"], width=1.2, dash="dash"),
                                   hovertemplate="%{x|%b %Y}: %{y:.0f} bps<extra></extra>",
                                   visible="legendonly"),
                        secondary_y=True,
                    )
                fig.update_yaxes(title_text=f"HRC ({currency}/t)", secondary_y=False)
                fig.update_yaxes(title_text="bps (spread / rate)", secondary_y=True,
                                 showgrid=False)
                fig.update_layout(**PLOT_BASE, height=480, margin=DEFAULT_MARGIN,
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="left", x=0))
                style_axes(fig)
                st.plotly_chart(fig, use_container_width=True)

                leg1, leg2, leg3 = st.columns(3)
                leg1.markdown("**Green** = Surplus liquidity (WACR below Repo)")
                leg2.markdown("**Grey** = Neutral")
                leg3.markdown("**Red** = Tight liquidity (WACR above Repo)")

                # Regime chart interpretation expander
                with st.expander("Interpretation: how to read this chart", expanded=False):
                    st.markdown(
                        "Shaded backgrounds mark each month's liquidity regime, classified by where "
                        "the WACR-Repo spread sits relative to its historical mean band:\n\n"
                        "- **Green stretches (Surplus)** correspond to periods when banking-system "
                        "liquidity was abundant — WACR traded below the policy repo, the RBI was "
                        "actively injecting liquidity, and credit growth was strong. The 2020–2022 "
                        "COVID stretch is the most visible example: WACR fell up to 86 bps below the "
                        "policy rate as the RBI flooded the system, and Indian HRC prices rallied "
                        "through this period (the supercycle).\n\n"
                        "- **Red stretches (Tight)** mark RBI tightening cycles. The Oct 2022 – Mar 2025 "
                        "stretch coincided with the post-pandemic rate hikes (Repo from 4.00% to 6.50%) "
                        "and the HRC correction.\n\n"
                        "- **Grey (Neutral)** is everything in between.\n\n"
                        "The transmission is asymmetric: surplus liquidity helps HRC almost immediately, "
                        "while tight liquidity hurts with a 2–5 month lag (working-capital pain takes "
                        "time to compress demand). Look for the *change* in regime as the early signal, "
                        "not the level."
                    )
            except Exception as e:
                st.warning(f"Regime chart could not render: {type(e).__name__}: {e}")

            st.markdown("---")

            # ---- 3. REGIME PERFORMANCE TABLE ----
            st.markdown("### HRC Performance by Liquidity Regime")
            try:
                hrc = rdf[region.target].dropna()
                perf = regime_performance(hrc, rdf['Liquidity_Regime'])
                if perf is not None and not perf.empty:
                    st.dataframe(
                        perf.style.format({
                            'Avg_Return_MoM_%': '{:+.2f}',
                            'Annualized_Return_%': '{:+.1f}',
                            'Volatility_Ann_%': '{:.1f}',
                            'Max_Drawdown_%': '{:.1f}',
                            'Hit_Rate_%': '{:.0f}',
                        }).background_gradient(
                            cmap='RdYlGn',
                            subset=['Annualized_Return_%'],
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Returns are month-over-month on the HRC price level, annualised by "
                        "compounding. Max drawdown shows the deepest peak-to-trough decline of the "
                        "cumulative return within each regime."
                    )

                    # Regime performance interpretation expander
                    with st.expander("Interpretation: what the table says about Indian HRC", expanded=False):
                        bullets = []
                        if 'Surplus' in perf.index:
                            s = perf.loc['Surplus']
                            bullets.append(
                                f"- **Surplus regimes ({int(s['Months'])} months)** — HRC delivered "
                                f"an annualised return of **{s['Annualized_Return_%']:+.1f}%**, "
                                f"with volatility {s['Volatility_Ann_%']:.1f}% and a hit rate of "
                                f"{s['Hit_Rate_%']:.0f}%. Surplus liquidity is the most favourable "
                                f"regime for HRC."
                            )
                        if 'Neutral' in perf.index:
                            s = perf.loc['Neutral']
                            bullets.append(
                                f"- **Neutral regimes ({int(s['Months'])} months)** — annualised "
                                f"return **{s['Annualized_Return_%']:+.1f}%**, volatility "
                                f"{s['Volatility_Ann_%']:.1f}%."
                            )
                        if 'Tight' in perf.index:
                            s = perf.loc['Tight']
                            bullets.append(
                                f"- **Tight regimes ({int(s['Months'])} months)** — annualised "
                                f"return **{s['Annualized_Return_%']:+.1f}%**, max drawdown "
                                f"{s['Max_Drawdown_%']:.1f}%, hit rate {s['Hit_Rate_%']:.0f}%. "
                                f"Tight liquidity is the most damaging regime."
                            )
                        st.markdown("\n".join(bullets))

                        # Headline conclusion (gap between surplus and tight, if both present)
                        if 'Surplus' in perf.index and 'Tight' in perf.index:
                            gap = perf.loc['Surplus', 'Annualized_Return_%'] - perf.loc['Tight', 'Annualized_Return_%']
                            st.markdown(
                                f"\n**Headline:** the gap between surplus and tight regimes is "
                                f"**{gap:+.1f} percentage points** of annualised return. This is the "
                                f"liquidity-cycle premium in Indian HRC — material, persistent, and "
                                f"asymmetric. The empirical case for tracking RBI liquidity as a "
                                f"first-order driver of steel prices in India is supported by this "
                                f"data."
                            )
                else:
                    st.info("Not enough data to compute regime performance.")
            except Exception as e:
                st.warning(f"Regime performance table could not render: {type(e).__name__}: {e}")

            st.markdown("---")

            # ---- 4. LEAD-LAG TABLE ----
            st.markdown("### Lead-Lag: Does Liquidity Lead HRC?")
            try:
                lag_var_candidates = ['WACR_Spread', 'GSec_Repo_Spread', 'Stress_Index',
                                      'Bank_Credit_YoY', 'Repo_6M_Change']
                lag_vars = [c for c in lag_var_candidates if c in rdf.columns]
                if lag_vars:
                    hrc = rdf[region.target].dropna()
                    ll = liquidity_lead_lag(hrc, rdf[lag_vars], lags=[0, 1, 3, 6, 12])
                    st.dataframe(
                        ll.style.background_gradient(cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                                                     axis=None).format('{:+.2f}'),
                        use_container_width=True,
                    )
                    st.caption(
                        "Each cell is the Pearson correlation between HRC month-over-month log "
                        "returns and a liquidity variable lagged by k months. A positive lag means "
                        "liquidity from k months ago is being matched against HRC return today. "
                        "Negative values for WACR_Spread or Stress_Index mean tight liquidity "
                        "precedes weaker HRC."
                    )

                    # Lead-lag interpretation expander
                    with st.expander("Interpretation: which liquidity signals lead HRC?", expanded=False):
                        try:
                            # Find the strongest absolute correlation per variable
                            findings = []
                            for var in ll.index:
                                row = ll.loc[var]
                                idx_max = row.abs().idxmax()
                                lag_k = int(idx_max.replace('lag_', '').replace('m', ''))
                                corr = row[idx_max]
                                if abs(corr) >= 0.15: # only report meaningful signals
                                    direction = "leads" if lag_k > 0 else "is contemporaneous with" if lag_k == 0 else "lags"
                                    sign = "negative" if corr < 0 else "positive"
                                    findings.append(
                                        f"- **{var}**: strongest correlation **{corr:+.2f}** at "
                                        f"`lag_{lag_k}m`. This {sign} correlation means {var} "
                                        f"{direction} HRC by {abs(lag_k)} month{'s' if abs(lag_k) != 1 else ''}."
                                    )
                            if findings:
                                st.markdown("\n".join(findings))
                            else:
                                st.markdown(
                                    "No liquidity variable shows a correlation with HRC returns above "
                                    "0.15 at any tested lag. The liquidity signal is currently weak; "
                                    "drivers other than monetary conditions are dominating HRC."
                                )

                            st.markdown(
                                "\n**Interpreting the signs:**\n"
                                "- **WACR_Spread, Stress_Index, GSec_Repo_Spread**: a *negative* "
                                "correlation at a *positive* lag means tighter liquidity preceded "
                                "weaker HRC — the textbook macro channel.\n"
                                "- **Bank_Credit_YoY**: typically positive (faster credit growth → "
                                "stronger demand → higher HRC), though contemporaneous credit growth "
                                "often correlates negatively in India because credit accelerates "
                                "*after* a demand pulse has already moved prices.\n"
                                "- **Repo_6M_Change**: a negative correlation at long lags (12m) "
                                "captures the slow transmission of RBI rate decisions through to "
                                "downstream financing costs and steel demand."
                            )
                        except Exception:
                            st.markdown(
                                "Interpretation could not be auto-generated. Look for the cells with "
                                "the strongest colors (red = negative, blue = positive) in the table "
                                "above — those are the lags at which liquidity variables most strongly "
                                "anticipate HRC returns."
                            )
                else:
                    st.info("No liquidity variables available for lead-lag analysis.")
            except Exception as e:
                st.warning(f"Lead-lag table could not render: {type(e).__name__}: {e}")

            st.markdown("---")

            # ---- 5. LIQUIDITY VARIABLE PANEL ----
            st.markdown("### Liquidity Variable Panel")
            st.caption("All base and derived liquidity series at a glance.")
            try:
                panel_cfg = [
                    ('WACR', 'WACR (%)', COLORS["accent"]),
                    ('Repo_Rate', 'Policy Repo Rate (%)', COLORS["danger"]),
                    ('GSec_10Y', '10Y G-Sec Yield (%)', COLORS["accent2"]),
                    ('CRR', 'CRR (%)', COLORS["muted"]),
                    ('WACR_Spread', 'WACR minus Repo Spread (%)', COLORS["warning"]),
                    ('Bank_Credit_YoY', 'Bank Credit YoY %', CHART_PALETTE[3]),
                ]
                panel_cfg = [(c, lbl, clr) for (c, lbl, clr) in panel_cfg if c in rdf.columns]

                n_cols = 2
                for i in range(0, len(panel_cfg), n_cols):
                    cols = st.columns(n_cols)
                    for j, col in enumerate(cols):
                        if i + j >= len(panel_cfg):
                            continue
                        cname, label, clr = panel_cfg[i + j]
                        s = rdf[cname].dropna()
                        if s.empty:
                            continue
                        with col:
                            sub = go.Figure()
                            sub.add_trace(go.Scatter(
                                x=s.index, y=s.values, mode="lines",
                                line=dict(color=clr, width=1.7),
                                hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
                            if cname == 'WACR_Spread':
                                sub.add_hline(y=0, line_dash="dot",
                                              line_color=COLORS["muted"])
                            sub.update_layout(**PLOT_BASE, margin=COMPACT_MARGIN,
                                              height=230, title=label)
                            style_axes(sub)
                            st.plotly_chart(sub, use_container_width=True)
            except Exception as e:
                st.warning(f"Variable panel could not render: {type(e).__name__}: {e}")

        except Exception as e:
            import traceback
            st.error(
                f"The Liquidity tab encountered a fatal error and could not render.\n\n"
                f"**{type(e).__name__}**: {e}\n\n"
                f"```\n{traceback.format_exc()}\n```\n\n"
                f"This usually means the India sheet is missing required liquidity columns. "
                f"Check the India sheet in `Raw_data.xlsx` has these columns: "
                f"WACR, Repo_Rate, GSec_10Y, CRR, Bank_Credit."
            )


# ===== GLOSSARY TAB (disabled; sidebar search only) =====
# Kept as guarded dead code so the rich content (87 entries, category browser) can
# be re-enabled by simply restoring the tab in the st.tabs() declaration above.
if tab_glossary is not None:
    with tab_glossary:
        st.markdown("### Glossary")
        st.caption(
            "Reference for every term used in the dashboard. Search by typing in the "
            "sidebar search box, or browse by category below. Click any entry to expand."
        )

        glossary = load_glossary()
        if not glossary:
            st.warning("Glossary not loaded — check that `data/glossary.yaml` exists.")
        else:
            # Top filter bar
            gc1, gc2 = st.columns([2, 3])
            with gc1:
                cat_filter = st.multiselect(
                    "Filter by category",
                    options=list(GLOSSARY_CATEGORIES.keys()),
                    default=list(GLOSSARY_CATEGORIES.keys()),
                    format_func=lambda c: f"{GLOSSARY_CATEGORIES[c][0]} {GLOSSARY_CATEGORIES[c][1]}",
                    key="glossary_cat_filter",
                )
            with gc2:
                tab_search = st.text_input(
                    "Search within glossary",
                    placeholder="Type to filter (term, full name, or definition)...",
                    key="glossary_tab_search",
                )

            # Filter entries
            all_entries = list(glossary.values())
            filtered = [
                e for e in all_entries
                if e.get("category") in cat_filter
            ]
            if tab_search:
                q = tab_search.strip().lower()
                filtered = [
                    e for e in filtered
                    if (q in e["term"].lower()
                        or q in e.get("full_name", "").lower()
                        or q in e.get("short", "").lower()
                        or q in e.get("detail", "").lower())
                ]

            # Counter + sort
            st.caption(f"Showing {len(filtered)} of {len(all_entries)} entries")
            st.markdown("---")

            # Group by category
            for cat_key, (icon, cat_label) in GLOSSARY_CATEGORIES.items():
                cat_entries = sorted(
                    [e for e in filtered if e.get("category") == cat_key],
                    key=lambda x: x["term"].lower()
                )
                if not cat_entries:
                    continue

                st.markdown(f"#### {icon} {cat_label} · {len(cat_entries)} entries")
                # Render in a 2-col grid for compactness
                cols = st.columns(2)
                for i, entry in enumerate(cat_entries):
                    with cols[i % 2]:
                        with st.expander(f"**{entry['term']}**" +
                                           (f" — {entry.get('full_name', '')}" if entry.get('full_name') else "")):
                            st.markdown(f"*{entry.get('short', '')}*")
                            if entry.get("detail"):
                                st.markdown(entry["detail"].strip())
                            if entry.get("see_also"):
                                valid = [t for t in entry["see_also"] if t.lower() in glossary]
                                if valid:
                                    st.caption(f"See also: {' · '.join(valid)}")
                st.markdown("") # spacing between categories


    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.caption("**Workflow**")
    st.sidebar.caption("1. Edit `data/Raw_data.xlsx`")
    st.sidebar.caption("2. Click Refresh data above")
    st.sidebar.caption("3. To regenerate the static report: `python3 run.py`")
