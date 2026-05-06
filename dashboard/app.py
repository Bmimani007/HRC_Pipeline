"""
HRC Steel Pipeline — Live Dashboard

Run with:    streamlit run dashboard/app.py

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
    page_icon="📊",
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
      • Local:  create .streamlit/secrets.toml with: app_password = "your-pw"
      • Cloud:  add app_password under "Secrets" in the Streamlit Cloud
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


@st.cache_data(show_spinner="Running walk-forward backtest (30-90s)...")
def cached_backtest(_region, region_key: str, model_type: str,
                       ar: int, d: int, ma: int, dl: int,
                       horizon: int, drivers_tuple: tuple,
                       min_train: int, step: int, file_mtime: float):
    from pipeline.forecasting import walk_forward_backtest
    drivers = list(drivers_tuple) if drivers_tuple else None
    return walk_forward_backtest(_region.y, _region.X, model_type=model_type,
                                    ar=ar, d=d, ma=ma, dl=dl,
                                    drivers=drivers,
                                    forecast_horizon=horizon,
                                    min_train_months=min_train,
                                    step_months=step)


@st.cache_data(show_spinner="Fitting GARCH + risk metrics...")
def cached_garch(_region, region_key: str, horizon: int, file_mtime: float):
    from pipeline.forecasting import fit_garch_with_risk
    return fit_garch_with_risk(_region.y, horizon=horizon)


# ---------- Render helper: convert narrator blocks to Streamlit HTML ----------
def render_interpretation(blocks, label: str = "📝 Interpretation",
                            settings_summary: str = None):
    """
    Render narrator output as a collapsible expander. Default collapsed.
    blocks: list of strings (paragraphs) and dicts ({'_kind': 'key_interpretation'} etc.)
    settings_summary: optional one-line summary of current control values.
    """
    with st.expander(label, expanded=False):
        if settings_summary:
            st.caption(f"⚙ {settings_summary}")
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
st.sidebar.markdown("**📄 HTML Report**")
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
        label="⬇ Download HTML report",
        data=st.session_state["report_bytes"],
        file_name=f"HRC_Steel_Report_{timestamp}.html",
        mime="text/html",
        use_container_width=True,
    )
    st.sidebar.caption(f"Built at {st.session_state.get('report_built_at', '?')}")


# ---------- CROSS-REGION VIEW ----------
if region_pick == "Cross-Region":
    st.title("Cross-Region Comparison")
    if "china" not in dataset.regions or "india" not in dataset.regions:
        st.warning("Need both China and India regions enabled in config.yaml.")
        st.stop()

    cs = cached_spread(dataset["china"], "china_full", file_mtime)
    ins = cached_spread(dataset["india"], "india_full", file_mtime)
    cmp = cross_region_comparison(cs, ins)

    if not cmp.get("available"):
        st.warning(f"Comparison not available: {cmp.get('reason')}")
        st.stop()

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
    fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=580, showlegend=True)
    style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Spreads are in different currencies (USD vs INR), so absolute comparison "
            "is not meaningful. We rank each region's spread within its own historical "
            "distribution, and use z-scores to measure divergence.")
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
tab_overview, tab_spread, tab_diag, tab_lead_lag, tab_regimes, tab_cyclicity, tab_attribution, tab_forecast, tab_macro = st.tabs([
    "Overview", "Spread", "Diagnostics", "Lead/Lag", "Regimes", "Cyclicity", "Attribution", "Forecasts", "Macro Calendar"
])


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
    spread = cached_spread(region, f"{region_pick}_full", file_mtime)
    if spread is None:
        st.warning(f"No spread config for {region_pick}. Add `spread:` block to "
                   f"config.yaml under data.regions.{region_pick}.")
    else:
        # Live interpretation panel (collapsed by default)
        render_interpretation(
            narrator.narrate_spread(spread, currency),
            label="📝 Spread interpretation",
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
                label="📝 Lead/Lag interpretation",
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
    if len(selected_drivers) == 0:
        st.warning("Select at least one driver in the sidebar.")
    else:
        n_reg = st.slider("Number of regimes", 2, 5, 3)
        regimes = cached_regimes(filt_region, filter_key, n_reg, file_mtime)

        # Live interpretation
        try:
            render_interpretation(
                narrator.narrate_regimes(regimes, currency),
                label="📝 Regimes interpretation",
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
            label="📝 Cyclicity interpretation",
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
                    label="📝 Attribution interpretation",
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


# ===== FORECASTS — Model Laboratory + Backtest + GARCH =====
with tab_forecast:
    st.markdown("### Forecast Laboratory")
    st.caption(
        "Configure ARIMAX/ARDL parameters and inspect forecasts, walk-forward backtests, "
        "and GARCH volatility/risk metrics. All operations are cached — first run "
        "takes 5-30 seconds, subsequent views are instant."
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
        key="fc_drivers",
        help="Uncheck a driver to fit the model without it. Useful for testing each driver's contribution."
    )

    if len(selected_drivers) == 0:
        st.warning("Select at least one driver to fit the model.")
        st.stop()

    # Fit + display the chosen model(s)
    drivers_tuple = tuple(sorted(selected_drivers))

    def _render_fit(fit_result, label, color):
        """Render one model fit's outputs."""
        if not fit_result.success:
            st.error(f"❌ **{label} failed**: {fit_result.error_msg}")
            st.caption(f"Configuration: {fit_result.config_summary}")
            return

        st.success(f"✓ **{label}** — {fit_result.config_summary}")

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("RMSE", f"{currency} {fit_result.rmse:,.0f}/t")
        m2.metric("MAPE", f"{fit_result.mape:.2f}%")
        m3.metric("R²", f"{fit_result.r2:.3f}")
        m4.metric("AIC", f"{fit_result.aic:.0f}")
        m5.metric("BIC", f"{fit_result.bic:.0f}")

        # Ljung-Box residual diagnostic
        if fit_result.ljung_box_p is not None:
            lb_status = "✓ no autocorrelation" if fit_result.ljung_box_p > 0.05 else "⚠ residual autocorrelation present"
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
                label="📝 ARIMAX interpretation",
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
                label="📝 ARDL interpretation",
                settings_summary=f"AR={ar_ardl}, DL={dl_ardl}, "
                                  f"horizon={forecast_horizon}m, "
                                  f"{len(selected_drivers)} drivers",
            )
        except Exception:
            pass
    else:  # Both (compare)
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
                    "Status": "✓",
                })
            else:
                comp_data.append({
                    "Model": name, "RMSE": "—", "MAPE %": "—", "R²": "—",
                    "AIC": "—", "BIC": "—", "Status": "❌",
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
                label="📝 Combined ARIMAX + ARDL interpretation",
                settings_summary=f"ARIMAX({ar_arx},{d_arx},{ma_arx}) vs "
                                  f"ARDL(ar={ar_ardl}, dl={dl_ardl}), "
                                  f"horizon={forecast_horizon}m, "
                                  f"{len(selected_drivers)} drivers",
            )
        except Exception:
            pass


    # ===== SECTION B: Walk-Forward Backtest =====
    st.markdown("---")
    st.markdown("#### Section B · Walk-Forward Backtest")
    st.caption(
        "Re-train the model month-by-month going back N years, each time forecasting H months ahead. "
        "Compare these rolling forecasts to actual prices to measure honest out-of-sample performance."
    )

    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
    with bt_c1:
        bt_model = st.selectbox("Model", ["ARIMAX", "ARDL"], key="fc_bt_model")
    with bt_c2:
        bt_horizon = st.slider("Forecast horizon", 1, 12, 6, key="fc_bt_horizon",
                                  help="Each fold forecasts this many months ahead.")
    with bt_c3:
        max_train = max(60, region.n_obs - bt_horizon - 12)
        bt_min_train = st.slider("Min training months", 36, max_train,
                                    min(60, max_train), step=12,
                                    key="fc_bt_min_train",
                                    help="Earliest training window starts here.")
    with bt_c4:
        bt_step = st.slider("Step (months)", 1, 6, 3, key="fc_bt_step",
                              help="Move the origin forward by this many months between folds.")

    if st.button("Run walk-forward backtest", key="fc_bt_run",
                   help="Click to run the backtest. Takes 30-90 seconds first time, instant on cache hit."):
        bt = cached_backtest(region, region_pick, bt_model.lower(),
                                ar_arx, d_arx, ma_arx, dl_ardl,
                                bt_horizon, drivers_tuple,
                                bt_min_train, bt_step, file_mtime)
        st.session_state["fc_bt_result"] = bt

    if "fc_bt_result" in st.session_state:
        bt = st.session_state["fc_bt_result"]
        if not bt.success:
            st.error(f"Backtest failed: {bt.error_msg}")
        else:
            # Live interpretation
            try:
                render_interpretation(
                    narrator.narrate_backtest(bt, bt_model, bt_horizon, currency),
                    label="📝 Backtest interpretation",
                    settings_summary=f"{bt_model}, horizon={bt_horizon}m, "
                                      f"min train={bt_min_train}m, step={bt_step}m, "
                                      f"{len(selected_drivers)} drivers",
                )
            except Exception:
                pass

            # Top-line metrics
            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Folds completed", bt.n_folds)
            tm2.metric("Out-of-sample RMSE", f"{currency} {bt.overall_rmse:,.0f}/t")
            tm3.metric("Out-of-sample MAPE", f"{bt.overall_mape:.2f}%")
            tm4.metric("Hit rate (direction)", f"{bt.overall_hit_rate:.0f}%",
                         help="% of forecasts that correctly predicted up vs down direction.")

            # Honest assessment callout
            st.info(
                f"**Honest assessment:** Over {bt.n_folds} historical retraining folds, this model's "
                f"forecasts were on average **±{bt.overall_mape:.1f}%** away from actual prices "
                f"in out-of-sample tests. It correctly predicted direction "
                f"**{bt.overall_hit_rate:.0f}%** of the time. "
                f"{'A hit rate well above 50% indicates genuine forecast skill.' if bt.overall_hit_rate > 60 else 'A hit rate near 50% suggests forecasts are no better than coin-flip on direction.'}"
            )

            # Metrics by horizon
            st.markdown("##### Performance degradation by horizon")
            st.dataframe(bt.metrics_by_horizon, use_container_width=True, hide_index=True)
            st.caption("Forecast accuracy typically degrades at longer horizons. RMSE/MAPE rising with h is expected.")

            # Rolling forecast vs actual chart
            st.markdown("##### Rolling forecasts vs actual")
            fig = go.Figure()
            target = region.y
            fig.add_trace(go.Scatter(
                x=target.index, y=target.values, mode="lines",
                line=dict(color=COLORS["ink"], width=1.5), name="Actual"
            ))
            # Plot forecasts at horizon = bt_horizon (terminal forecasts only for clarity)
            terminal = bt.fold_results[bt.fold_results["h_months"] == bt_horizon]
            fig.add_trace(go.Scatter(
                x=terminal["target_date"], y=terminal["forecast"],
                mode="markers",
                marker=dict(color=COLORS["accent"], size=6, opacity=0.7,
                              symbol="diamond"),
                name=f"{bt_horizon}m-ahead forecast (each origin)",
            ))
            fig.update_layout(**PLOT_BASE, margin=DEFAULT_MARGIN, height=420,
                                yaxis_title=f"{currency}/t",
                                hovermode="x unified")
            style_axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("All fold results"):
                st.dataframe(bt.fold_results.round(2), use_container_width=True,
                              hide_index=True)


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
                label="📝 GARCH / risk interpretation",
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
        regime_emoji = {"low": "🟢", "normal": "⚪", "elevated": "🟠", "extreme": "🔴"}.get(g.regime_label, "⚪")
        gm4.metric("Volatility regime", f"{regime_emoji} {g.regime_label.upper()}")

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
                vol_h = g.forecast_vol.values[: len(point_fc)] / 100.0  # to decimal
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
        st.caption(f"Window: **{cal.window_start}** to **{cal.window_end}**  ·  "
                    f"Library: {cal.n_total_in_library} total events  ·  "
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
                label="📝 Calendar overview interpretation",
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
                    {ev.days_until} days from today  ·  {ev.consensus}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Use expander for the detail content
            with st.expander("Details: mechanism, reaction, analogues", expanded=False):
                st.markdown(f"**Affects:** {', '.join(r.title() for r in ev.affects_regions)}  ·  "
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


# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.caption("**Workflow**")
st.sidebar.caption("1. Edit `data/Raw_data.xlsx`")
st.sidebar.caption("2. Click 🔄 Refresh data above")
st.sidebar.caption("3. To regenerate the static report: `python3 run.py`")
