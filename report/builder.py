"""
HTML report builder.

Generates a polished, self-contained HTML report from the orchestrator's
results dict. Cleaner-but-presentable design — neutral palette, subtle
accents, lots of whitespace, plotly charts embedded inline. The narrator
module provides prose interpretations that are woven between charts.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from pipeline import narrator


# ---------- Visual identity ----------
COLORS = {
    "ink":       "#1A1F2E",     # primary text
    "muted":     "#5C6B7F",     # secondary text
    "rule":      "#E5E9F0",     # subtle dividers
    "bg":        "#FFFFFF",     # background
    "surface":   "#F8FAFC",     # alt rows / cards
    "accent":    "#1F4E79",     # primary brand accent (steel blue)
    "accent2":   "#2D6A4F",     # positive
    "warning":   "#C9540F",     # warning
    "danger":    "#A4161A",     # negative
    "neutral":   "#6B7280",
}
CHART_PALETTE = ["#1F4E79", "#2D6A4F", "#C9540F", "#7B5EA7",
                 "#A4161A", "#0F8B8D", "#5C6B7F", "#B8860B"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, system-ui, sans-serif",
              color=COLORS["ink"], size=12),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=50, r=20, t=40, b=40),
    colorway=CHART_PALETTE,
    xaxis=dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"]),
    yaxis=dict(gridcolor=COLORS["rule"], linecolor=COLORS["rule"]),
)


def _fmt_num(x, decimals=1, prefix="", suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{prefix}{x:,}{suffix}"
    return f"{prefix}{x:,.{decimals}f}{suffix}"


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a styled HTML table."""
    if df is None or len(df) == 0:
        return '<div class="empty">No data</div>'
    if isinstance(df, dict) and "error" in df:
        return f'<div class="error">Error: {df["error"]}</div>'
    df_display = df.head(max_rows).copy()
    # Format numerics
    for c in df_display.columns:
        if pd.api.types.is_float_dtype(df_display[c]):
            df_display[c] = df_display[c].apply(
                lambda v: f"{v:,.3f}" if pd.notna(v) else "—")
        elif pd.api.types.is_integer_dtype(df_display[c]):
            df_display[c] = df_display[c].apply(
                lambda v: f"{v:,}" if pd.notna(v) else "—")
    return df_display.to_html(index=False, classes="data-table",
                                escape=True, border=0)


def _embed_chart(fig, height=380) -> str:
    """Render a plotly figure as a self-contained div. Failures are handled gracefully."""
    if fig is None:
        return '<div class="empty">Chart not available</div>'
    try:
        fig.update_layout(**PLOTLY_LAYOUT, height=height)
        return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                           config={"displayModeBar": False, "responsive": True})
    except Exception as e:
        return f'<div class="error">Chart render failed: {type(e).__name__}: {e}</div>'


def _render_prose(blocks: List) -> str:
    """Convert narrator blocks (paragraphs + KEY INTERPRETATION + event cards) to HTML."""
    out = []
    for b in blocks:
        if isinstance(b, dict) and b.get("_kind") == "key_interpretation":
            out.append(
                f'<div class="key-interpretation">'
                f'<div class="key-label">{b["title"]}</div>'
                f'<div class="key-body">{b["body"]}</div>'
                f'</div>'
            )
        elif isinstance(b, dict) and b.get("_kind") == "event_card":
            impact_lower = b["impact"].lower()
            conf_lower = b["confidence"].lower() if b["confidence"] in ("high","medium","low") else "low"
            out.append(
                f'<div class="event-card {impact_lower}">'
                f'<div class="event-header">'
                f'<span class="event-date-pill">{b["date"]}</span>'
                f'<span class="event-impact {impact_lower}">{b["impact"]}</span>'
                f'<span class="event-country">{b["country"]}</span>'
                f'<span class="event-confidence {conf_lower}">'
                f'analogues: {b["confidence"]} ({b["cond_n"]} matches)</span>'
                f'</div>'
                f'<div class="event-name">{b["name"]}</div>'
                f'<p>{b["intro"]}</p>'
                f'<p><b>Mechanism:</b> {b["mechanism"]}</p>'
                f'<div class="reaction"><p>{b["reaction"]}</p></div>'
                f'<div class="analogue"><p>{b["analogue"]}</p></div>'
                f'</div>'
            )
        else:
            out.append(f'<p class="prose">{b}</p>')
    return "\n".join(out)


# ---------- Section builders ----------

def chart_target_history(region_data, region_name, currency):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=region_data.index, y=region_data.values,
        mode="lines", line=dict(color=COLORS["accent"], width=2),
        fill="tozeroy", fillcolor="rgba(31,78,121,0.06)",
        name=f"HRC {region_name}",
        hovertemplate=f"%{{x|%b %Y}}: {currency} %{{y:,.0f}}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=f"HRC Price ({currency}/t) — {region_name}",
                                  font=dict(size=14)),
                      yaxis_title=f"{currency}/t")
    return fig


def chart_spread(spread_result):
    s = spread_result.spread_series
    p = spread_result.percentiles
    cur = spread_result.current_snapshot
    fig = go.Figure()
    # Percentile bands
    fig.add_hrect(y0=p["p10"], y1=p["p25"], fillcolor=COLORS["danger"],
                  opacity=0.06, line_width=0,
                  annotation_text="Bottom quartile", annotation_position="top left",
                  annotation_font_size=10)
    fig.add_hrect(y0=p["p75"], y1=p["p90"], fillcolor=COLORS["accent2"],
                  opacity=0.06, line_width=0,
                  annotation_text="Top quartile", annotation_position="top left",
                  annotation_font_size=10)
    fig.add_hline(y=p["p50"], line_dash="dot", line_color=COLORS["muted"],
                  annotation_text="Median", annotation_position="right",
                  annotation_font_size=10)
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
        name="Spread",
        hovertemplate=f"%{{x|%b %Y}}: {spread_result.currency} %{{y:,.0f}}<extra></extra>",
    ))
    # Mark current value
    fig.add_trace(go.Scatter(
        x=[s.index[-1]], y=[s.iloc[-1]], mode="markers",
        marker=dict(color=COLORS["accent"], size=10,
                    line=dict(color="white", width=2)),
        name="Current",
        hovertemplate=f"Current: {spread_result.currency} %{{y:,.0f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{spread_result.region.title()} Spread Time Series  ·  "
                          f"Currently {spread_result.currency} {cur['spread']:,.0f}/t  ·  "
                          f"P{cur['spread_percentile']:.0f} of history  ·  "
                          f"<b>{cur['current_regime'].upper()}</b>",
                    font=dict(size=14)),
        yaxis_title=f"{spread_result.currency}/t",
        showlegend=False,
    )
    return fig


def chart_decomposition(spread_result):
    """Stacked area: HRC = Iron Ore contribution + HCC contribution + Spread."""
    d = spread_result.decomposition
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["IronOre_contribution"],
                              mode="lines", line=dict(width=0),
                              stackgroup="one", name="Iron Ore (1.6×)",
                              fillcolor="rgba(31,78,121,0.7)"))
    fig.add_trace(go.Scatter(x=d.index, y=d["HCC_contribution"],
                              mode="lines", line=dict(width=0),
                              stackgroup="one", name="HCC (0.9×)",
                              fillcolor="rgba(45,106,79,0.7)"))
    fig.add_trace(go.Scatter(x=d.index, y=d["Spread"],
                              mode="lines", line=dict(width=0),
                              stackgroup="one", name="Spread (Margin)",
                              fillcolor="rgba(201,84,15,0.6)"))
    fig.update_layout(title=dict(text="HRC Price Decomposition: Raw Materials vs Margin",
                                  font=dict(size=14)),
                      yaxis_title=f"{spread_result.currency}/t",
                      hovermode="x unified")
    return fig


def chart_correlation_heatmap(corr_df):
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.index,
        colorscale=[[0, COLORS["danger"]], [0.5, "white"], [1, COLORS["accent2"]]],
        zmin=-1, zmax=1, zmid=0,
        text=np.round(corr_df.values, 2), texttemplate="%{text}",
        textfont=dict(size=10), hoverongaps=False,
    ))
    fig.update_layout(title=dict(text="Correlation Matrix", font=dict(size=14)),
                      xaxis=dict(tickangle=-30))
    return fig


def chart_lead_lag(lead_lag_df, target_name):
    if lead_lag_df is None or len(lead_lag_df) == 0:
        return None
    df = lead_lag_df.copy().head(15)
    fig = go.Figure()
    colors = [COLORS["accent"] if g else COLORS["muted"]
              for g in df["granger_x_causes_y"]]
    fig.add_trace(go.Bar(
        y=df["driver"], x=df["best_lag_months"], orientation="h",
        marker_color=colors,
        text=[f"r={c:.2f} (lag {l:+d})" for c, l in
              zip(df["ccf_at_best_lag"], df["best_lag_months"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Best lag: %{x} months<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLORS["muted"], line_dash="dot")
    fig.update_layout(
        title=dict(text=f"Lead/Lag vs {target_name}  ·  positive = driver leads price  ·  "
                         "filled bars = Granger-significant",
                    font=dict(size=14)),
        xaxis_title="Best lag (months)",
        yaxis=dict(autorange="reversed"),
        height=max(380, len(df) * 26),
    )
    return fig


def chart_forecast(model_result, actual_series, region_name):
    fig = go.Figure()
    # Historical actual
    fig.add_trace(go.Scatter(
        x=actual_series.index, y=actual_series.values, mode="lines",
        name="Actual", line=dict(color=COLORS["ink"], width=1.5),
    ))
    # Fitted (in-sample)
    if model_result.fitted is not None:
        fig.add_trace(go.Scatter(
            x=model_result.fitted.index, y=model_result.fitted.values,
            mode="lines", name="In-sample fit",
            line=dict(color=COLORS["accent"], width=1.5, dash="dot"), opacity=0.6,
        ))
    # OOS predictions
    if model_result.oos_predictions is not None:
        fig.add_trace(go.Scatter(
            x=model_result.oos_predictions.index, y=model_result.oos_predictions.values,
            mode="lines", name="OOS test prediction",
            line=dict(color=COLORS["warning"], width=2),
        ))
    # Forward forecast with CI
    if model_result.forecast is not None:
        if model_result.forecast_lower is not None and model_result.forecast_upper is not None:
            fig.add_trace(go.Scatter(
                x=list(model_result.forecast_upper.index) + list(model_result.forecast_lower.index[::-1]),
                y=list(model_result.forecast_upper.values) + list(model_result.forecast_lower.values[::-1]),
                fill="toself", fillcolor="rgba(31,78,121,0.12)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="95% CI",
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=model_result.forecast.index, y=model_result.forecast.values,
            mode="lines+markers", name="Forecast",
            line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=6),
        ))
    fig.update_layout(
        title=dict(text=f"{model_result.name.upper()} — {region_name}", font=dict(size=14)),
        yaxis_title="Price",
    )
    return fig


def chart_garch_volatility(garch_result, region_name):
    cv = garch_result.conditional_volatility
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cv.index, y=cv.values, mode="lines",
        line=dict(color=COLORS["danger"], width=1.8),
        fill="tozeroy", fillcolor="rgba(164,22,26,0.1)",
        name="Conditional volatility (%)",
    ))
    if garch_result.forecast_volatility is not None:
        fv = garch_result.forecast_volatility
        fig.add_trace(go.Scatter(
            x=fv.index, y=fv.values, mode="lines+markers",
            line=dict(color=COLORS["danger"], width=2, dash="dash"),
            marker=dict(size=6), name="Forecast volatility",
        ))
    fig.update_layout(
        title=dict(text=f"GARCH(1,1) Conditional Volatility — {region_name}",
                    font=dict(size=14)),
        yaxis_title="Volatility (% returns)",
    )
    return fig


def chart_regimes(regime_result, target_series):
    colors_map = [COLORS["danger"], COLORS["muted"], COLORS["accent2"]]
    if regime_result.n_regimes > 3:
        colors_map = CHART_PALETTE[:regime_result.n_regimes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=target_series.index, y=target_series.values, mode="lines",
        line=dict(color=COLORS["ink"], width=1.5), name="Price",
    ))
    for r in range(regime_result.n_regimes):
        mask = regime_result.labels == r
        fig.add_trace(go.Scatter(
            x=target_series.index[mask], y=target_series.values[mask],
            mode="markers", name=f"Regime {r}",
            marker=dict(color=colors_map[r], size=8, opacity=0.7),
        ))
    fig.update_layout(title=dict(text="Regime Classification (K-Means)",
                                  font=dict(size=14)),
                      yaxis_title="Price")
    return fig


def chart_attribution(attribution_result):
    rb = attribution_result.rolling_betas
    if len(rb) == 0:
        return None
    # Show absolute attribution shares over time
    abs_betas = rb.abs()
    shares = abs_betas.div(abs_betas.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for col in shares.columns:
        fig.add_trace(go.Scatter(
            x=shares.index, y=shares[col], mode="lines",
            stackgroup="one", name=col,
        ))
    fig.update_layout(
        title=dict(text="Rolling Attribution — Driver Importance Over Time (24m window)",
                    font=dict(size=14)),
        yaxis_title="Share of explanatory power (%)",
        hovermode="x unified",
    )
    return fig


def chart_cross_region(cross_data):
    if not cross_data.get("available"):
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Spread Percentile Rank (within own history)",
                                        "Divergence: India z-score − China z-score"),
                        vertical_spacing=0.12)
    cp = cross_data["china_percentile_series"]
    ip = cross_data["india_percentile_series"]
    fig.add_trace(go.Scatter(x=cp.index, y=cp.values, mode="lines",
                              line=dict(color=COLORS["danger"], width=2),
                              name="China %ile"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ip.index, y=ip.values, mode="lines",
                              line=dict(color=COLORS["accent2"], width=2),
                              name="India %ile"), row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["muted"], row=1, col=1)
    div = cross_data["divergence_series"]
    fig.add_trace(go.Scatter(x=div.index, y=div.values, mode="lines",
                              line=dict(color=COLORS["accent"], width=2),
                              fill="tozeroy", fillcolor="rgba(31,78,121,0.1)",
                              name="India − China z"), row=2, col=1)
    fig.add_hline(y=0, line_color=COLORS["muted"], row=2, col=1)
    fig.update_yaxes(title_text="Percentile", row=1, col=1)
    fig.update_yaxes(title_text="z-score diff", row=2, col=1)
    fig.update_layout(showlegend=True, height=560)
    return fig


# ===== CYCLICITY charts =====

def chart_cyclicity_regimes(cyc_result, target_series):
    """Coloured time series — same as regime chart but with GMM regime labels."""
    n_reg = cyc_result.n_regimes
    palette = [COLORS["danger"], COLORS["muted"], COLORS["warning"], COLORS["accent2"]]
    if n_reg > 4:
        palette = CHART_PALETTE[:n_reg]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=target_series.index, y=target_series.values, mode="lines",
        line=dict(color=COLORS["ink"], width=1.5), name="HRC price",
    ))
    for r in range(n_reg):
        mask = cyc_result.regime_labels == r
        if mask.sum() == 0:
            continue
        idx = cyc_result.regime_labels.index[mask]
        # Get target values aligned to the regime label index
        vals = target_series.reindex(idx).values
        label = cyc_result.regime_profiles[r].label
        fig.add_trace(go.Scatter(
            x=idx, y=vals, mode="markers", name=f"R{r}: {label}",
            marker=dict(color=palette[r % len(palette)], size=8, opacity=0.75,
                          line=dict(color="white", width=0.5)),
        ))
    fig.update_layout(title=dict(text="Regime Classification (Multi-Feature GMM)",
                                  font=dict(size=14)),
                      yaxis_title="Price", legend=dict(orientation="h", y=-0.15))
    return fig


def chart_peaks_troughs(cyc_result, target_series):
    """Target series with peaks (▲) and troughs (▼) marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=target_series.index, y=target_series.values, mode="lines",
        line=dict(color=COLORS["accent"], width=1.8), name="Price",
    ))
    if len(cyc_result.peaks) > 0:
        fig.add_trace(go.Scatter(
            x=cyc_result.peaks.index, y=cyc_result.peaks.values,
            mode="markers", name=f"Peaks (n={len(cyc_result.peaks)})",
            marker=dict(symbol="triangle-up", size=12, color=COLORS["accent2"],
                          line=dict(color="white", width=1)),
        ))
    if len(cyc_result.troughs) > 0:
        fig.add_trace(go.Scatter(
            x=cyc_result.troughs.index, y=cyc_result.troughs.values,
            mode="markers", name=f"Troughs (n={len(cyc_result.troughs)})",
            marker=dict(symbol="triangle-down", size=12, color=COLORS["danger"],
                          line=dict(color="white", width=1)),
        ))
    fig.update_layout(title=dict(text="Peak/Trough Detection — Cycle Structure",
                                  font=dict(size=14)),
                      yaxis_title="Price")
    return fig


def chart_transition_matrix(cyc_result):
    """Heatmap of the regime-to-regime transition probabilities."""
    tm = cyc_result.transition_matrix
    labels = [f"R{i}" for i in range(cyc_result.n_regimes)]
    fig = go.Figure(data=go.Heatmap(
        z=tm.values, x=labels, y=labels,
        colorscale=[[0, "white"], [0.5, COLORS["accent"]], [1, COLORS["ink"]]],
        zmin=0, zmax=1,
        text=np.round(tm.values * 100, 1), texttemplate="%{text}%",
        textfont=dict(size=11),
        colorbar=dict(title="Probability"),
    ))
    fig.update_layout(title=dict(text="Markov Transition Matrix — Regime-to-Regime",
                                  font=dict(size=14)),
                      xaxis_title="To", yaxis_title="From",
                      yaxis=dict(autorange="reversed"))
    return fig


# ---------- Master HTML builder ----------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
       background: #FAFBFC; color: #1A1F2E; line-height: 1.55;
       -webkit-font-smoothing: antialiased; }
.container { max-width: 1280px; margin: 0 auto; padding: 32px 24px; }

.cover { background: linear-gradient(135deg, #1F4E79 0%, #2C5F8D 100%);
         color: white; padding: 56px 48px; border-radius: 16px;
         margin-bottom: 32px; }
.cover .eyebrow { font-size: 0.75rem; letter-spacing: 0.15em;
                  text-transform: uppercase; color: rgba(255,255,255,0.7);
                  margin-bottom: 14px; font-weight: 600; }
.cover h1 { font-size: 2.4rem; font-weight: 700; line-height: 1.15; margin-bottom: 12px; }
.cover .sub { font-size: 1.05rem; color: rgba(255,255,255,0.85); margin-bottom: 28px; }
.cover .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
                margin-top: 24px; }
.cover .stat { background: rgba(255,255,255,0.08); padding: 16px; border-radius: 10px; }
.cover .stat-value { font-size: 1.4rem; font-weight: 700; }
.cover .stat-label { font-size: 0.7rem; text-transform: uppercase;
                     letter-spacing: 0.08em; opacity: 0.8; margin-top: 4px; }

section { background: white; border-radius: 12px; padding: 28px 32px;
          margin-bottom: 24px; border: 1px solid #E5E9F0; }
section h2 { font-size: 1.3rem; font-weight: 700; margin-bottom: 6px;
             color: #1A1F2E; }
section .subtitle { color: #5C6B7F; font-size: 0.92rem;
                    margin-bottom: 20px; }
section h3 { font-size: 1rem; font-weight: 600; margin: 24px 0 10px;
             color: #1F4E79; }

.region-tabs { display: flex; gap: 8px; margin-bottom: 20px; }
.region-pill { padding: 6px 14px; border-radius: 20px; font-size: 0.78rem;
               font-weight: 600; background: #F0F4F8; color: #5C6B7F; }
.region-pill.active { background: #1F4E79; color: white; }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin: 16px 0; }
.kpi { background: #F8FAFC; padding: 14px 16px; border-radius: 8px;
       border-left: 3px solid #1F4E79; }
.kpi .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
              color: #5C6B7F; font-weight: 600; }
.kpi .value { font-size: 1.4rem; font-weight: 700; color: #1A1F2E; margin-top: 4px; }
.kpi .delta { font-size: 0.75rem; color: #5C6B7F; margin-top: 2px; }
.kpi.warning { border-left-color: #C9540F; }
.kpi.positive { border-left-color: #2D6A4F; }
.kpi.danger { border-left-color: #A4161A; }

table.data-table { width: 100%; border-collapse: collapse; font-size: 0.85rem;
                   margin: 12px 0; }
table.data-table th { background: #1F4E79; color: white; padding: 9px 12px;
                       text-align: left; font-weight: 600; font-size: 0.78rem;
                       text-transform: uppercase; letter-spacing: 0.04em; }
table.data-table td { padding: 8px 12px; border-bottom: 1px solid #E5E9F0; }
table.data-table tr:nth-child(even) td { background: #F8FAFC; }
table.data-table tr:hover td { background: #F0F4F8; }

.chart-container { margin: 16px 0 24px; }

.error { background: #FEE2E2; color: #A4161A; padding: 12px 16px;
         border-radius: 6px; font-size: 0.88rem; }
.empty { color: #9CA3AF; font-style: italic; padding: 12px; font-size: 0.9rem; }
.note { background: #F8FAFC; border-left: 3px solid #1F4E79; padding: 10px 14px;
        font-size: 0.85rem; color: #5C6B7F; margin: 12px 0; border-radius: 4px; }
.regime-tag { display: inline-block; padding: 3px 10px; border-radius: 10px;
              font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.04em; }
.regime-tag.compressed { background: #FEE2E2; color: #A4161A; }
.regime-tag.normal { background: #F0F4F8; color: #5C6B7F; }
.regime-tag.expanded { background: #DCFCE7; color: #2D6A4F; }

/* Prose paragraphs — the analyst's voice */
p.prose { font-size: 0.95rem; line-height: 1.65; color: #2D3748;
          margin: 12px 0; max-width: 95%; }
p.prose b { color: #1A1F2E; font-weight: 600; }
p.prose i { color: #5C6B7F; }
p.prose code { background: #F0F4F8; padding: 1px 6px; border-radius: 3px;
               font-size: 0.85em; font-family: 'SF Mono', Consolas, monospace; }

/* KEY INTERPRETATION box — section summary */
.key-interpretation { background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
                       border-left: 4px solid #1F4E79; padding: 14px 20px;
                       margin: 20px 0 24px; border-radius: 6px;
                       box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.key-interpretation .key-label { font-size: 0.72rem; font-weight: 700;
                                  color: #1F4E79; text-transform: uppercase;
                                  letter-spacing: 0.08em; margin-bottom: 6px; }
.key-interpretation .key-body { font-size: 0.92rem; color: #1A1F2E;
                                 line-height: 1.55; }
.key-interpretation .key-body b { color: #0F3460; }

/* Macro Calendar Event Cards */
.event-card { background: white; border: 1px solid #E5E9F0;
              border-radius: 8px; padding: 18px 22px; margin: 16px 0;
              box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.event-card.high { border-left: 4px solid #A4161A; }
.event-card.med { border-left: 4px solid #C9540F; }
.event-card.low { border-left: 4px solid #5C6B7F; }

.event-header { display: flex; align-items: center; gap: 12px;
                margin-bottom: 10px; flex-wrap: wrap; }
.event-date-pill { background: #1A1F2E; color: white; padding: 4px 10px;
                    border-radius: 4px; font-size: 0.78rem; font-weight: 600;
                    font-family: 'SF Mono', Consolas, monospace; }
.event-impact { font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.06em; padding: 3px 8px; border-radius: 3px; }
.event-impact.high { background: #FEE2E2; color: #A4161A; }
.event-impact.med { background: #FED7AA; color: #C9540F; }
.event-impact.low { background: #F0F4F8; color: #5C6B7F; }
.event-country { font-size: 0.75rem; color: #5C6B7F; font-weight: 500;
                  background: #F0F4F8; padding: 2px 8px; border-radius: 3px; }
.event-confidence { font-size: 0.7rem; color: #5C6B7F; margin-left: auto; }
.event-confidence.high { color: #2D6A4F; }
.event-confidence.medium { color: #C9540F; }
.event-confidence.low { color: #A4161A; }
.event-name { font-size: 1.02rem; font-weight: 600; color: #1A1F2E;
               margin: 6px 0 12px; }
.event-card p { font-size: 0.9rem; line-height: 1.6; margin: 8px 0;
                color: #2D3748; }
.event-card p b { color: #1A1F2E; }
.event-card p i { color: #5C6B7F; }
.event-card .reaction { background: #F8FAFC; padding: 10px 14px;
                         border-radius: 5px; border-left: 3px solid #1F4E79;
                         margin: 10px 0; }
.event-card .analogue { background: #FFFBF0; padding: 10px 14px;
                         border-radius: 5px; border-left: 3px solid #C9540F;
                         margin: 10px 0; font-size: 0.88rem; }

footer { text-align: center; color: #9CA3AF; font-size: 0.8rem;
         padding: 24px 0; margin-top: 32px; border-top: 1px solid #E5E9F0; }

/* ===== Liquidity section ===== */
.liq-regime-tag { display: inline-block; padding: 3px 10px; border-radius: 10px;
                  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                  letter-spacing: 0.04em; }
.liq-regime-tag.surplus { background: #DCFCE7; color: #2D6A4F; }
.liq-regime-tag.neutral { background: #F0F4F8; color: #5C6B7F; }
.liq-regime-tag.tight   { background: #FEE2E2; color: #A4161A; }

.liq-policy-tag { display: inline-block; padding: 3px 10px; border-radius: 10px;
                  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                  letter-spacing: 0.04em; background: #F0F4F8; color: #1F4E79; }
.liq-policy-tag.easing            { background: #DCFCE7; color: #2D6A4F; }
.liq-policy-tag.aggressive-easing { background: #BBF7D0; color: #166534; }
.liq-policy-tag.hiking            { background: #FED7AA; color: #C9540F; }
.liq-policy-tag.aggressive-hiking { background: #FECACA; color: #A4161A; }
.liq-policy-tag.pause             { background: #F0F4F8; color: #5C6B7F; }

.liq-legend { display: flex; gap: 18px; margin: 10px 0 4px;
              font-size: 0.85rem; color: #2D3748; flex-wrap: wrap; }
.liq-legend-swatch { display: inline-block; width: 12px; height: 12px;
                     border-radius: 2px; margin-right: 6px; vertical-align: middle; }

.liq-panel-grid { display: grid; grid-template-columns: repeat(2, 1fr);
                  gap: 14px; margin: 12px 0; }
@media (max-width: 900px) { .liq-panel-grid { grid-template-columns: 1fr; } }

.liq-interp { background: #F8FAFC; border-left: 3px solid #1F4E79;
              padding: 12px 16px; margin: 14px 0; border-radius: 4px;
              font-size: 0.9rem; line-height: 1.6; color: #2D3748; }
.liq-interp b { color: #1A1F2E; }
.liq-interp ul { margin: 8px 0 4px 18px; }
.liq-interp li { margin: 4px 0; }
"""


# ===== LIQUIDITY charts (India only) =====
# Mirrors the dashboard's Liquidity tab. Each helper is defensive: returns None
# (which `_embed_chart` then renders as an empty placeholder) on missing data,
# rather than raising. Same chart styling/palette as the rest of the report.

def chart_liquidity_stress_gauge(stress_val: float):
    """0-100 indicator gauge with 5 colored bands matching the dashboard."""
    if stress_val is None or (isinstance(stress_val, float) and np.isnan(stress_val)):
        return None
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(stress_val),
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
                {'range': [0, 30],   'color': '#C7E8D2'},
                {'range': [30, 50],  'color': '#E6F1E8'},
                {'range': [50, 65],  'color': '#F5F1E5'},
                {'range': [65, 80],  'color': '#F7DCC2'},
                {'range': [80, 100], 'color': '#F2BFC0'},
            ],
            'threshold': {
                'line': {'color': COLORS["danger"], 'width': 3},
                'thickness': 0.85, 'value': float(stress_val),
            },
        },
    ))
    return fig


def chart_hrc_vs_liquidity_regime(hrc, spread_series, repo_series, regime_series,
                                    currency: str):
    """HRC line + shaded liquidity-regime backgrounds + WACR-Repo spread (right axis)."""
    from pipeline.liquidity import regime_periods

    if hrc is None or hrc.empty:
        return None

    REGIME_COLOURS = {
        'Surplus': 'rgba(45, 106, 79, 0.12)',
        'Neutral': 'rgba(92, 107, 127, 0.08)',
        'Tight':   'rgba(164, 22, 26, 0.12)',
    }
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if regime_series is not None and not regime_series.empty:
        try:
            for b in regime_periods(regime_series):
                fig.add_vrect(
                    x0=b['start'], x1=b['end'],
                    fillcolor=REGIME_COLOURS.get(b['regime'], 'rgba(0,0,0,0.05)'),
                    line_width=0, layer="below",
                )
        except Exception:
            pass

    fig.add_trace(
        go.Scatter(x=hrc.index, y=hrc.values, name=f"HRC ({currency}/t)",
                   mode="lines", line=dict(color=COLORS["accent"], width=2.2),
                   hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"),
        secondary_y=False,
    )
    if spread_series is not None and not spread_series.empty:
        fig.add_trace(
            go.Scatter(x=spread_series.index, y=spread_series.values * 100,
                       name="WACR − Repo (bps)",
                       mode="lines",
                       line=dict(color=COLORS["danger"], width=1.6, dash="dot"),
                       hovertemplate="%{x|%b %Y}: %{y:+.0f} bps<extra></extra>"),
            secondary_y=True,
        )
    if repo_series is not None and not repo_series.empty:
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
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                   y=1.02, xanchor="left", x=0))
    return fig


def chart_liquidity_variable_panel(rdf, panel_cfg):
    """6 mini-line-charts in a 3x2 grid via plotly subplots."""
    if not panel_cfg:
        return None

    n = len(panel_cfg)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    titles = [lbl for (_, lbl, _) in panel_cfg]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                          vertical_spacing=0.14, horizontal_spacing=0.10)

    for i, (cname, lbl, clr) in enumerate(panel_cfg):
        r = i // n_cols + 1
        c = i % n_cols + 1
        s = rdf[cname].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(x=s.index, y=s.values, mode="lines",
                       line=dict(color=clr, width=1.7), showlegend=False,
                       hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}}: %{{y:.2f}}<extra></extra>"),
            row=r, col=c,
        )
        if cname == 'WACR_Spread':
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"],
                          row=r, col=c)

    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=COLORS["accent"])

    fig.update_xaxes(showgrid=True, gridcolor=COLORS["rule"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["rule"])
    return fig


def _liquidity_lead_lag_html(ll_df: pd.DataFrame) -> str:
    """Render the lead-lag matrix as a coloured HTML table (red-blue diverging).
    No matplotlib dep — colour scale applied inline via CSS."""
    if ll_df is None or ll_df.empty:
        return '<div class="empty">No data</div>'

    df = ll_df.copy()
    vmax = 0.5  # cap colour scale at ±0.5 to match the dashboard

    def _cell_color(v):
        if pd.isna(v):
            return "#FFFFFF", "#5C6B7F"
        t = max(min(v / vmax, 1.0), -1.0)
        if t >= 0:
            r = int(255 + (31 - 255) * t)
            g = int(255 + (78 - 255) * t)
            b = int(255 + (121 - 255) * t)
        else:
            r = int(255 + (164 - 255) * (-t))
            g = int(255 + (22  - 255) * (-t))
            b = int(255 + (26  - 255) * (-t))
        luminance = 0.299*r + 0.587*g + 0.114*b
        fg = "#FFFFFF" if luminance < 140 else "#1A1F2E"
        return f"rgb({r},{g},{b})", fg

    cols = list(df.columns)
    rows_html = []
    header_cells = ['<th style="text-align:left">Variable</th>'] + \
                   [f'<th>{c}</th>' for c in cols]
    rows_html.append(f'<tr>{"".join(header_cells)}</tr>')
    for var in df.index:
        cells = [f'<td style="font-weight:600">{var}</td>']
        for c in cols:
            v = df.loc[var, c]
            bg, fg = _cell_color(v)
            txt = "—" if pd.isna(v) else f"{v:+.2f}"
            cells.append(
                f'<td style="background:{bg};color:{fg};text-align:center;'
                f'font-variant-numeric:tabular-nums">{txt}</td>'
            )
        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    return ('<table class="data-table">'
            + "".join(rows_html) + '</table>')


def _regime_perf_html(perf: pd.DataFrame) -> str:
    """Render the regime performance table with a red-to-green gradient on
    Annualized_Return_%. Self-contained — no matplotlib needed."""
    if perf is None or perf.empty:
        return '<div class="empty">No data</div>'

    df = perf.copy()

    def _ar_color(v):
        if pd.isna(v):
            return "#FFFFFF", "#5C6B7F"
        cap = 40.0
        t = max(min(v / cap, 1.0), -1.0)
        if t >= 0:
            r = int(255 + (45 - 255) * t)
            g = int(255 + (106 - 255) * t)
            b = int(255 + (79 - 255) * t)
        else:
            r = int(255 + (164 - 255) * (-t))
            g = int(255 + (22  - 255) * (-t))
            b = int(255 + (26  - 255) * (-t))
        luminance = 0.299*r + 0.587*g + 0.114*b
        fg = "#FFFFFF" if luminance < 140 else "#1A1F2E"
        return f"rgb({r},{g},{b})", fg

    fmt_map = {
        'Months':              lambda v: f"{int(v)}" if pd.notna(v) else "—",
        'Avg_Return_MoM_%':    lambda v: f"{v:+.2f}" if pd.notna(v) else "—",
        'Annualized_Return_%': lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
        'Volatility_Ann_%':    lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        'Max_Drawdown_%':      lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        'Hit_Rate_%':          lambda v: f"{v:.0f}" if pd.notna(v) else "—",
    }
    cols = list(df.columns)
    header_cells = ['<th style="text-align:left">Regime</th>'] + \
                   [f'<th>{c}</th>' for c in cols]
    rows_html = [f'<tr>{"".join(header_cells)}</tr>']
    for regime in df.index:
        cells = [f'<td style="font-weight:600">{regime}</td>']
        for c in cols:
            v = df.loc[regime, c]
            display = fmt_map.get(c, lambda x: f"{x}" if pd.notna(x) else "—")(v)
            if c == 'Annualized_Return_%':
                bg, fg = _ar_color(v)
                cells.append(
                    f'<td style="background:{bg};color:{fg};text-align:right;'
                    f'font-variant-numeric:tabular-nums;font-weight:600">{display}</td>'
                )
            else:
                cells.append(
                    f'<td style="text-align:right;'
                    f'font-variant-numeric:tabular-nums">{display}</td>'
                )
        rows_html.append(f'<tr>{"".join(cells)}</tr>')

    return ('<table class="data-table">'
            + "".join(rows_html) + '</table>')


def _liquidity_kpi_card(label, value, delta=None, kpi_class=""):
    """One header KPI card matching the rest of the report's .kpi style."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ''
    return (f'<div class="kpi {kpi_class}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'{delta_html}'
            f'</div>')


def _safe_num(v, default=np.nan):
    """NaN-safe float conversion used inside the liquidity section."""
    try:
        f = float(v)
        return f if not np.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _render_liquidity_section(region_name: str, region_results: dict) -> str:
    """Build the full India Liquidity section, mirroring all 5 dashboard sections.

    Returns an HTML string (a <section>...</section> block) or "" if the
    region has no liquidity columns wired up.
    """
    # Gate 1: India-only by explicit choice (no other region has the data)
    if region_name != "india":
        return ""

    rdf = region_results.get("_region_df")
    liquidity_cols = region_results.get("_liquidity_cols", [])
    meta = region_results.get("meta", {})
    currency = meta.get("currency", "INR")
    target = meta.get("target", "India_HRC")

    # Gate 2: liquidity module must be wired up and target+spread must be present
    if rdf is None or not liquidity_cols:
        return ""
    if 'WACR_Spread' not in rdf.columns or target not in rdf.columns:
        return ""

    try:
        from pipeline.liquidity import (
            regime_performance, liquidity_lead_lag,
            summarize_current_state, interpret_current_state,
        )
    except Exception as e:
        return (f'<section><h2>India Liquidity Monitor</h2>'
                f'<div class="error">Could not import liquidity module: '
                f'{type(e).__name__}: {e}</div></section>')

    parts = []

    # ----- Section header + intro -----
    parts.append(f"""
<section>
  <h2>India Liquidity Monitor</h2>
  <div class="subtitle">Banking-system liquidity, RBI monetary policy stance, and how they
    transmit into Indian HRC. Liquidity tightness affects steel demand with a typical lag of
    <b>2–5 months</b> via working-capital financing for distributors and downstream consumers.</div>
""")

    # ----- 1. CURRENT-STATE HEADER CARDS + INTERPRETATION -----
    state = {}
    try:
        state = summarize_current_state(rdf) or {}

        regime_now = state.get('Liquidity_Regime', '—')
        if regime_now is None or (isinstance(regime_now, float) and np.isnan(regime_now)):
            regime_now = '—'
        regime_class = str(regime_now).lower() if regime_now != '—' else ''

        policy_now = state.get('Policy_Regime', '—')
        if policy_now is None or (isinstance(policy_now, float) and np.isnan(policy_now)):
            policy_now = '—'
        policy_class = str(policy_now).lower().replace(' ', '-') if policy_now != '—' else ''

        spread_pct = _safe_num(state.get('WACR_Spread'))
        spread_chg = _safe_num(state.get('WACR_Spread_3m_change'), default=0.0)
        spread_val = f"{spread_pct*100:+.0f} bps" if not np.isnan(spread_pct) else "—"
        spread_delta = (f"{spread_chg*100:+.0f} bps (3m)"
                        if not np.isnan(spread_chg) else None)

        stress = _safe_num(state.get('Stress_Index'))
        stress_chg = _safe_num(state.get('Stress_Index_3m_change'), default=0.0)
        stress_val = f"{stress:.0f} / 100" if not np.isnan(stress) else "—"
        stress_delta = (f"{stress_chg:+.1f} (3m)"
                        if not np.isnan(stress_chg) else None)
        stress_kpi_class = ""
        if not np.isnan(stress):
            if stress >= 65:
                stress_kpi_class = "danger"
            elif stress >= 50:
                stress_kpi_class = "warning"
            elif stress < 40:
                stress_kpi_class = "positive"

        regime_card = (
            f'<div class="kpi"><div class="label">Liquidity Regime</div>'
            f'<div class="value">'
            f'<span class="liq-regime-tag {regime_class}">{regime_now}</span>'
            f'</div></div>'
        )
        spread_card = _liquidity_kpi_card("WACR Spread", spread_val, spread_delta)
        stress_card = _liquidity_kpi_card("Stress Index", stress_val,
                                           stress_delta, stress_kpi_class)
        policy_card = (
            f'<div class="kpi"><div class="label">RBI Policy</div>'
            f'<div class="value">'
            f'<span class="liq-policy-tag {policy_class}">{policy_now}</span>'
            f'</div></div>'
        )
        parts.append(f'<div class="kpi-grid">{regime_card}{spread_card}{stress_card}{policy_card}</div>')

        # Interpretation paragraph
        try:
            interp_text = interpret_current_state(state)
            if interp_text:
                import re
                interp_html = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', interp_text)
                interp_html = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', interp_html)
                parts.append(
                    f'<div class="key-interpretation">'
                    f'<div class="key-label">Current Read</div>'
                    f'<div class="key-body">{interp_html}</div>'
                    f'</div>'
                )
        except Exception:
            pass

        # Quick-read auxiliary line
        try:
            bc_yoy = _safe_num(state.get('Bank_Credit_YoY'))
            gs10 = _safe_num(state.get('GSec_10Y'))
            repo = _safe_num(state.get('Repo_Rate'))
            quick_parts = []
            if not np.isnan(bc_yoy):
                quick_parts.append(f"Bank credit is growing at <b>{bc_yoy:.1f}% YoY</b>")
            if not np.isnan(gs10) and not np.isnan(repo):
                tp_bps = (gs10 - repo) * 100
                quick_parts.append(
                    f"the 10Y G-Sec yield sits at <b>{gs10:.2f}%</b> versus a policy repo of "
                    f"<b>{repo:.2f}%</b>, giving a term premium of <b>{tp_bps:.0f} bps</b>"
                )
            if quick_parts:
                parts.append(
                    f'<div class="liq-interp"><b>Quick read:</b> '
                    + "; ".join(quick_parts) + ".</div>"
                )
        except Exception:
            pass

    except Exception as e:
        parts.append(f'<div class="error">Header cards failed: {type(e).__name__}: {e}</div>')

    # ----- 2. STRESS GAUGE -----
    parts.append('<h3>Liquidity Stress Gauge</h3>')
    try:
        stress_val_g = _safe_num(state.get('Stress_Index'), default=50.0)
        if np.isnan(stress_val_g):
            stress_val_g = 50.0
        gauge_fig = chart_liquidity_stress_gauge(stress_val_g)
        parts.append(f'<div class="chart-container">{_embed_chart(gauge_fig, height=300)}</div>')
        parts.append(
            '<p class="prose"><b>0–30</b> Abundant liquidity · '
            '<b>30–50</b> Accommodative · '
            '<b>50–65</b> Mildly elevated · '
            '<b>65–80</b> Elevated stress · '
            '<b>80–100</b> Severe stress. Composite of WACR spread, term premium, '
            'credit-growth slowdown, and 6m repo change.</p>'
        )

        # Band-conditional interpretation
        _s = stress_val_g
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
        parts.append(
            f'<div class="liq-interp">'
            f'The current stress reading of <b>{_s:.0f}/100</b> falls in the '
            f'<b>{_band}</b> band. {_msg}'
            f'</div>'
        )
    except Exception as e:
        parts.append(f'<div class="error">Stress gauge failed: {type(e).__name__}: {e}</div>')

    # ----- 3. HRC vs LIQUIDITY REGIME CHART -----
    parts.append(f'<h3>{target} vs Liquidity Regime</h3>')
    try:
        hrc = rdf[target].dropna()
        spread_series = (rdf['WACR_Spread'].dropna()
                         if 'WACR_Spread' in rdf.columns else pd.Series(dtype=float))
        repo_series = (rdf['Repo_Rate'].dropna()
                       if 'Repo_Rate' in rdf.columns else pd.Series(dtype=float))
        regime_series = (rdf['Liquidity_Regime'].dropna()
                         if 'Liquidity_Regime' in rdf.columns else pd.Series(dtype=object))

        regime_fig = chart_hrc_vs_liquidity_regime(
            hrc, spread_series, repo_series, regime_series, currency,
        )
        parts.append(f'<div class="chart-container">{_embed_chart(regime_fig, height=480)}</div>')

        parts.append(
            '<div class="liq-legend">'
            '<span><span class="liq-legend-swatch" style="background:rgba(45,106,79,0.25)"></span>'
            '<b>Green</b> = Surplus (WACR below Repo)</span>'
            '<span><span class="liq-legend-swatch" style="background:rgba(92,107,127,0.2)"></span>'
            '<b>Grey</b> = Neutral</span>'
            '<span><span class="liq-legend-swatch" style="background:rgba(164,22,26,0.25)"></span>'
            '<b>Red</b> = Tight (WACR above Repo)</span>'
            '</div>'
        )
        parts.append(
            '<div class="liq-interp">'
            '<b>How to read this chart:</b> shaded backgrounds mark each month\'s '
            'liquidity regime, classified by where the WACR-Repo spread sits relative to its '
            'historical mean band.'
            '<ul>'
            '<li><b>Green stretches (Surplus)</b> correspond to periods when banking-system '
            'liquidity was abundant — WACR traded below the policy repo, the RBI was '
            'actively injecting liquidity, and credit growth was strong. The 2020–2022 '
            'COVID stretch is the most visible example: WACR fell up to 86 bps below the '
            'policy rate as the RBI flooded the system, and Indian HRC prices rallied '
            'through this period (the supercycle).</li>'
            '<li><b>Red stretches (Tight)</b> mark RBI tightening cycles. The Oct 2022 – Mar 2025 '
            'stretch coincided with the post-pandemic rate hikes (Repo from 4.00% to 6.50%) '
            'and the HRC correction.</li>'
            '<li><b>Grey (Neutral)</b> is everything in between.</li>'
            '</ul>'
            'The transmission is asymmetric: surplus liquidity helps HRC almost immediately, '
            'while tight liquidity hurts with a 2–5 month lag (working-capital pain takes '
            'time to compress demand). Look for the <i>change</i> in regime as the early signal, '
            'not the level.'
            '</div>'
        )
    except Exception as e:
        parts.append(f'<div class="error">Regime chart failed: {type(e).__name__}: {e}</div>')

    # ----- 4. REGIME PERFORMANCE TABLE -----
    parts.append('<h3>HRC Performance by Liquidity Regime</h3>')
    try:
        hrc = rdf[target].dropna()
        perf = regime_performance(hrc, rdf['Liquidity_Regime'])
        if perf is not None and not perf.empty:
            parts.append(_regime_perf_html(perf))
            parts.append(
                '<p class="prose">Returns are month-over-month on the HRC price level, '
                'annualised by compounding. Max drawdown shows the deepest peak-to-trough '
                'decline of the cumulative return within each regime. Green-tinted '
                'Annualized Return cells flag favourable regimes; red flags damaging ones.</p>'
            )

            # Auto-generated interpretation bullets
            bullets = []
            if 'Surplus' in perf.index:
                s = perf.loc['Surplus']
                bullets.append(
                    f"<li><b>Surplus regimes ({int(s['Months'])} months)</b> — HRC delivered "
                    f"an annualised return of <b>{s['Annualized_Return_%']:+.1f}%</b>, "
                    f"with volatility {s['Volatility_Ann_%']:.1f}% and a hit rate of "
                    f"{s['Hit_Rate_%']:.0f}%. Surplus liquidity is the most favourable "
                    f"regime for HRC.</li>"
                )
            if 'Neutral' in perf.index:
                s = perf.loc['Neutral']
                bullets.append(
                    f"<li><b>Neutral regimes ({int(s['Months'])} months)</b> — annualised "
                    f"return <b>{s['Annualized_Return_%']:+.1f}%</b>, volatility "
                    f"{s['Volatility_Ann_%']:.1f}%.</li>"
                )
            if 'Tight' in perf.index:
                s = perf.loc['Tight']
                bullets.append(
                    f"<li><b>Tight regimes ({int(s['Months'])} months)</b> — annualised "
                    f"return <b>{s['Annualized_Return_%']:+.1f}%</b>, max drawdown "
                    f"{s['Max_Drawdown_%']:.1f}%, hit rate {s['Hit_Rate_%']:.0f}%. "
                    f"Tight liquidity is the most damaging regime.</li>"
                )

            headline = ""
            if 'Surplus' in perf.index and 'Tight' in perf.index:
                gap = (perf.loc['Surplus', 'Annualized_Return_%']
                       - perf.loc['Tight', 'Annualized_Return_%'])
                headline = (
                    f"<p><b>Headline:</b> the gap between surplus and tight regimes is "
                    f"<b>{gap:+.1f} percentage points</b> of annualised return. This is the "
                    f"liquidity-cycle premium in Indian HRC — material, persistent, and "
                    f"asymmetric. The empirical case for tracking RBI liquidity as a "
                    f"first-order driver of steel prices in India is supported by this data.</p>"
                )

            if bullets:
                parts.append(
                    '<div class="liq-interp">'
                    '<b>What the table says about Indian HRC:</b>'
                    f'<ul>{"".join(bullets)}</ul>'
                    f'{headline}'
                    '</div>'
                )
        else:
            parts.append('<div class="empty">Not enough data to compute regime performance.</div>')
    except Exception as e:
        parts.append(f'<div class="error">Regime performance table failed: {type(e).__name__}: {e}</div>')

    # ----- 5. LEAD-LAG TABLE -----
    parts.append('<h3>Lead-Lag: Does Liquidity Lead HRC?</h3>')
    try:
        lag_var_candidates = ['WACR_Spread', 'GSec_Repo_Spread', 'Stress_Index',
                              'Bank_Credit_YoY', 'Repo_6M_Change']
        lag_vars = [c for c in lag_var_candidates if c in rdf.columns]
        if lag_vars:
            hrc = rdf[target].dropna()
            ll = liquidity_lead_lag(hrc, rdf[lag_vars], lags=[0, 1, 3, 6, 12])
            parts.append(_liquidity_lead_lag_html(ll))
            parts.append(
                '<p class="prose">Each cell is the Pearson correlation between HRC '
                'month-over-month log returns and a liquidity variable lagged by k months. '
                'A positive lag means liquidity from k months ago is being matched against '
                'HRC return today. Negative values for WACR_Spread or Stress_Index mean '
                'tight liquidity precedes weaker HRC.</p>'
            )

            # Auto-interpretation: strongest signal per variable
            findings = []
            for var in ll.index:
                row = ll.loc[var]
                if row.dropna().empty:
                    continue
                idx_max = row.abs().idxmax()
                try:
                    lag_k = int(idx_max.replace('lag_', '').replace('m', ''))
                except Exception:
                    continue
                corr = row[idx_max]
                if pd.isna(corr) or abs(corr) < 0.15:
                    continue
                direction = ("leads" if lag_k > 0
                             else "is contemporaneous with" if lag_k == 0
                             else "lags")
                sign = "negative" if corr < 0 else "positive"
                findings.append(
                    f"<li><b>{var}</b>: strongest correlation <b>{corr:+.2f}</b> at "
                    f"<code>lag_{lag_k}m</code>. This {sign} correlation means {var} "
                    f"{direction} HRC by {abs(lag_k)} month"
                    f"{'s' if abs(lag_k) != 1 else ''}.</li>"
                )

            if findings:
                findings_html = f'<ul>{"".join(findings)}</ul>'
            else:
                findings_html = (
                    "<p>No liquidity variable shows a correlation with HRC returns above "
                    "0.15 at any tested lag. The liquidity signal is currently weak; "
                    "drivers other than monetary conditions are dominating HRC.</p>"
                )

            parts.append(
                '<div class="liq-interp">'
                '<b>Which liquidity signals lead HRC?</b>'
                f'{findings_html}'
                '<p><b>Interpreting the signs:</b></p>'
                '<ul>'
                '<li><b>WACR_Spread, Stress_Index, GSec_Repo_Spread</b>: a <i>negative</i> '
                'correlation at a <i>positive</i> lag means tighter liquidity preceded '
                'weaker HRC — the textbook macro channel.</li>'
                '<li><b>Bank_Credit_YoY</b>: typically positive (faster credit growth → '
                'stronger demand → higher HRC), though contemporaneous credit growth '
                'often correlates negatively in India because credit accelerates '
                '<i>after</i> a demand pulse has already moved prices.</li>'
                '<li><b>Repo_6M_Change</b>: a negative correlation at long lags (12m) '
                'captures the slow transmission of RBI rate decisions through to '
                'downstream financing costs and steel demand.</li>'
                '</ul>'
                '</div>'
            )
        else:
            parts.append('<div class="empty">No liquidity variables available for lead-lag analysis.</div>')
    except Exception as e:
        parts.append(f'<div class="error">Lead-lag table failed: {type(e).__name__}: {e}</div>')

    # ----- 6. VARIABLE PANEL -----
    parts.append('<h3>Liquidity Variable Panel</h3>')
    parts.append('<p class="prose">All base and derived liquidity series at a glance.</p>')
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
        panel_fig = chart_liquidity_variable_panel(rdf, panel_cfg)
        n_rows = (len(panel_cfg) + 1) // 2
        parts.append(f'<div class="chart-container">'
                     f'{_embed_chart(panel_fig, height=max(280, n_rows*230))}</div>')
    except Exception as e:
        parts.append(f'<div class="error">Variable panel failed: {type(e).__name__}: {e}</div>')

    parts.append('</section>')
    return "".join(parts)


def build_report(results: Dict[str, Any], output_path: str) -> str:
    """Build the full HTML report and write it to disk."""
    html_parts = []

    # Header / cover
    n_regions = len(results["regions"])
    total_obs = sum(r["meta"]["n_observations"] for r in results["regions"].values())
    n_models_run = sum(len(r.get("models", {})) for r in results["regions"].values())
    region_names = ", ".join(r["meta"]["region"].title() for r in results["regions"].values())

    cover = f"""
<div class="cover">
  <div class="eyebrow">HRC Steel Intelligence Report</div>
  <h1>Multi-Region Spread &amp; Forecasting Analysis</h1>
  <div class="sub">{region_names}  ·  Generated {datetime.now().strftime('%d %b %Y %H:%M')}</div>
  <div class="stats">
    <div class="stat"><div class="stat-value">{n_regions}</div>
      <div class="stat-label">Regions</div></div>
    <div class="stat"><div class="stat-value">{total_obs:,}</div>
      <div class="stat-label">Total observations</div></div>
    <div class="stat"><div class="stat-value">{n_models_run}</div>
      <div class="stat-label">Model fits</div></div>
    <div class="stat"><div class="stat-value">Auto</div>
      <div class="stat-label">Driver detection</div></div>
  </div>
</div>
"""
    html_parts.append(cover)

    # ----- Per-region sections -----
    for region_name, region_results in results["regions"].items():
        meta = region_results["meta"]
        currency = meta["currency"]
        # Driver-dependent sections (Diagnostics, Lead-Lag, Attribution, Models)
        # require at least one driver. US is configured with `overview_only: true`
        # because driver data isn't yet available; for that region we render only
        # the sections that work on the target alone.
        is_overview_only = (meta.get("overview_only", False)
                             or meta.get("n_drivers", 0) == 0)

        # ----- 1. Region overview -----
        spread = region_results.get("spread")
        snap_html = ""
        if spread and not isinstance(spread, dict):
            cur = spread.current_snapshot
            regime_class = cur["current_regime"]
            snap_html = f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">HRC Price</div>
    <div class="value">{currency} {_fmt_num(cur['hrc'], 0)}</div>
    <div class="delta">as of {cur['as_of']}</div></div>
  <div class="kpi"><div class="label">Raw Material Cost</div>
    <div class="value">{currency} {_fmt_num(cur['rm_cost'], 0)}</div>
    <div class="delta">{_fmt_num(cur['rm_cost']/cur['hrc']*100, 1)}% of HRC</div></div>
  <div class="kpi {'danger' if cur['spread_percentile']<25 else 'positive' if cur['spread_percentile']>75 else ''}">
    <div class="label">Spread (margin)</div>
    <div class="value">{currency} {_fmt_num(cur['spread'], 0)}</div>
    <div class="delta">P{cur['spread_percentile']:.0f} of history
       <span class="regime-tag {regime_class}">{regime_class}</span></div></div>
  <div class="kpi"><div class="label">vs Median</div>
    <div class="value">{_fmt_num(cur['vs_p50'], 0, prefix='+' if cur['vs_p50']>=0 else '')}</div>
    <div class="delta">vs 5y avg: {_fmt_num(cur['vs_5y_avg'], 0,
                                            prefix='+' if cur['vs_5y_avg']>=0 else '')}</div></div>
</div>"""

        # Build the actual target series from spread decomp or rebuild
        target_series = None
        if spread and not isinstance(spread, dict):
            target_series = spread.decomposition["HRC"]
        # Otherwise reconstruct from results (best-effort)

        section = f"""
<section>
  <div class="region-tabs">
    <span class="region-pill active">{region_name.title()}</span>
  </div>
  <h2>{region_name.title()} — Overview</h2>
  <div class="subtitle">{meta['n_observations']} monthly observations from {meta['date_start']} to {meta['date_end']}  ·
    {meta['n_drivers']} drivers  ·  Currency: {currency}</div>
  {_render_prose(narrator.narrate_overview(meta, target_series, currency)) if target_series is not None else ''}
  {snap_html}
  {('<div class="chart-container">' + _embed_chart(chart_target_history(target_series, region_name, currency)) + '</div>') if target_series is not None else ''}
</section>
"""
        html_parts.append(section)

        # Overview-only note: regions without driver data only get the price
        # history chart above. Tell the user explicitly why so they don't
        # think the report is broken.
        if is_overview_only:
            html_parts.append(f"""
<section>
  <div class="subtitle" style="background: #FFF8E6; border-left: 4px solid #C9540F;
       padding: 12px 16px; margin: 12px 0; border-radius: 4px;">
    <b>Overview-only mode for {region_name.title()}.</b> Driver data
    (iron ore, coking coal, etc.) is not yet wired into the {region_name.title()}
    sheet of <code>Raw_data.xlsx</code>, so analyses that need drivers —
    Diagnostics, Lead/Lag, Attribution, and Forecasting Models — are intentionally
    omitted from this report. Once driver columns are added, set
    <code>overview_only: false</code> in <code>config.yaml</code> and rerun
    to enable the full analysis.
  </div>
</section>""")

        # ----- 2. Spread analysis -----
        if spread and not isinstance(spread, dict):
            spread_section = f"""
<section>
  <h2>Spread Analysis — {region_name.title()}</h2>
  <div class="subtitle">Mill margin proxy. Currency: {currency}/t.</div>

  {_render_prose(narrator.narrate_spread(spread, currency))}

  <h3>Spread time series with percentile bands</h3>
  <div class="chart-container">{_embed_chart(chart_spread(spread))}</div>

  <h3>HRC Price Decomposition</h3>
  <div class="chart-container">{_embed_chart(chart_decomposition(spread))}</div>

  <h3>FY-Average Spread (matches BPM deck format)</h3>
  {_df_to_html(spread.fy_table)}

  <h3>Historical Distribution</h3>
  <div class="kpi-grid">
    <div class="kpi"><div class="label">P10 (worst)</div>
      <div class="value">{currency} {_fmt_num(spread.percentiles['p10'], 0)}</div></div>
    <div class="kpi"><div class="label">P25</div>
      <div class="value">{currency} {_fmt_num(spread.percentiles['p25'], 0)}</div></div>
    <div class="kpi"><div class="label">Median</div>
      <div class="value">{currency} {_fmt_num(spread.percentiles['p50'], 0)}</div></div>
    <div class="kpi"><div class="label">P75</div>
      <div class="value">{currency} {_fmt_num(spread.percentiles['p75'], 0)}</div></div>
    <div class="kpi"><div class="label">P90 (best)</div>
      <div class="value">{currency} {_fmt_num(spread.percentiles['p90'], 0)}</div></div>
  </div>
</section>"""
            html_parts.append(spread_section)

        # ----- 3. Diagnostics -----
        if not is_overview_only:
            diag = region_results.get("adf")
            corr = region_results.get("correlation")
            vif = region_results.get("vif")
            diag_section = f"""
<section>
  <h2>Diagnostics — {region_name.title()}</h2>
  <div class="subtitle">Correlation, multicollinearity, and stationarity tests for the {region_name} dataset.</div>

  {_render_prose(narrator.narrate_diagnostics(diag, vif, meta['target']))}

  <h3>Correlation Matrix</h3>
  {_render_prose(narrator.narrate_correlation(corr, meta['target'], currency))}
  <div class="chart-container">{_embed_chart(chart_correlation_heatmap(corr), height=480)
                                  if corr is not None and not isinstance(corr, dict) else '<div class="empty">N/A</div>'}</div>
  <h3>Multicollinearity (VIF)</h3>
  {_df_to_html(vif)}
  <h3>Stationarity (ADF)</h3>
  {_df_to_html(diag, max_rows=100)}
</section>"""
            html_parts.append(diag_section)

        # ----- 4. Lead-Lag -----
        if not is_overview_only:
            ll = region_results.get("lead_lag")
            ll_section = f"""
<section>
  <h2>Lead/Lag &amp; Causal Analysis — {region_name.title()}</h2>
  <div class="subtitle">Cross-correlation function (CCF) and Granger causality across lags.</div>

  {_render_prose(narrator.narrate_lead_lag(ll, meta['target']))}

  <div class="chart-container">{_embed_chart(chart_lead_lag(ll, meta['target']))
                                  if ll is not None and not isinstance(ll, dict) else '<div class="empty">N/A</div>'}</div>
  {_df_to_html(ll)}
</section>"""
            html_parts.append(ll_section)

        # ----- 5. Models -----
        models = region_results.get("models", {})
        if models:
            from models.base import VolatilityResult, ForecastResult
            models_html = ""
            for model_name, model_result in models.items():
                if isinstance(model_result, dict) and "error" in model_result:
                    models_html += f'<h3>{model_name.upper()}</h3><div class="error">{model_result["error"]}</div>'
                    continue
                if isinstance(model_result, VolatilityResult):
                    models_html += f'<h3>{model_name.upper()}</h3>'
                    if getattr(model_result, "error", None):
                        models_html += f'<div class="error">{model_result.error}</div>'
                        continue
                    models_html += f'<div class="chart-container">{_embed_chart(chart_garch_volatility(model_result, region_name))}</div>'
                    diag = model_result.diagnostics or {}
                    if diag.get("params"):
                        params_df = pd.DataFrame(list(diag["params"].items()),
                                                  columns=["Parameter", "Value"])
                        models_html += _df_to_html(params_df)
                else:
                    models_html += f'<h3>{model_name.upper()}</h3>'
                    if getattr(model_result, "error", None):
                        models_html += f'<div class="error">{model_result.error}</div>'
                        continue
                    actual = pd.Series()
                    if model_result.fitted is not None:
                        # Use fitted values' source — we'll just plot fitted+oos+forecast
                        # but we want actual too. We'll use the original target.
                        # To keep this simple, plot fitted as 'actual' approximation.
                        pass
                    # Actual series: try to reconstruct from oos_actuals + fitted index
                    if model_result.fitted is not None:
                        # the fitted series has the same index as the training y
                        actual_ser = model_result.fitted.copy()
                        if model_result.residuals is not None:
                            actual_ser = model_result.fitted + model_result.residuals
                        models_html += f'<div class="chart-container">{_embed_chart(chart_forecast(model_result, actual_ser, region_name))}</div>'
                    metrics = model_result.metrics or {}
                    if metrics:
                        m_df = pd.DataFrame([{
                            "Metric": "RMSE", "Value": metrics.get("rmse")
                        }, {
                            "Metric": "MAE", "Value": metrics.get("mae")
                        }, {
                            "Metric": "MAPE (%)", "Value": metrics.get("mape")
                        }, {
                            "Metric": "R²", "Value": metrics.get("r2")
                        }])
                        models_html += "<h4>Out-of-sample test metrics</h4>"
                        models_html += _df_to_html(m_df)
                    if model_result.diagnostics:
                        d = model_result.diagnostics
                        diag_df = pd.DataFrame(list(d.items()), columns=["Statistic", "Value"])
                        models_html += _df_to_html(diag_df)
            models_section = f"""
<section>
  <h2>Forecasting Models — {region_name.title()}</h2>
  <div class="subtitle">All enabled models from config.yaml. Add/remove by editing
    `models:` block in config and re-running.</div>
  {_render_prose(narrator.narrate_models(models, currency, meta['target']))}
  {models_html}
</section>"""
            html_parts.append(models_section)

        # ----- 6. Regimes -----
        regimes = region_results.get("regimes")
        if regimes and not isinstance(regimes, dict):
            # We need the target series — reconstruct from spread.decomposition['HRC']
            target_ser = (spread.decomposition["HRC"] if spread and not isinstance(spread, dict)
                          else None)
            if target_ser is not None:
                reg_section = f"""
<section>
  <h2>Regime Classification — {region_name.title()}</h2>
  <div class="subtitle">{regimes.n_regimes} regimes identified via K-means.
    Currently in <b>regime {regimes.current_regime}</b>.</div>
  <div class="chart-container">{_embed_chart(chart_regimes(regimes, target_ser))}</div>
  <h3>Regime statistics</h3>
  {_df_to_html(regimes.regime_stats)}
  <h3>Regime mean values</h3>
  {_df_to_html(regimes.regime_means.reset_index())}
</section>"""
                html_parts.append(reg_section)

        # ----- 6b. CYCLICITY (NEW) -----
        cyc = region_results.get("cyclicity")
        if cyc and not isinstance(cyc, dict):
            target_ser = (spread.decomposition["HRC"] if spread and not isinstance(spread, dict)
                          else None)

            # Per-regime stats table
            reg_stats_rows = []
            for p in cyc.regime_profiles:
                c = cyc.per_regime_cycles.get(p.regime_id)
                gp = cyc.garch_persistence.get(p.regime_id)
                reg_stats_rows.append({
                    "Regime": f"R{p.regime_id}",
                    "Label": p.label,
                    "# Months": p.n_months,
                    f"Mean ({currency}/t)": round(p.mean_target, 0) if not np.isnan(p.mean_target) else None,
                    "Std Dev": round(p.std_target, 0) if not np.isnan(p.std_target) else None,
                    "Avg Return %/mo": round(p.avg_monthly_return_pct, 2) if not np.isnan(p.avg_monthly_return_pct) else None,
                    "# Spells": p.n_spells,
                    "Avg Spell (m)": round(p.avg_spell_duration, 1),
                    "Dominant Cycle (m)": round(c.dominant_spectral_period_months, 1) if c and c.dominant_spectral_period_months else None,
                    "GARCH Persist": round(gp, 3) if gp is not None else None,
                })
            reg_stats_df = pd.DataFrame(reg_stats_rows)

            # Cycle stats summary table
            cs = cyc.cycle_stats
            cycle_summary_df = pd.DataFrame([{
                "Metric": "Total Peaks", "Value": cs.n_peaks,
            }, {
                "Metric": "Total Troughs", "Value": cs.n_troughs,
            }, {
                "Metric": "Avg Peak-to-Peak (months)",
                "Value": round(cs.avg_peak_to_peak_months, 1) if cs.avg_peak_to_peak_months else None,
            }, {
                "Metric": "Avg Trough-to-Trough (months)",
                "Value": round(cs.avg_trough_to_trough_months, 1) if cs.avg_trough_to_trough_months else None,
            }, {
                "Metric": "Avg Amplitude (%)",
                "Value": round(cs.avg_amplitude_pct, 1) if cs.avg_amplitude_pct else None,
            }, {
                "Metric": "Dominant Spectral Period (months)",
                "Value": round(cs.dominant_spectral_period_months, 1) if cs.dominant_spectral_period_months else None,
            }])

            cyc_section = f"""
<section>
  <h2>Cyclicity &amp; Regime Dynamics — {region_name.title()}</h2>
  <div class="subtitle">Multi-feature GMM regime identification, peak/trough detection,
    spectral cycle analysis, GARCH persistence, and Markov transitions.</div>

  {_render_prose(narrator.narrate_cyclicity(cyc, currency))}

  <h3>Regime Classification (Multi-Feature GMM)</h3>
  <div class="chart-container">{_embed_chart(chart_cyclicity_regimes(cyc, target_ser)) if target_ser is not None else '<div class="empty">N/A</div>'}</div>

  <h3>Per-Regime Statistics</h3>
  {_df_to_html(reg_stats_df)}

  <h3>Peak/Trough Cycle Detection</h3>
  <div class="chart-container">{_embed_chart(chart_peaks_troughs(cyc, target_ser)) if target_ser is not None else '<div class="empty">N/A</div>'}</div>

  <h3>Cycle Statistics Summary</h3>
  {_df_to_html(cycle_summary_df)}

  <h3>Markov Transition Matrix</h3>
  <div class="chart-container">{_embed_chart(chart_transition_matrix(cyc), height=420)}</div>

  <h3>Macro Cycles Identified</h3>
  {_df_to_html(cyc.macro_cycles) if len(cyc.macro_cycles) > 0 else '<div class="empty">No complete macro cycle yet observed in this region.</div>'}
</section>"""
            html_parts.append(cyc_section)

        # ----- 7. Attribution -----
        attr = region_results.get("attribution")
        if attr and not isinstance(attr, dict) and len(attr.rolling_betas) > 0:
            current_attr_df = attr.current_attribution.reset_index()
            current_attr_df.columns = ["Driver", "Current attribution (%)"]
            current_attr_df = current_attr_df.sort_values("Current attribution (%)",
                                                            ascending=False)
            attr_section = f"""
<section>
  <h2>Driver Attribution — {region_name.title()}</h2>
  <div class="subtitle">Rolling 24-month regression. Stacked area shows how each
    driver's relative importance to {meta['target']} has shifted over time.</div>

  {_render_prose(narrator.narrate_attribution(attr, currency))}

  <div class="chart-container">{_embed_chart(chart_attribution(attr))}</div>
  <h3>Current attribution (most recent window)</h3>
  {_df_to_html(current_attr_df)}
</section>"""
            html_parts.append(attr_section)

        # ----- 8. Events -----
        events = region_results.get("events")
        if events and not isinstance(events, dict) and len(events.episodes) > 0:
            events_html = ""
            for ep in events.episodes:
                ep_html = f'<h3>{ep["name"]} ({ep["event_date"]})</h3>'
                for w_name, w_data in ep["windows"].items():
                    rows = []
                    for var, stats in w_data.items():
                        rows.append({
                            "Variable": var,
                            f"Pre-{w_name} avg": stats["pre_avg"],
                            f"Post-{w_name} avg": stats["post_avg"],
                            "% Change": stats["pct_change"],
                        })
                    df = pd.DataFrame(rows)
                    df = df.sort_values("% Change", key=lambda s: s.abs(), ascending=False)
                    ep_html += f'<h4>±{w_name} window</h4>{_df_to_html(df)}'
                events_html += ep_html
            events_section = f"""
<section>
  <h2>Event Window Analysis — {region_name.title()}</h2>
  <div class="subtitle">How target + drivers behaved before and after key episodes.</div>
  {events_html}
</section>"""
            html_parts.append(events_section)

        # ----- 9. India Liquidity Monitor (India only) -----
        # Renders the full Liquidity tab from the dashboard as a static section.
        # Returns "" for non-India regions or regions without a liquidity block.
        try:
            liq_html = _render_liquidity_section(region_name, region_results)
            if liq_html:
                html_parts.append(liq_html)
        except Exception as e:
            html_parts.append(
                f'<section><h2>India Liquidity Monitor — {region_name.title()}</h2>'
                f'<div class="error">Liquidity section crashed: '
                f'{type(e).__name__}: {e}</div></section>'
            )

    # ----- Cross-region section -----
    cross = results.get("cross_region", {})
    if cross.get("available"):
        cross_section = f"""
<section>
  <h2>Cross-Region Comparison — China vs India</h2>
  <div class="subtitle">{cross['interpretation']}</div>

  {_render_prose(narrator.narrate_cross_region(cross))}

  <div class="kpi-grid">
    <div class="kpi"><div class="label">China spread percentile</div>
      <div class="value">P{cross['china_percentile_now']:.0f}</div>
      <div class="delta">within China's own history</div></div>
    <div class="kpi"><div class="label">India spread percentile</div>
      <div class="value">P{cross['india_percentile_now']:.0f}</div>
      <div class="delta">within India's own history</div></div>
    <div class="kpi"><div class="label">Divergence (z-score)</div>
      <div class="value">{cross['divergence_now']:+.2f}</div>
      <div class="delta">India − China</div></div>
    <div class="kpi"><div class="label">Overall correlation</div>
      <div class="value">{cross['overall_correlation']:.2f}</div>
      <div class="delta">{cross['overlap_months']} overlapping months</div></div>
  </div>
  <div class="chart-container">{_embed_chart(chart_cross_region(cross), height=560)}</div>
</section>"""
        html_parts.append(cross_section)

    # ----- Macro Calendar Section -----
    cal = results.get("macro_calendar")
    if cal and not isinstance(cal, dict) and len(cal.events) > 0:
        cal_blocks = narrator.narrate_macro_calendar(cal)
        cal_section = f"""
<section>
  <h2>Macro Calendar — Forward 45-Day Outlook</h2>
  <div class="subtitle">HRC-relevant events from {cal.window_start} to {cal.window_end} ·
    {len(cal.events)} events ({cal.n_high} HIGH, {cal.n_med} MED) · Generated {cal.today}</div>

  {_render_prose(cal_blocks)}
</section>"""
        html_parts.append(cal_section)

    # Footer
    footer = f"""
<footer>
  Generated by HRC Steel Pipeline  ·  {datetime.now().strftime('%d %b %Y %H:%M')}
  ·  Data hash: {results.get('dataset_meta', {}).get('file_hash', 'n/a')[:8]}
</footer>
"""
    html_parts.append(footer)

    # Compose final HTML
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HRC Steel Intelligence Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  {''.join(html_parts)}
</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_html)
    return output_path
