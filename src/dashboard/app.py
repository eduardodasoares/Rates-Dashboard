import os

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.signals.signal_engine import SignalEngine

FRED_KEY = os.environ.get("FRED_API_KEY")
if not FRED_KEY:
    raise EnvironmentError("FRED_API_KEY environment variable is not set.")
REFRESH_INTERVAL_MS = 300_000  # 5 minutes — FRED data is daily


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline():
    df = DataLoader(FRED_KEY).load_treasury_data()
    df = FeatureEngineer.add_curve_features(df)
    df["composite_score"] = SignalEngine.composite_score(df)
    df["gated_score"]     = SignalEngine.gated_score(df)
    nop = SignalEngine.non_overlapping_performance(df)
    return df, nop


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Rates Signal Dashboard",
)


# ── Card helpers ──────────────────────────────────────────────────────────────

def _yield_card(tenor: str, value: float, chg_bps: float) -> dbc.Card:
    color = "text-danger" if chg_bps > 0 else "text-success"
    return dbc.Card(
        dbc.CardBody([
            html.H6(tenor, className="text-muted mb-1"),
            html.H4(f"{value:.2f}%", className="mb-0"),
            html.Small(f"{chg_bps:+.1f} bps", className=color),
        ]),
        className="text-center m-1 h-100",
    )


def _score_card(score: float, gated: float, t_steep: float, t_flat: float,
                cr_z60: float, macro_reg: str, macro_sc: float) -> dbc.Card:
    if score >= t_steep:
        direction, badge_color = "STEEPENER ▲", "success"
    elif score <= -t_flat:
        direction, badge_color = "FLATTENER ▼", "danger"
    else:
        direction, badge_color = "NEUTRAL ─", "secondary"

    scalar_text = {
        1.0:  "Full weight",
        0.5:  "Half weight (gate active)",
        0.25: "Qtr weight (gate active)",
    }.get(macro_sc, f"{macro_sc:.2f}× weight")

    macro_color = {
        "Neutral": "secondary", "Hike Priced": "warning",
        "Hike Priced+": "danger", "Cut Priced": "info", "Cut Priced+": "primary",
    }.get(macro_reg, "secondary")

    bar_pct = min(abs(score) / 4.0 * 100, 100)

    return dbc.Card(
        dbc.CardBody([
            html.H6("Composite Score", className="text-muted mb-1"),
            html.H1(f"{score:+.2f}", className="mb-1 fw-bold"),
            html.Small(f"Gated: {gated:+.2f}  ·  {scalar_text}", className="text-muted d-block mb-2"),
            dbc.Progress(value=bar_pct, color="success" if score >= 0 else "danger",
                         style={"height": "6px"}, className="mb-2"),
            dbc.Badge(direction, color=badge_color, className="me-2 fs-6"),
            dbc.Badge(macro_reg, color=macro_color, className="fs-6"),
            html.Hr(style={"borderColor": "#444", "margin": "10px 0"}),
            html.Small(
                f"Entry bar:  steep ≥ {t_steep:.2f}  /  flat ≥ {t_flat:.2f}",
                className="text-muted d-block",
            ),
            html.Small(
                f"Carry z60: {cr_z60:+.2f}σ  (base 1.30)",
                className="text-muted",
            ),
        ]),
        className="text-center m-1 h-100",
    )


def _component_card(label: str, contrib: float, detail: str) -> dbc.Card:
    if contrib > 0.05:
        val_class = "text-success"
        sign = "+"
    elif contrib < -0.05:
        val_class = "text-danger"
        sign = ""
    else:
        val_class = "text-muted"
        sign = ""
    return dbc.Card(
        dbc.CardBody([
            html.H6(label, className="text-muted mb-1", style={"fontSize": "11px"}),
            html.H3(f"{sign}{contrib:.2f}", className=f"{val_class} mb-1"),
            html.Small(detail, className="text-muted"),
        ]),
        className="text-center m-1 h-100",
    )


def _sizing_card(direction_label: str, score_frac: float, vol_sc: float,
                 fwd_vol: float, risk_b: float, dv01_rec: float) -> dbc.Card:
    if "STEEP" in direction_label:
        dir_color = "text-success"
    elif "FLAT" in direction_label:
        dir_color = "text-danger"
    else:
        dir_color = "text-muted"
    return dbc.Card(
        dbc.CardBody([
            html.H6("Position Sizing", className="text-muted mb-1"),
            html.H4(direction_label, className=f"{dir_color} mb-2"),
            html.Small(f"Score frac: {score_frac:.2f}  ·  Vol scalar: {vol_sc:.2f}×",
                       className="text-muted d-block"),
            html.Small(f"Fwd vol: {fwd_vol:.2f} bps  ·  Risk: {risk_b:.1f} bps",
                       className="text-muted d-block"),
            html.Small(f"DV01 @ $100k max: ${dv01_rec:,.0f}/bp",
                       className="text-muted d-block"),
        ]),
        className="text-center m-1 h-100",
    )


def _carry_card(carry_5d: float, roll_5d: float, cr_5d: float,
                sofr: float, direction_label: str) -> dbc.Card:
    if "STEEP" in direction_label:
        trade_sign = +1
    elif "FLAT" in direction_label:
        trade_sign = -1
    else:
        trade_sign = None

    cr_trade  = cr_5d * trade_sign if pd.notna(cr_5d) and trade_sign is not None else np.nan
    cr_annual = cr_5d * 252 / 5   if pd.notna(cr_5d) else np.nan
    cr_color  = "text-success" if pd.notna(cr_trade) and cr_trade > 0 else "text-danger"

    return dbc.Card(
        dbc.CardBody([
            html.H6("Carry & Roll  (steepener, 5d)", className="text-muted mb-1"),
            html.H4(f"{cr_5d:+.3f} bps" if pd.notna(cr_5d) else "N/A", className="mb-1"),
            html.Small(f"Carry: {carry_5d:+.3f}  |  Roll: {roll_5d:+.3f}",
                       className="text-muted d-block") if pd.notna(carry_5d) else html.Span(),
            html.Small(f"Ann: {cr_annual:+.1f} bps/yr  ·  SOFR {sofr:.2f}%",
                       className="text-muted d-block") if pd.notna(cr_annual) and pd.notna(sofr) else html.Span(),
            html.Small(f"Your trade: {cr_trade:+.3f} bps",
                       className=f"{cr_color} d-block") if pd.notna(cr_trade) else html.Span(),
        ]),
        className="text-center m-1 h-100",
    )


def _slope_card(label: str, value_bps: float, z20: float, z60: float, regime: str) -> dbc.Card:
    badge_color = {
        "Bear Steepener": "danger", "Bear Flattener": "warning",
        "Bull Steepener": "info",   "Bull Flattener": "success",
    }.get(regime, "secondary")
    return dbc.Card(
        dbc.CardBody([
            html.H6(label, className="text-muted mb-1"),
            html.H4(f"{value_bps:+.0f} bps", className="mb-1"),
            html.Small(f"z20: {z20:.2f}  |  z60: {z60:.2f}", className="text-muted d-block mb-1"),
            dbc.Badge(regime, color=badge_color, className="fs-6"),
        ]),
        className="text-center m-1 h-100",
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def _cutoff(df: pd.DataFrame, years: int = 2) -> pd.DataFrame:
    return df[df.index >= df.index.max() - pd.DateOffset(years=years)]


def _dark_layout(title: str, **kwargs) -> dict:
    return dict(
        title=title,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        **kwargs,
    )


def _score_history_chart(df: pd.DataFrame) -> go.Figure:
    """Composite score with dynamic entry thresholds overlaid."""
    d = _cutoff(df, years=1)
    q5_thr = float(df["composite_score"].quantile(0.8))
    q1_thr = float(df["composite_score"].quantile(0.2))

    if "carry_roll_5d_z60" in d.columns:
        ts, tf = SignalEngine.effective_threshold(d)
        ts = ts.reindex(d.index)
        tf = tf.reindex(d.index)
    else:
        ts = tf = pd.Series(1.3, index=d.index)

    fig = go.Figure()
    fig.add_hrect(y0=q5_thr, y1=6,     fillcolor="#00c853", opacity=0.08, line_width=0)
    fig.add_hrect(y0=-6,     y1=q1_thr, fillcolor="#d50000", opacity=0.08, line_width=0)

    fig.add_trace(go.Scatter(x=d.index, y=d["composite_score"],
                             name="Score", line=dict(color="#ffffff", width=2)))
    fig.add_trace(go.Scatter(x=d.index, y=ts.values, name="Steep threshold",
                             line=dict(color="#00c853", width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=d.index, y=-tf.values, name="Flat threshold",
                             line=dict(color="#d50000", width=1.2, dash="dash")))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(**_dark_layout(
        "Composite Score — 1Y  (dashed lines = carry-adjusted entry thresholds)"
    ))
    return fig


def _component_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked contribution of each score component over 1Y."""
    d = _cutoff(df, years=1)

    regime_map = {
        "Bear Steepener": -1, "Bull Steepener": -1,
        "Bear Flattener": +1, "Bull Flattener": +1, "Unknown": 0,
    }
    c_regime   = d["regime"].map(regime_map).fillna(0)
    c_voladj   = (-d["2s10s_voladj"] / 2).clip(-1, 1).fillna(0)
    c_vix      = (d["VIX_z20"] / 2).clip(-1, 1).fillna(0)
    c_frontend = (d["1y2y_z20"] / 2).clip(-1, 1).fillna(0)

    fig = go.Figure()
    for label, series, color in [
        ("Regime (contrarian)",  c_regime,   "#00bcd4"),
        ("Vol-adj (contrarian)", c_voladj,   "#ff9800"),
        ("VIX momentum",         c_vix,      "#e91e63"),
        ("Front-end 1y2y",       c_frontend, "#9c27b0"),
    ]:
        fig.add_trace(go.Bar(x=d.index, y=series.values,
                             name=label, marker_color=color, opacity=0.8))

    fig.add_trace(go.Scatter(x=d.index, y=d["composite_score"],
                             name="Total score", line=dict(color="white", width=2)))
    fig.update_layout(**_dark_layout("Score Component Breakdown — 1Y"), barmode="relative")
    return fig


def _pnl_chart(nop: pd.DataFrame) -> go.Figure:
    """Cumulative carry-adjusted P&L from non-overlapping directional signals (1990+)."""
    d = nop[
        (nop["direction"] != "flat") &
        (nop.index >= "1990-01-01")
    ].dropna(subset=["aligned_adj_bps"])

    if len(d) == 0:
        return go.Figure()

    cum_adj = d["aligned_adj_bps"].cumsum()
    cum_raw = d["aligned_bps"].cumsum()

    sh_adj = (d["aligned_adj_bps"].mean() / d["aligned_adj_bps"].std() * np.sqrt(252 / 5)
              if d["aligned_adj_bps"].std() > 0 else 0)
    sh_raw = (d["aligned_bps"].mean() / d["aligned_bps"].std() * np.sqrt(252 / 5)
              if d["aligned_bps"].std() > 0 else 0)
    hit = d["hit_adj"].dropna().mean()
    n   = len(d)

    bar_colors = [
        "rgba(0,230,118,0.55)" if v > 0 else "rgba(255,23,68,0.45)"
        for v in d["aligned_adj_bps"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d.index, y=d["aligned_adj_bps"].values,
        name="Per-trade P&L (adj)", marker_color=bar_colors,
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=cum_adj.index, y=cum_adj.values,
        name=f"Cumul adj  (Sharpe {sh_adj:.2f})",
        line=dict(color="#00e676", width=2.5),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=cum_raw.index, y=cum_raw.values,
        name=f"Cumul raw  (Sharpe {sh_raw:.2f})",
        line=dict(color="#aaaaaa", width=1.5, dash="dash"),
        yaxis="y2",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        **_dark_layout(
            f"Signal P&L — Non-Overlapping Trades (1990–present)  "
            f"·  {n} trades  ·  hit {hit:.0%}  ·  adj Sharpe {sh_adj:.2f}"
        ),
        yaxis=dict(title="Per-trade bps"),
        yaxis2=dict(title="Cumulative bps", overlaying="y", side="right"),
        barmode="relative",
    )
    return fig


def _zscore_chart(df: pd.DataFrame) -> go.Figure:
    """Slope z-scores with ±1σ / ±2σ reference bands."""
    d = _cutoff(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["2s10s_z20"], name="2s10s z20",
                             line=dict(color="#00bcd4", width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["5s30s_z20"], name="5s30s z20",
                             line=dict(color="#ff9800", width=1.5)))
    for level, style in [(2, "dot"), (1, "dash"), (-1, "dash"), (-2, "dot")]:
        fig.add_hline(y=level, line_dash=style, line_color="gray", opacity=0.4,
                      annotation_text=f"{level:+d}σ", annotation_position="right")
    fig.update_layout(**_dark_layout("Slope Z-Scores (20d) with ±1σ / ±2σ Bands"))
    return fig


def _carry_chart(df: pd.DataFrame) -> go.Figure:
    """Carry+roll history and its z-score (carry environment indicator)."""
    d = _cutoff(df, years=2)
    cr = d["carry_roll_5d"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cr.index, y=cr.values,
        name="C+R total (steepener)",
        line=dict(color="#00bcd4", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,188,212,0.08)",
    ))
    if "carry_5d" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["carry_5d"],
                                 name="Carry", line=dict(color="#4caf50", width=1, dash="dash"),
                                 opacity=0.75))
    if "roll_5d" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["roll_5d"],
                                 name="Roll", line=dict(color="#ff9800", width=1, dash="dot"),
                                 opacity=0.75))
    if "carry_roll_5d_z60" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["carry_roll_5d_z60"],
                                 name="C+R z60 (rhs)", line=dict(color="#9c27b0", width=1.5),
                                 yaxis="y2", opacity=0.85))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        **_dark_layout("Carry & Roll — 2s10s Steepener  (bps / 5d)  ·  2Y history"),
        yaxis=dict(title="bps / 5d"),
        yaxis2=dict(title="z60", overlaying="y", side="right",
                    zeroline=True, zerolinecolor="gray", zerolinewidth=0.5),
    )
    return fig


def _recent_signals_table(nop: pd.DataFrame) -> dash_table.DataTable:
    """Last 30 non-overlapping directional signals with colour-coded outcomes."""
    d = nop[nop["direction"] != "flat"].tail(30).copy().reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")

    show = ["date", "score", "direction", "thresh_steep", "thresh_flat",
            "fwd_bps", "carry_roll_bps", "aligned_bps", "aligned_adj_bps", "hit_adj"]
    show = [c for c in show if c in d.columns]
    d = d[show]
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].round(2)

    cond = []
    for col in ["aligned_bps", "aligned_adj_bps", "carry_roll_bps", "fwd_bps"]:
        if col not in show:
            continue
        cond += [
            {"if": {"filter_query": f"{{{col}}} > 0", "column_id": col},
             "backgroundColor": "#1a3a1a", "color": "#66ff66"},
            {"if": {"filter_query": f"{{{col}}} < 0", "column_id": col},
             "backgroundColor": "#3a1a1a", "color": "#ff6666"},
        ]
    if "hit_adj" in show:
        cond += [
            {"if": {"filter_query": "{hit_adj} = 1", "column_id": "hit_adj"},
             "backgroundColor": "#1a3a1a", "color": "#66ff66"},
            {"if": {"filter_query": "{hit_adj} = 0", "column_id": "hit_adj"},
             "backgroundColor": "#3a1a1a", "color": "#ff6666"},
        ]

    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[{"name": c, "id": c} for c in show],
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#222236", "color": "white",
            "fontSize": 12, "padding": "5px 8px", "border": "1px solid #333",
        },
        style_header={
            "backgroundColor": "#2a2a3e", "fontWeight": "bold",
            "border": "1px solid #444",
        },
        style_data_conditional=cond,
        sort_action="native",
    )


# ── Layout builder ────────────────────────────────────────────────────────────

def build_layout(df: pd.DataFrame, nop: pd.DataFrame) -> html.Div:
    last        = df.iloc[-1]
    regime      = last.get("regime", "Unknown")
    macro_reg   = last.get("macro_regime", "Neutral")
    macro_sc    = float(last.get("macro_scalar", 1.0))
    score       = float(last.get("composite_score", 0.0))
    gated       = float(last.get("gated_score",     0.0))
    latest_date = df.index[-1].strftime("%Y-%m-%d")

    # Sizing + thresholds
    sizing   = SignalEngine.position_size(df).iloc[-1]
    t_steep  = float(sizing.get("thresh_steep", 1.3))
    t_flat   = float(sizing.get("thresh_flat",  1.3))
    cr_z60   = last.get("carry_roll_5d_z60", 0.0)
    cr_z60   = float(cr_z60) if pd.notna(cr_z60) else 0.0

    direction_raw = str(sizing.get("direction", "flat"))
    if direction_raw == "steepener":
        direction_label = "STEEPENER ▲"
    elif direction_raw == "flattener":
        direction_label = "FLATTENER ▼"
    else:
        direction_label = "NEUTRAL ─"

    # Score components today
    regime_map = {
        "Bear Steepener": -1, "Bull Steepener": -1,
        "Bear Flattener": +1, "Bull Flattener": +1, "Unknown": 0,
    }
    voladj_val   = last.get("2s10s_voladj", np.nan)
    vix_z20_val  = last.get("VIX_z20",      np.nan)
    fe_z20_val   = last.get("1y2y_z20",     np.nan)

    c_regime   = float(regime_map.get(regime, 0))
    c_voladj   = float(np.clip((-voladj_val)   / 2, -1, 1)) if pd.notna(voladj_val) else 0.0
    c_vix      = float(np.clip(vix_z20_val      / 2, -1, 1)) if pd.notna(vix_z20_val) else 0.0
    c_frontend = float(np.clip(fe_z20_val        / 2, -1, 1)) if pd.notna(fe_z20_val) else 0.0

    # Carry
    carry_5d = last.get("carry_5d",       np.nan)
    roll_5d  = last.get("roll_5d",        np.nan)
    cr_5d    = last.get("carry_roll_5d",  np.nan)
    sofr_val = last.get("SOFR",           np.nan)

    # Sizing numbers
    score_frac = float(sizing.get("score_fraction",   0))
    vol_sc     = float(sizing.get("vol_scalar",       0))
    fwd_vol    = float(sizing.get("fwd_vol_bps",      0))
    risk_b     = float(sizing.get("risk_bps",         0))
    dv01_rec   = float(sizing.get("recommended_dv01", 0))

    header = dbc.Row([
        dbc.Col(html.H2("Rates Signal Dashboard", className="text-light mb-0"), width="auto"),
        dbc.Col(html.Small(f"Last update: {latest_date}", className="text-muted align-self-center"),
                width="auto"),
    ], className="my-3 align-items-center")

    # ── Row 1: Score card + 4 component mini-cards ────────────────────────────
    score_row = dbc.Row([
        dbc.Col(_score_card(score, gated, t_steep, t_flat, cr_z60, macro_reg, macro_sc), width=4),
        dbc.Col(_component_card("Regime (contrarian)", c_regime, regime), width=2),
        dbc.Col(_component_card("Vol-adj (contrarian)", c_voladj,
                                f"voladj: {voladj_val:+.2f}" if pd.notna(voladj_val) else "N/A"), width=2),
        dbc.Col(_component_card("VIX momentum", c_vix,
                                f"VIX z20: {vix_z20_val:+.2f}" if pd.notna(vix_z20_val) else "N/A"), width=2),
        dbc.Col(_component_card("Front-end 1y2y", c_frontend,
                                f"1y2y z20: {fe_z20_val:+.2f}" if pd.notna(fe_z20_val) else "N/A"), width=2),
    ], className="mb-3")

    # ── Row 2: Sizing + C+R + slope cards ─────────────────────────────────────
    context_row = dbc.Row([
        dbc.Col(_sizing_card(direction_label, score_frac, vol_sc, fwd_vol, risk_b, dv01_rec), width=4),
        dbc.Col(_carry_card(carry_5d, roll_5d, cr_5d, sofr_val, direction_label), width=4),
        dbc.Col(_slope_card("2s10s", last["2s10s"] * 100,
                            last["2s10s_z20"], last["2s10s_z60"], regime), width=2),
        dbc.Col(_slope_card("5s30s", last["5s30s"] * 100,
                            last["5s30s_z20"], last["5s30s_z60"], regime), width=2),
    ], className="mb-3")

    # ── Row 3: Yield cards ────────────────────────────────────────────────────
    yield_row = dbc.Row([
        dbc.Col(_yield_card(t, last[t], last[f"{t}_chg"]), width=3)
        for t in ["2Y", "5Y", "10Y", "30Y"]
    ], className="mb-3")

    # ── Signal-focused charts (ordered by relevance) ──────────────────────────
    charts = [
        dbc.Row(dbc.Col(dcc.Graph(figure=_score_history_chart(df))),  className="mb-3"),
        dbc.Row(dbc.Col(dcc.Graph(figure=_component_chart(df))),      className="mb-3"),
        dbc.Row(dbc.Col(dcc.Graph(figure=_pnl_chart(nop))),           className="mb-3"),
        dbc.Row(dbc.Col(dcc.Graph(figure=_zscore_chart(df))),         className="mb-3"),
        dbc.Row(dbc.Col(dcc.Graph(figure=_carry_chart(df))),          className="mb-3"),
    ]

    # ── Recent signals table ───────────────────────────────────────────────────
    recent_section = dbc.Row(
        dbc.Col([
            html.H5("Recent Directional Signals — Last 30 Non-Overlapping Trades",
                    className="text-muted mb-2"),
            _recent_signals_table(nop),
        ], width=12),
        className="mb-4",
    )

    return html.Div([
        dcc.Interval(id="interval", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dbc.Container([
            header,
            score_row,
            context_row,
            yield_row,
            *charts,
            recent_section,
        ], fluid=True),
    ])


# ── Initial layout ─────────────────────────────────────────────────────────────

_df, _nop = run_pipeline()
app.layout = build_layout(_df, _nop)


# ── Refresh callback ───────────────────────────────────────────────────────────

@app.callback(
    Output("interval", "disabled"),
    Input("interval", "n_intervals"),
)
def refresh(n: int):
    global app
    df, nop = run_pipeline()
    app.layout = build_layout(df, nop)
    return False


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
