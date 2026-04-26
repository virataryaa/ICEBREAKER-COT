"""
COT Dashboard — ICEBREAKER
===========================
CIT (KC/CC/SB/CT) + Disaggregated (RC/LCC)
Run: streamlit run cot_app.py
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR    = Path(__file__).resolve().parent
DB_DIR      = CODE_DIR.parent / "Database"
ROLLEX_DIR  = DB_DIR / "Rollex"

CIT_FILE    = DB_DIR / "cot_cit.parquet"
DISAGG_FILE = DB_DIR / "cot_disagg.parquet"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="COT Dashboard", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
  [data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#fafafa!important}
  [data-testid="stHeader"]{background:transparent!important}
  .block-container{padding-top:1.5rem!important;padding-bottom:2rem;max-width:1600px}
  hr{border:none!important;border-top:1px solid #e8e8ed!important;margin:.6rem 0!important}
  div[data-testid="stExpander"]{border:1px solid #e0e0e8!important;border-radius:6px!important}
</style>""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY  = "#0a2463"
RED   = "#c0392b"
GREEN = "#27ae60"
DRED  = "#e74c3c"
AMBER = "#f39c12"
GRAY  = "#6e6e73"

# Chart-specific colours (softer, modern)
C_LONG  = "#27ae60"
C_SHORT = "#e74c3c"
C_PRICE = "#f39c12"

COMM_COLORS = {
    "KC":  "#1a56db",
    "CC":  "#d97706",
    "SB":  "#059669",
    "CT":  "#7c3aed",
    "RC":  "#dc2626",
    "LCC": "#0891b2",
}
ACCENT = COMM_COLORS["KC"]  # fixed theme colour regardless of selected commodity
COMM_NAMES = {
    "KC":  "KC — Arabica",
    "CC":  "CC — NYC Cocoa",
    "SB":  "SB — Sugar #11",
    "CT":  "CT — Cotton",
    "RC":  "RC — Robusta",
    "LCC": "LCC — LDN Cocoa",
}

_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,BlinkMacSystemFont,Helvetica Neue,sans-serif",
              color="#2d2d2d", size=11),
)

def _ax(x=False):
    base = dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=1,
        zeroline=True, zerolinecolor="rgba(0,0,0,0.14)", zerolinewidth=1,
        showline=True, linecolor="rgba(0,0,0,0.10)", linewidth=1,
        tickfont=dict(size=10, color="#888"),
        tickcolor="rgba(0,0,0,0)",
    )
    if x:
        base.update(showgrid=False, tickangle=-25)
    return base

CIT_COMMS    = ["KC", "CC", "SB", "CT"]
DISAGG_COMMS = ["RC", "LCC"]

CIT_POS_COLS = [
    "Comm Long", "Comm Short",
    "Spec Long", "Spec Short", "Spec Spread",
    "Index Long", "Index Short",
    "Non Rep Long", "Non Rep Short",
    "Total OI",
]
DISAGG_POS_COLS = [
    "Comm Long", "Comm Short",
    "Swap Long", "Swap Short", "Swap Spread",
    "MM Long", "MM Short", "MM Spread",
    "Other Long", "Other Short",
    "Non Rep Long", "Non Rep Short",
    "Total OI",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def load_cit() -> pd.DataFrame:
    df = pd.read_parquet(CIT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    for c in CIT_POS_COLS:
        df[c] = df[c] / 1000.0
    df["Comm Net"]            = df["Comm Long"]  - df["Comm Short"]
    df["Spec Net"]            = (df["Spec Long"] - df["Spec Short"]
                                 + df["Non Rep Long"] - df["Non Rep Short"])
    df["Spec Net (Idx inc.)"] = df["Spec Net"] + df["Index Long"] - df["Index Short"]
    df["Index Net"]           = df["Index Long"]  - df["Index Short"]
    df["Non Rep Net"]         = df["Non Rep Long"] - df["Non Rep Short"]
    df["Comm Participation"]  = (df["Comm Long"] + df["Comm Short"]) / df["Total OI"]
    df["Spec Participation"]  = (df["Spec Long"] + df["Spec Short"]
                                 + df["Non Rep Long"] + df["Non Rep Short"]) / df["Total OI"]
    return df.sort_values(["Commodity", "Date"]).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_disagg() -> pd.DataFrame:
    df = pd.read_parquet(DISAGG_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    for c in DISAGG_POS_COLS:
        if c in df.columns:
            df[c] = df[c] / 1000.0
    df["Comm Net"]           = df["Comm Long"]  - df["Comm Short"]
    df["Swap Net"]           = df["Swap Long"]  - df["Swap Short"]
    df["MM Net"]             = df["MM Long"]    - df["MM Short"]
    df["Others Net"]         = df["Other Long"] - df["Other Short"]
    df["Non Rep Net"]        = df["Non Rep Long"] - df["Non Rep Short"]
    # Broad spec = MM + Other + Non Rep
    df["Spec Net"]           = df["MM Net"] + df["Others Net"] + df["Non Rep Net"]
    df["Spec Participation"] = (
        df["MM Long"] + df["MM Short"] +
        df["Other Long"] + df["Other Short"] +
        df["Non Rep Long"] + df["Non Rep Short"]
    ) / df["Total OI"]
    df["Comm Participation"] = (df["Comm Long"] + df["Comm Short"]) / df["Total OI"]
    return df.sort_values(["Commodity", "Date"]).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_rollex(comm: str):
    path = ROLLEX_DIR / f"rollex_{comm}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index.name = "Date"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def spec_col(is_cit: bool, include_idx: bool = True) -> str:
    if is_cit:
        return "Spec Net (Idx inc.)" if include_idx else "Spec Net"
    return "MM Net"


def _align_to_cot(cot_dates: pd.Series, ext_df: pd.DataFrame,
                  date_col: str, value_col: str) -> np.ndarray:
    cot = pd.DataFrame({"Date": pd.to_datetime(cot_dates)}).sort_values("Date")
    ext = ext_df[[date_col, value_col]].copy()
    ext[date_col] = pd.to_datetime(ext[date_col])
    ext = ext.dropna().sort_values(date_col)
    merged = pd.merge_asof(cot, ext, left_on="Date", right_on=date_col, direction="nearest")
    return merged[value_col].values


# ══════════════════════════════════════════════════════════════════════════════
# Z-SCORE
# ══════════════════════════════════════════════════════════════════════════════
def _zscore(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if len(s) < 5:
        return np.nan
    mu, sd = float(s.mean()), float(s.std())
    return float((value - mu) / sd) if sd else np.nan


def build_zscore_matrix(df_cit: pd.DataFrame, df_disagg: pd.DataFrame,
                        include_idx: bool = True) -> pd.DataFrame:
    rows = []
    specs = [
        ("KC",  df_cit,    True),
        ("CC",  df_cit,    True),
        ("SB",  df_cit,    True),
        ("CT",  df_cit,    True),
        ("RC",  df_disagg, False),
        ("LCC", df_disagg, False),
    ]
    for comm, df, is_cit in specs:
        d = df[df["Commodity"] == comm].sort_values("Date")
        if len(d) < 10:
            continue
        sc  = spec_col(is_cit, include_idx)
        row = {"Commodity": comm}
        col_map = [
            ("Spec Δ",       sc),
            ("Spec Long Δ",  "Spec Long"  if is_cit else "MM Long"),
            ("Spec Short Δ", "Spec Short" if is_cit else "MM Short"),
            ("Comm Δ",       "Comm Net"),
            ("Comm Long Δ",  "Comm Long"),
            ("Comm Short Δ", "Comm Short"),
        ]
        for col_name, col in col_map:
            if col not in d.columns:
                continue
            chg = d[col].diff()
            if chg.dropna().empty:
                continue
            row[col_name] = round(_zscore(chg, float(chg.iloc[-1])), 2)

        rl = load_rollex(comm)
        if rl is not None:
            rl_reset = rl[["rollex_px"]].reset_index()
            rl_reset.columns = ["Date", "Rollex"]
            aligned = _align_to_cot(d["Date"], rl_reset, "Date", "Rollex")
            rl_chg = pd.Series(aligned).diff()
            if rl_chg.dropna().shape[0] > 5:
                row["Rollex Δ"] = round(_zscore(rl_chg, float(rl_chg.iloc[-1])), 2)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Commodity")


def _color_z(val):
    if pd.isna(val):
        return "background-color:#f5f5f5;color:#bbb"
    if val >= 2:
        return "background-color:#1a7a1a;color:white;font-weight:bold"
    if val >= 1:
        return "background-color:#a8d5a2;color:#1a1a1a"
    if val <= -2:
        return "background-color:#c0392b;color:white;font-weight:bold"
    if val <= -1:
        return "background-color:#f5a9a2;color:#1a1a1a"
    return "background-color:#fff8e8;color:#1a1a1a"


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def comm_header(comm: str):
    color = ACCENT
    st.markdown(
        f"<h4 style='font-size:1rem;font-weight:700;color:{color};"
        f"margin:18px 0 6px;padding:5px 14px;"
        f"border-left:4px solid {color};background:#f7f7fa;border-radius:0 5px 5px 0'>"
        f"{COMM_NAMES.get(comm, comm)}</h4>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list, comm: str):
    color = ACCENT
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px'>"
    for lbl, val, sub in items:
        sub_color = "#16a34a" if sub and sub.startswith("▲") else "#dc2626" if sub and sub.startswith("▼") else "#888"
        sub_h = (f"<span style='font-size:.65rem;color:{sub_color};margin-left:4px'>{sub}</span>"
                 if sub else "")
        html += (
            f"<div style='background:rgba({r},{g},{b},0.06);border:1px solid rgba({r},{g},{b},0.15);"
            f"border-radius:10px;padding:7px 15px;min-width:105px;display:flex;flex-direction:column'>"
            f"<span style='font-size:.56rem;color:#999;text-transform:uppercase;"
            f"letter-spacing:.1em;margin-bottom:2px'>{lbl}</span>"
            f"<span style='font-size:.92rem;font-weight:700;color:{color}'>{val}{sub_h}</span>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def weekly_change_bars(df: pd.DataFrame, comm: str, is_cit: bool, spec: bool,
                       include_idx: bool = True) -> go.Figure:
    sc = spec_col(is_cit, include_idx)
    d  = df[df["Commodity"] == comm].sort_values("Date").tail(13)
    if len(d) < 2:
        return go.Figure().update_layout(**_BASE, height=340)

    color = ACCENT
    if spec:
        lc, shc, nc = (("Spec Long", "Spec Short", sc) if is_cit
                       else ("MM Long", "MM Short", sc))
        label = "Spec" if is_cit else "MM"
    else:
        lc, shc, nc = (("Comm Long", "Comm Short", "Comm Net") if is_cit
                       else ("Swap Long", "Swap Short", "Swap Net"))
        label = "Comm" if is_cit else "Swap"

    title  = f"{label} Weekly Change  ·  k lots"
    dates  = d["Date"].iloc[1:]
    ld, sd, nd = d[lc].diff().iloc[1:].values, d[shc].diff().iloc[1:].values, d[nc].diff().iloc[1:].values

    def _bar(y, name, clr, opacity=0.82):
        return go.Bar(
            x=dates, y=y, name=name,
            marker=dict(color=clr, opacity=opacity, line=dict(width=0)),
            hovertemplate=f"<b>%{{x|%d %b %y}}</b><br>{name}: %{{y:+.1f}}k<extra></extra>",
        )

    fig = go.Figure([
        _bar(ld, "Long Δ",  C_LONG),
        _bar(sd, "Short Δ", C_SHORT),
        _bar(nd, "Net Δ",   color, opacity=0.95),
    ])
    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.18)")
    fig.update_layout(
        **_BASE, barmode="group", height=340,
        title=dict(text=title, font=dict(size=12, color="#444"), x=0),
        margin=dict(l=50, r=16, t=40, b=80),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), tickformat="%d %b '%y"),
        yaxis=dict(**_ax(), title_text="k lots", title_font_size=10),
        bargap=0.18, bargroupgap=0.06,
    )
    return fig


def gross_net_lines(df: pd.DataFrame, comm: str, is_cit: bool, spec: bool,
                    include_idx: bool = True) -> go.Figure:
    sc    = spec_col(is_cit, include_idx)
    d     = df[df["Commodity"] == comm].sort_values("Date")
    color = ACCENT
    if d.empty:
        return go.Figure().update_layout(**_BASE, height=360)

    if spec:
        lc, shc, nc = (("Spec Long", "Spec Short", sc) if is_cit
                       else ("MM Long", "MM Short", sc))
        label = "Spec" if is_cit else "MM"
    else:
        lc, shc, nc = (("Comm Long", "Comm Short", "Comm Net") if is_cit
                       else ("Swap Long", "Swap Short", "Swap Net"))
        label = "Comm" if is_cit else "Swap"

    title = f"{label} Gross & Net  ·  k lots"
    fig   = make_subplots(specs=[[{"secondary_y": True}]])

    # filled net area
    fig.add_trace(go.Scatter(
        x=d["Date"], y=d[nc], name="Net",
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
        line=dict(color=color, width=2.2, shape="spline", smoothing=0.6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Net: %{y:.1f}k<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=d["Date"], y=d[lc], name="Long",
        line=dict(color=C_LONG, width=1.6, shape="spline", smoothing=0.6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Long: %{y:.1f}k<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=d["Date"], y=d[shc], name="Short",
        line=dict(color=C_SHORT, width=1.6, shape="spline", smoothing=0.6),
        hovertemplate="<b>%{x|%b %Y}</b><br>Short: %{y:.1f}k<extra></extra>",
    ), secondary_y=False)

    rl = load_rollex(comm)
    if rl is not None:
        rl_reset = rl[["rollex_px"]].reset_index()
        rl_reset.columns = ["Date", "Rollex"]
        rollex_vals = _align_to_cot(d["Date"], rl_reset, "Date", "Rollex")
        px_y, px_label = rollex_vals, "Rollex"
    else:
        px_y, px_label = d["Px"].values, "Price"

    fig.add_trace(go.Scatter(
        x=d["Date"], y=px_y, name=px_label,
        line=dict(color=C_PRICE, width=1.4, dash="dot"),
        opacity=0.7,
        hovertemplate="<b>%{x|%b %Y}</b><br>" + px_label + ": %{y:.2f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **_BASE, height=360,
        title=dict(text=title, font=dict(size=12, color="#444"), x=0),
        margin=dict(l=50, r=55, t=40, b=80),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(x=True), tickformat="%b '%y"),
        yaxis=dict(**_ax()),
    )
    rl_label = "Rollex" if (load_rollex(comm) is not None) else "Price"
    fig.update_yaxes(title_text="k lots",   title_font_size=10, secondary_y=False, **_ax())
    fig.update_yaxes(title_text=rl_label,   title_font_size=10, secondary_y=True,
                     showgrid=False, tickfont=dict(size=10, color="#888"))
    return fig


def _scatter_base(x, y, dates, color, title, xlabel, ylabel, height=380) -> go.Figure:
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    mask  = ~(np.isnan(x) | np.isnan(y))
    x, y, dates = x[mask], y[mask], dates[mask]
    if len(x) < 5:
        return go.Figure().update_layout(
            **_BASE, title=dict(text=title + "  [insufficient data]", font_size=12), height=height)

    r2       = float(np.corrcoef(x, y)[0, 1] ** 2)
    sl, ic   = np.polyfit(x, y, 1)
    xl       = np.linspace(x.min(), x.max(), 200)
    recency  = (dates - dates.min()).astype("timedelta64[D]").astype(float)
    norm_rec = recency / recency.max() if recency.max() > 0 else recency

    r, g, b  = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            color=norm_rec, colorscale=[[0, "rgba(200,210,230,0.5)"],
                                         [1, f"rgba({r},{g},{b},0.85)"]],
            size=7, line=dict(width=0.5, color="white"),
        ),
        text=pd.to_datetime(dates).strftime("%Y-%m-%d"),
        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xl, y=sl * xl + ic, mode="lines",
        line=dict(color=color, width=1.6, dash="dash"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]], mode="markers", showlegend=False,
        marker=dict(symbol="star", size=14, color=DRED, line=dict(width=1.2, color="white")),
        hovertemplate=f"<b>{pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}</b><br>X: {x[-1]:.2f}<br>Y: {y[-1]:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE,
        title=dict(text=f"{title}   <span style='font-size:11px;color:#888'>R²={r2:.2f}</span>",
                   font=dict(size=12, color="#444"), x=0),
        height=height,
        margin=dict(l=55, r=20, t=48, b=48),
        xaxis=dict(**_ax(x=True), title_text=xlabel, title_font_size=10),
        yaxis=dict(**_ax(), title_text=ylabel, title_font_size=10),
        legend=dict(font_size=9, x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def px_chg_vs_cot_scatter(df: pd.DataFrame, comm: str, x_col: str) -> go.Figure:
    d = df[df["Commodity"] == comm].sort_values("Date")
    return _scatter_base(
        x=d["Px"].pct_change().values * 100,
        y=d[x_col].diff().values,
        dates=d["Date"].values,
        color=ACCENT,
        title=f"{x_col} Δ vs Price Chg %",
        xlabel="Price weekly chg %", ylabel=f"{x_col} Δ (k lots)",
    )


def position_vs_price_scatter(df: pd.DataFrame, comm: str, y_col: str) -> go.Figure:
    d = df[df["Commodity"] == comm].sort_values("Date").dropna(subset=[y_col, "Px"])
    return _scatter_base(
        x=d["Px"].values, y=d[y_col].values, dates=d["Date"].values,
        color=ACCENT,
        title=f"{y_col} vs Price",
        xlabel="Price", ylabel=f"{y_col} (k lots)",
    )


def histogram_trio(df: pd.DataFrame, comm: str, is_cit: bool,
                   include_idx: bool = True) -> go.Figure:
    sc          = spec_col(is_cit, include_idx)
    d           = df[df["Commodity"] == comm].sort_values("Date")
    primary_net = "Comm Net" if is_cit else "Swap Net"
    color       = ACCENT
    specs_list  = [(sc, color), (primary_net, "#64748b"), ("Px", C_PRICE)]
    labels      = [f"{sc} Δ", f"{primary_net} Δ", "Px Δ"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=labels, horizontal_spacing=0.08)
    for i, (col, clr) in enumerate(specs_list, 1):
        if col not in d.columns:
            continue
        chg = d[col].diff().dropna()
        if chg.empty:
            continue
        lv = float(chg.iloc[-1])
        r, g, b = int(clr[1:3], 16), int(clr[3:5], 16), int(clr[5:7], 16)
        fig.add_trace(go.Histogram(
            x=chg, nbinsx=30, showlegend=False,
            marker=dict(color=f"rgba({r},{g},{b},0.72)",
                        line=dict(color="white", width=0.6)),
        ), row=1, col=i)
        fig.add_vline(x=lv, line_dash="dash", line_color=DRED, line_width=1.6,
                      annotation_text=f" {lv:+.1f}", annotation_font_size=9,
                      annotation_font_color=DRED, row=1, col=i)

    fig.update_layout(
        **_BASE,
        title=dict(text="Weekly Change Distributions", font=dict(size=12, color="#444"), x=0),
        height=300, margin=dict(l=40, r=20, t=48, b=36), showlegend=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(**_ax(x=True), row=1, col=i)
        fig.update_yaxes(**_ax(), row=1, col=i)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PER-COMMODITY BLOCK
# ══════════════════════════════════════════════════════════════════════════════
def render_commodity(df: pd.DataFrame, comm: str, is_cit: bool, include_idx: bool = True):
    d = df[df["Commodity"] == comm].sort_values("Date")
    if d.empty:
        st.info(f"No data for {comm} in selected date range.")
        return

    comm_header(comm)
    sc     = spec_col(is_cit, include_idx)
    latest = d.iloc[-1]
    prev   = d.iloc[-2] if len(d) > 1 else latest

    def _fmt(col):
        v = latest.get(col, np.nan)
        return "—" if pd.isna(v) else f"{v:.0f}k"

    def _chg(col):
        v, p = latest.get(col, np.nan), prev.get(col, np.nan)
        if pd.isna(v) or pd.isna(p):
            return ""
        c = v - p
        return f"{'▲' if c > 0 else '▼'}{abs(c):.1f}k"

    def _px_chg():
        v, p = latest.get("Px", np.nan), prev.get("Px", np.nan)
        if pd.isna(v) or pd.isna(p):
            return ""
        c = v - p
        pct = c / p * 100 if p else 0
        return f"{'▲' if c > 0 else '▼'}{abs(c):.1f} ({abs(pct):.1f}%)"

    if is_cit:
        kpi_items = [
            (sc,           _fmt(sc),          _chg(sc)),
            ("Comm Net",   _fmt("Comm Net"),   _chg("Comm Net")),
            ("Index Net",  _fmt("Index Net"),  _chg("Index Net")),
            ("Price",      f"{latest['Px']:.2f}" if pd.notna(latest["Px"]) else "—", _px_chg()),
            ("Spec %",     f"{latest['Spec Participation']*100:.1f}%" if pd.notna(latest["Spec Participation"]) else "—", ""),
            ("Comm %",     f"{latest['Comm Participation']*100:.1f}%" if pd.notna(latest["Comm Participation"]) else "—", ""),
        ]
    else:
        kpi_items = [
            ("MM Net",     _fmt("MM Net"),     _chg("MM Net")),
            ("Comm Net",   _fmt("Comm Net"),   _chg("Comm Net")),
            ("Swap Net",   _fmt("Swap Net"),   _chg("Swap Net")),
            ("Others Net", _fmt("Others Net"), _chg("Others Net")),
            ("Price",      f"{latest['Px']:.2f}" if pd.notna(latest["Px"]) else "—", _px_chg()),
            ("Spec %",     f"{latest['Spec Participation']*100:.1f}%" if pd.notna(latest["Spec Participation"]) else "—", ""),
            ("Comm %",     f"{latest['Comm Participation']*100:.1f}%" if pd.notna(latest["Comm Participation"]) else "—", ""),
        ]
    kpi_row(kpi_items, comm)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(weekly_change_bars(df, comm, is_cit, spec=True,  include_idx=include_idx), use_container_width=True)
    with c2:
        st.plotly_chart(weekly_change_bars(df, comm, is_cit, spec=False, include_idx=include_idx), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(gross_net_lines(df, comm, is_cit, spec=True,  include_idx=include_idx), use_container_width=True)
    with c2:
        st.plotly_chart(gross_net_lines(df, comm, is_cit, spec=False, include_idx=include_idx), use_container_width=True)

    if is_cit:
        cot_opts = [sc, "Comm Net", "Index Net", "Non Rep Net",
                    "Spec Long", "Spec Short", "Comm Long", "Comm Short",
                    "Spec Participation", "Comm Participation"]
    else:
        cot_opts = ["MM Net", "Comm Net", "Swap Net", "Others Net", "Non Rep Net", "Spec Net",
                    "MM Long", "MM Short", "Comm Long", "Comm Short", "Swap Long", "Swap Short", "Spec Participation"]

    with st.expander(f"{comm} — Scatter: Price Chg % vs COT Element Δ", expanded=False):
        sel_x = st.selectbox("COT element (X-axis)", cot_opts, key=f"px_cot_{comm}")
        st.plotly_chart(px_chg_vs_cot_scatter(df, comm, sel_x), use_container_width=True)

    if is_cit:
        pos_opts = [sc, "Comm Net", "Spec Long", "Comm Long",
                    "Spec Short", "Comm Short", "Index Net", "Non Rep Net"]
    else:
        pos_opts = ["MM Net", "Comm Net", "Swap Net", "Others Net", "Spec Net",
                    "MM Long", "MM Short", "Comm Long", "Comm Short", "Swap Long", "Swap Short", "Non Rep Net"]

    with st.expander(f"{comm} — Scatter: Net / Gross Long vs Price", expanded=False):
        sel_y = st.selectbox("Position (Y-axis)", pos_opts, key=f"pos_px_{comm}")
        st.plotly_chart(position_vs_price_scatter(df, comm, sel_y), use_container_width=True)

    st.plotly_chart(histogram_trio(df, comm, is_cit, include_idx=include_idx), use_container_width=True)
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df_cit    = load_cit()
    df_disagg = load_disagg()

    all_dates = pd.concat([df_cit["Date"], df_disagg["Date"]])
    min_d     = all_dates.min().date()
    max_d     = all_dates.max().date()
    def_start = datetime.date(max(min_d.year, max_d.year - 5), 1, 1)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<p style='font-size:.7rem;font-weight:700;color:{GRAY};"
            f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px'>"
            f"ICEBREAKER · COT</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        ALL_COMMS = CIT_COMMS + DISAGG_COMMS
        comm = st.selectbox(
            "Commodity",
            ALL_COMMS,
            format_func=lambda c: COMM_NAMES.get(c, c),
        )
        is_cit = comm in CIT_COMMS

        st.markdown("---")

        idx_choice  = st.radio("Spec Net", ["Include Index", "Exclude Index"], index=0)
        include_idx = idx_choice == "Include Index"

        st.markdown("---")

        date_range = st.slider(
            "Date range",
            min_value=min_d, max_value=max_d,
            value=(def_start, max_d),
            format="MMM YYYY",
        )

    # ── Main area ─────────────────────────────────────────────────────────────
    d_start = pd.Timestamp(date_range[0])
    d_end   = pd.Timestamp(date_range[1])

    cit_f    = df_cit[(df_cit["Date"]       >= d_start) & (df_cit["Date"]       <= d_end)]
    disagg_f = df_disagg[(df_disagg["Date"] >= d_start) & (df_disagg["Date"]    <= d_end)]
    df_f     = cit_f if is_cit else disagg_f

    report = "CIT" if is_cit else "Disaggregated"
    st.markdown(
        f"<h2 style='font-size:1.4rem;font-weight:700;color:{NAVY};margin-bottom:0'>"
        f"COT Dashboard</h2>"
        f"<p style='font-size:.76rem;color:{GRAY};margin-top:2px'>"
        f"{report} · {date_range[0]:%d %b %Y} – {date_range[1]:%d %b %Y} · Positions in k lots</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab_comm, tab_cross = st.tabs([COMM_NAMES.get(comm, comm), "Cross Commodity Analysis"])

    with tab_comm:
        render_commodity(df_f, comm, is_cit=is_cit, include_idx=include_idx)

    with tab_cross:
        idx_label = "Index included" if include_idx else "Index excluded"
        st.markdown(
            f"<p style='font-size:.8rem;color:{GRAY};margin-bottom:12px'>"
            f"Weekly Δ z-scores across all commodities · same date range · {idx_label}</p>",
            unsafe_allow_html=True,
        )
        zdf = build_zscore_matrix(cit_f, disagg_f, include_idx=include_idx)
        if not zdf.empty:
            z_cols = list(zdf.columns)
            styled = (
                zdf.style
                   .map(_color_z, subset=z_cols)
                   .format("{:.2f}", na_rep="—", subset=z_cols)
            )
            st.dataframe(styled, use_container_width=True, height=280)
        else:
            st.info("Not enough data in selected range.")


if __name__ == "__main__":
    main()
