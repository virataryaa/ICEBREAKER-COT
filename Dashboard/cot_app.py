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
RED   = "#8b1a00"
GREEN = "#1a7a1a"
DRED  = "#c0392b"
AMBER = "#e8a020"
GRAY  = "#6e6e73"

COMM_COLORS = {
    "KC":  "#0a2463",
    "CC":  "#e8a020",
    "SB":  "#1a7a1a",
    "CT":  "#7b2d8b",
    "RC":  "#8b1a00",
    "LCC": "#4a7fb5",
}
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
    font=dict(family="-apple-system,Helvetica Neue,sans-serif", color="#1d1d1f", size=11),
)

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
def spec_col(is_cit: bool) -> str:
    return "Spec Net (Idx inc.)" if is_cit else "MM Net"


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


def build_zscore_matrix(df_cit: pd.DataFrame, df_disagg: pd.DataFrame) -> pd.DataFrame:
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
        sc  = spec_col(is_cit)
        row = {"Commodity": comm}
        for col_name, col in [("Spec Δ", sc), ("Comm Δ" if is_cit else "Swap Δ",
                                               "Comm Net" if is_cit else "Swap Net"),
                               ("Px Δ", "Px")]:
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
    color = COMM_COLORS.get(comm, NAVY)
    st.markdown(
        f"<h4 style='font-size:1rem;font-weight:700;color:{color};"
        f"margin:18px 0 6px;padding:5px 14px;"
        f"border-left:4px solid {color};background:#f7f7fa;border-radius:0 5px 5px 0'>"
        f"{COMM_NAMES.get(comm, comm)}</h4>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list, comm: str):
    color = COMM_COLORS.get(comm, NAVY)
    html = "<div style='display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px'>"
    for lbl, val, sub in items:
        sub_color = GREEN if sub and sub.startswith("▲") else DRED if sub and sub.startswith("▼") else "#888"
        sub_h = (f"<span style='font-size:.65rem;color:{sub_color};margin-left:5px'>{sub}</span>"
                 if sub else "")
        html += (
            f"<div style='background:#f0f2f8;border-radius:8px;padding:6px 13px;"
            f"min-width:100px;display:flex;flex-direction:column'>"
            f"<span style='font-size:.57rem;color:{GRAY};text-transform:uppercase;"
            f"letter-spacing:.09em'>{lbl}</span>"
            f"<span style='font-size:.9rem;font-weight:700;color:{color}'>{val}{sub_h}</span>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def weekly_change_bars(df: pd.DataFrame, comm: str, is_cit: bool, spec: bool) -> go.Figure:
    sc = spec_col(is_cit)
    d  = df[df["Commodity"] == comm].sort_values("Date").tail(9)
    if len(d) < 2:
        return go.Figure().update_layout(**_BASE, height=340)

    if spec:
        lc   = "Spec Long"  if is_cit else "MM Long"
        shc  = "Spec Short" if is_cit else "MM Short"
        nc   = sc
        title = f"{comm} — {'Spec' if is_cit else 'MM'} Weekly Δ (k lots)"
    else:
        lc   = "Comm Long"  if is_cit else "Swap Long"
        shc  = "Comm Short" if is_cit else "Swap Short"
        nc   = "Comm Net"   if is_cit else "Swap Net"
        title = f"{comm} — {'Comm' if is_cit else 'Swap'} Weekly Δ (k lots)"

    dates = d["Date"].iloc[1:]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=d[lc].diff().iloc[1:].values,  name="Long Δ",  marker_color="#2ecc71"))
    fig.add_trace(go.Bar(x=dates, y=d[shc].diff().iloc[1:].values, name="Short Δ", marker_color=DRED))
    fig.add_trace(go.Bar(x=dates, y=d[nc].diff().iloc[1:].values,  name="Net Δ",   marker_color=NAVY))
    fig.add_hline(y=0, line_width=0.8, line_color="#aaa")
    fig.update_layout(
        **_BASE, barmode="group", height=340,
        title=dict(text=title, font_size=12, x=0),
        margin=dict(l=50, r=20, t=45, b=90),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font_size=10),
        xaxis=dict(tickformat="%d %b '%y", tickangle=-30),
    )
    return fig


def gross_net_lines(df: pd.DataFrame, comm: str, is_cit: bool, spec: bool) -> go.Figure:
    sc = spec_col(is_cit)
    d  = df[df["Commodity"] == comm].sort_values("Date")
    if d.empty:
        return go.Figure().update_layout(**_BASE, height=340)

    if spec:
        lc   = "Spec Long"  if is_cit else "MM Long"
        shc  = "Spec Short" if is_cit else "MM Short"
        nc   = sc
        title = f"{comm} — {'Spec' if is_cit else 'MM'} Gross & Net (k lots)"
    else:
        lc   = "Comm Long"  if is_cit else "Swap Long"
        shc  = "Comm Short" if is_cit else "Swap Short"
        nc   = "Comm Net"   if is_cit else "Swap Net"
        title = f"{comm} — {'Comm' if is_cit else 'Swap'} Gross & Net (k lots)"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d["Date"], y=d[lc],  name="Long",
                             line=dict(color=GREEN, width=1.6)), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["Date"], y=d[shc], name="Short",
                             line=dict(color=DRED, width=1.6)), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["Date"], y=d[nc],  name="Net",
                             line=dict(color=NAVY, width=2, dash="dash")), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Px"], name="Px",
                             line=dict(color=AMBER, width=1.2), opacity=0.65), secondary_y=True)
    fig.update_layout(
        **_BASE, title=dict(text=title, font_size=12, x=0), height=340,
        margin=dict(l=50, r=50, t=45, b=90),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font_size=10),
        xaxis=dict(tickformat="%b '%y"),
    )
    fig.update_yaxes(title_text="k lots", secondary_y=False, title_font_size=10)
    fig.update_yaxes(title_text="Price",  secondary_y=True,  title_font_size=10, showgrid=False)
    return fig


def _scatter_base(x, y, dates, color, title, xlabel, ylabel, height=360) -> go.Figure:
    x     = np.asarray(x, dtype=float)
    y     = np.asarray(y, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    mask  = ~(np.isnan(x) | np.isnan(y))
    x, y, dates = x[mask], y[mask], dates[mask]
    if len(x) < 5:
        return go.Figure().update_layout(**_BASE,
               title=dict(text=title + "  [insufficient data]", font_size=12), height=height)

    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    sl, ic = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 200)
    recency = (dates - dates.min()).astype("timedelta64[D]").astype(float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=recency, colorscale="Blues", size=6, opacity=0.75,
                    colorbar=dict(title="Recent →", thickness=8, len=0.65,
                                  tickvals=[], ticktext=[])),
        text=pd.to_datetime(dates).strftime("%Y-%m-%d"),
        hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(x=xl, y=sl * xl + ic, mode="lines",
                             line=dict(color=color, width=1.8, dash="dash"), showlegend=False))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]], mode="markers", name="Latest",
        marker=dict(symbol="star", size=15, color=DRED, line=dict(width=1, color="white")),
        hovertemplate=f"{pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')}<br>X: {x[-1]:.2f}<br>Y: {y[-1]:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE,
        title=dict(text=f"{title}   R²={r2:.2f}", font_size=12, x=0),
        height=height,
        margin=dict(l=55, r=20, t=50, b=50),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(font_size=9, x=0.01, y=0.99),
    )
    return fig


def px_chg_vs_cot_scatter(df: pd.DataFrame, comm: str, x_col: str) -> go.Figure:
    d = df[df["Commodity"] == comm].sort_values("Date")
    px_chg_pct = d["Px"].pct_change() * 100
    cot_chg    = d[x_col].diff()
    color      = COMM_COLORS.get(comm, NAVY)
    return _scatter_base(
        x=px_chg_pct.values, y=cot_chg.values, dates=d["Date"].values, color=color,
        title=f"{comm} — {x_col} Δ vs Px Chg %",
        xlabel="Px weekly chg %", ylabel=f"{x_col} weekly Δ (k lots)",
    )


def position_vs_price_scatter(df: pd.DataFrame, comm: str, y_col: str) -> go.Figure:
    d = df[df["Commodity"] == comm].sort_values("Date").dropna(subset=[y_col, "Px"])
    color = COMM_COLORS.get(comm, NAVY)
    return _scatter_base(
        x=d["Px"].values, y=d[y_col].values, dates=d["Date"].values, color=color,
        title=f"{comm} — {y_col} vs Price",
        xlabel="Price", ylabel=f"{y_col} (k lots)",
    )


def histogram_trio(df: pd.DataFrame, comm: str, is_cit: bool) -> go.Figure:
    sc  = spec_col(is_cit)
    d   = df[df["Commodity"] == comm].sort_values("Date")
    primary_net = "Comm Net" if is_cit else "Swap Net"
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[f"{sc} Δ", f"{primary_net} Δ", "Px Δ"],
                        horizontal_spacing=0.07)
    for i, (col, color) in enumerate([(sc, GREEN), (primary_net, RED), ("Px", AMBER)], 1):
        if col not in d.columns:
            continue
        chg = d[col].diff().dropna()
        if chg.empty:
            continue
        lv = float(chg.iloc[-1])
        fig.add_trace(go.Histogram(x=chg, nbinsx=28, marker_color=color,
                                   opacity=0.75, showlegend=False), row=1, col=i)
        fig.add_vline(x=lv, line_dash="dash", line_color=DRED,
                      annotation_text=f" {lv:+.1f}", annotation_font_size=9,
                      row=1, col=i)
    fig.update_layout(
        **_BASE,
        title=dict(text=f"{comm} — Weekly Change Distributions", font_size=12, x=0),
        height=300, margin=dict(l=40, r=20, t=50, b=40), showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PER-COMMODITY BLOCK
# ══════════════════════════════════════════════════════════════════════════════
def render_commodity(df: pd.DataFrame, comm: str, is_cit: bool):
    d = df[df["Commodity"] == comm].sort_values("Date")
    if d.empty:
        st.info(f"No data for {comm} in selected date range.")
        return

    comm_header(comm)
    sc     = spec_col(is_cit)
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
            ("Swap Net",   _fmt("Swap Net"),   _chg("Swap Net")),
            ("Others Net", _fmt("Others Net"), _chg("Others Net")),
            ("Price",      f"{latest['Px']:.2f}" if pd.notna(latest["Px"]) else "—", _px_chg()),
            ("Spec %",     f"{latest['Spec Participation']*100:.1f}%" if pd.notna(latest["Spec Participation"]) else "—", ""),
        ]
    kpi_row(kpi_items, comm)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(weekly_change_bars(df, comm, is_cit, spec=True),  use_container_width=True)
    with c2:
        st.plotly_chart(weekly_change_bars(df, comm, is_cit, spec=False), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(gross_net_lines(df, comm, is_cit, spec=True),  use_container_width=True)
    with c2:
        st.plotly_chart(gross_net_lines(df, comm, is_cit, spec=False), use_container_width=True)

    if is_cit:
        cot_opts = [sc, "Comm Net", "Index Net", "Non Rep Net",
                    "Spec Long", "Spec Short", "Comm Long", "Comm Short",
                    "Spec Participation", "Comm Participation"]
    else:
        cot_opts = ["MM Net", "Swap Net", "Others Net", "Non Rep Net", "Spec Net",
                    "MM Long", "MM Short", "Swap Long", "Swap Short", "Spec Participation"]

    with st.expander(f"{comm} — Scatter: Price Chg % vs COT Element Δ", expanded=False):
        sel_x = st.selectbox("COT element (X-axis)", cot_opts, key=f"px_cot_{comm}")
        st.plotly_chart(px_chg_vs_cot_scatter(df, comm, sel_x), use_container_width=True)

    if is_cit:
        pos_opts = [sc, "Comm Net", "Spec Long", "Comm Long",
                    "Spec Short", "Comm Short", "Index Net", "Non Rep Net"]
    else:
        pos_opts = ["MM Net", "Swap Net", "Others Net", "Spec Net",
                    "MM Long", "MM Short", "Swap Long", "Swap Short", "Non Rep Net"]

    with st.expander(f"{comm} — Scatter: Net / Gross Long vs Price", expanded=False):
        sel_y = st.selectbox("Position (Y-axis)", pos_opts, key=f"pos_px_{comm}")
        st.plotly_chart(position_vs_price_scatter(df, comm, sel_y), use_container_width=True)

    st.plotly_chart(histogram_trio(df, comm, is_cit), use_container_width=True)
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
        comm_labels = {k: v for k, v in COMM_NAMES.items()}
        comm = st.selectbox(
            "Commodity",
            ALL_COMMS,
            format_func=lambda c: comm_labels.get(c, c),
        )
        is_cit = comm in CIT_COMMS

        st.markdown("---")

        date_range = st.slider(
            "Date range",
            min_value=min_d, max_value=max_d,
            value=(def_start, max_d),
            format="MMM YYYY",
        )

        st.markdown("---")

        with st.expander("Z-Score Matrix", expanded=True):
            d_start_z  = pd.Timestamp(date_range[0])
            d_end_z    = pd.Timestamp(date_range[1])
            cit_z      = df_cit[(df_cit["Date"] >= d_start_z) & (df_cit["Date"] <= d_end_z)]
            disagg_z   = df_disagg[(df_disagg["Date"] >= d_start_z) & (df_disagg["Date"] <= d_end_z)]
            zdf = build_zscore_matrix(cit_z, disagg_z)
            if not zdf.empty:
                z_cols = list(zdf.columns)
                styled = (
                    zdf.style
                       .map(_color_z, subset=z_cols)
                       .format("{:.2f}", na_rep="—", subset=z_cols)
                )
                st.dataframe(styled, use_container_width=True)
            else:
                st.info("Not enough data.")

    # ── Main area ─────────────────────────────────────────────────────────────
    d_start = pd.Timestamp(date_range[0])
    d_end   = pd.Timestamp(date_range[1])

    df      = df_cit    if is_cit else df_disagg
    df_f    = df[(df["Date"] >= d_start) & (df["Date"] <= d_end)]

    report  = "CIT" if is_cit else "Disaggregated"
    st.markdown(
        f"<h2 style='font-size:1.4rem;font-weight:700;color:{NAVY};margin-bottom:0'>"
        f"COT Dashboard</h2>"
        f"<p style='font-size:.76rem;color:{GRAY};margin-top:2px'>"
        f"{report} · {date_range[0]:%d %b %Y} – {date_range[1]:%d %b %Y} · Positions in k lots</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    render_commodity(df_f, comm, is_cit=is_cit)


if __name__ == "__main__":
    main()
