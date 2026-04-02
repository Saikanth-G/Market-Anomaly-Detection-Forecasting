"""
Market Anomaly Detection Dashboard
===================================
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Anomaly Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")

COLORS = {
    "AAPL": "#1565C0", "AMZN": "#E65100", "GLD": "#2E7D32",
    "JPM":  "#C62828", "SPY":  "#6A1B9A", "XOM": "#4E342E",
}

EVENT_DATES = {
    "2020-03-16": "COVID Black Monday",
    "2020-03-20": "COVID Bottom",
    "2022-01-24": "Rate Hike Fear",
    "2022-09-13": "CPI Shock",
    "2023-03-10": "SVB Collapse",
}

RF_ANNUAL = 0.045
RF_DAILY  = RF_ANNUAL / 252

# ── Data loading (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    market   = pd.read_csv(DATA_DIR / "market_data.csv",            parse_dates=["Date"])
    anomaly  = pd.read_csv(DATA_DIR / "anomaly_results.csv",        parse_dates=["Date"])
    try:
        backtest = pd.read_csv(DATA_DIR / "backtest_trades.csv",    parse_dates=["Date"])
    except FileNotFoundError:
        backtest = pd.DataFrame()
    return market, anomaly, backtest

market_df, anomaly_df, backtest_df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Market Anomaly")
st.sidebar.markdown("**6-Year OHLCV Analysis**  \nAAPL · JPM · XOM · AMZN · SPY · GLD")
st.sidebar.divider()

all_tickers = sorted(market_df["Ticker"].unique())
selected_tickers = st.sidebar.multiselect(
    "Select Tickers", all_tickers, default=all_tickers
)

date_min = market_df["Date"].min().date()
date_max = market_df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

show_events = st.sidebar.checkbox("Show Market Events", value=True)
show_anomalies = st.sidebar.checkbox("Show Confirmed Anomalies", value=True)

st.sidebar.divider()
st.sidebar.caption(
    "**Stack:** Python · pandas · DuckDB SQL · "
    "Isolation Forest · Prophet · Streamlit"
)

# ── Filter data ────────────────────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = pd.Timestamp(date_min), pd.Timestamp(date_max)

mkt = market_df[
    market_df["Ticker"].isin(selected_tickers) &
    market_df["Date"].between(d0, d1)
].copy()

anom = anomaly_df[
    anomaly_df["Ticker"].isin(selected_tickers) &
    anomaly_df["Date"].between(d0, d1) &
    (anomaly_df["IsConfirmed"] == 1)
].copy()

# ── Header KPI strip ──────────────────────────────────────────────────────
st.title("Market Anomaly Detection Dashboard")
st.caption(f"Data: {d0.date()} → {d1.date()}  |  Tickers: {', '.join(selected_tickers)}")

con = duckdb.connect()
con.register("mkt", mkt)
con.register("anom_all", anomaly_df)

kpis = con.execute(f"""
    SELECT
        COUNT(DISTINCT Date)                                              AS trading_days,
        ROUND(AVG(DailyReturn)*252*100, 1)                                AS avg_ann_ret,
        SUM(CASE WHEN Ticker IN ({','.join([f"'{t}'" for t in selected_tickers])})
                  AND IsConfirmed = 1 THEN 1 ELSE 0 END)                  AS total_confirmed
    FROM anom_all
    JOIN (SELECT DISTINCT Date FROM mkt) d USING (Date)
""").df()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trading Days",        f"{len(mkt['Date'].unique()):,}")
col2.metric("Confirmed Anomalies", f"{len(anom):,}")
col3.metric("Tickers Analysed",    len(selected_tickers))
col4.metric("Date Span",
            f"{(d1 - d0).days // 365}y {((d1 - d0).days % 365) // 30}m")

st.divider()

# ── Tab layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price & Anomalies",
    "📉 Volatility",
    "⚠️ Risk Metrics",
    "🔍 SQL Explorer",
    "📈 Forecast",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Anomalies
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Normalised Price (Log Scale) + Confirmed Anomaly Markers")

    # Rebase prices
    first_close = (
        mkt.sort_values("Date").groupby("Ticker")["Close"].first().rename("first")
    )
    mkt2 = mkt.merge(first_close, on="Ticker")
    mkt2["rebased"] = mkt2["Close"] / mkt2["first"] * 100

    fig = go.Figure()
    for ticker in selected_tickers:
        sub = mkt2[mkt2["Ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=sub["Date"], y=sub["rebased"],
            mode="lines", name=ticker,
            line=dict(color=COLORS.get(ticker, "#999"), width=1.8),
        ))

    if show_anomalies:
        anom2 = anom.merge(mkt2[["Date","Ticker","rebased"]], on=["Date","Ticker"], how="left")
        fig.add_trace(go.Scatter(
            x=anom2["Date"], y=anom2["rebased"],
            mode="markers", name="Confirmed Anomaly",
            marker=dict(symbol="circle-open", size=10, color="red", line=dict(width=2)),
        ))

    if show_events:
        for date_str, label in EVENT_DATES.items():
            fig.add_vline(x=date_str, line_color="orange", line_dash="dot", line_width=1.2)
            fig.add_annotation(x=date_str, yref="paper", y=0.98,
                               text=label, showarrow=False,
                               xanchor="right", font=dict(color="darkorange", size=9))

    fig.update_yaxes(type="log", title="Indexed Price — log scale (100 = start)")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    if show_anomalies and len(anom) > 0:
        st.subheader("Confirmed Anomaly Events")
        disp = anom[["Date","Ticker","Close","DailyReturn","ZScore","^VIX"]].copy()
        disp["DailyReturn"] = (disp["DailyReturn"] * 100).round(2)
        disp["ZScore"]      = disp["ZScore"].round(2)
        disp["^VIX"]        = disp["^VIX"].round(1)
        disp = disp.rename(columns={"DailyReturn": "Return%", "^VIX": "VIX"})
        disp["Date"] = disp["Date"].dt.date
        st.dataframe(
            disp.sort_values("Date", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Volatility
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Rolling 30-Day Annualised Volatility")

    mkt_vol = mkt.sort_values(["Ticker","Date"]).copy()
    mkt_vol["vol30"] = (
        mkt_vol.groupby("Ticker")["DailyReturn"]
               .transform(lambda x: x.rolling(30).std() * np.sqrt(252) * 100)
    )

    fig2 = go.Figure()
    for ticker in selected_tickers:
        sub = mkt_vol[mkt_vol["Ticker"] == ticker].dropna(subset=["vol30"])
        fig2.add_trace(go.Scatter(
            x=sub["Date"], y=sub["vol30"],
            mode="lines", name=ticker,
            line=dict(color=COLORS.get(ticker, "#999"), width=1.6),
            fill="tozeroy", fillcolor=COLORS.get(ticker, "#999").replace("#","rgba(") + ",0.05)",
        ))

    if show_events:
        for date_str, label in EVENT_DATES.items():
            fig2.add_vline(x=date_str, line_color="red", line_dash="dash", line_width=1)

    fig2.update_layout(
        template="plotly_white", height=450,
        xaxis_title="Date", yaxis_title="Volatility (% annualised)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Daily Return Correlation")
    pivot = mkt.pivot(index="Date", columns="Ticker", values="DailyReturn").dropna()
    corr  = pivot.corr().round(2)
    fig3  = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1, aspect="auto",
        title="Daily Return Correlation (selected period)",
    )
    fig3.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Risk Metrics
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Risk Scorecard  (RF = {RF_ANNUAL*100:.1f}% p.a.)")

    con.register("mkt_risk", mkt)
    risk = con.execute(f"""
        WITH base AS (
            SELECT Ticker, DailyReturn,
                   DailyReturn - {RF_DAILY}            AS excess_ret,
                   LEAST(DailyReturn - {RF_DAILY}, 0)  AS downside_ret
            FROM mkt_risk WHERE DailyReturn IS NOT NULL
        ),
        var_tbl AS (
            SELECT Ticker,
                   PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY DailyReturn) AS var_95
            FROM base GROUP BY Ticker
        ),
        stats AS (
            SELECT b.Ticker, COUNT(*) AS n,
                   AVG(b.DailyReturn)     AS avg_ret,
                   STDDEV(b.DailyReturn)  AS std_ret,
                   STDDEV(b.downside_ret) AS dn_std,
                   AVG(b.excess_ret)      AS avg_exc,
                   v.var_95,
                   AVG(CASE WHEN b.DailyReturn <= v.var_95 THEN b.DailyReturn END) AS cvar_95
            FROM base b JOIN var_tbl v ON b.Ticker = v.Ticker
            GROUP BY b.Ticker, v.var_95
        ),
        dd AS (
            SELECT Ticker,
                   MIN((Close - MAX(Close) OVER (
                       PARTITION BY Ticker ORDER BY Date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   )) / MAX(Close) OVER (
                       PARTITION BY Ticker ORDER BY Date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   )) AS max_dd
            FROM mkt_risk GROUP BY Ticker
        )
        SELECT s.Ticker,
               ROUND(s.avg_ret*252*100, 2)                           AS ann_ret_pct,
               ROUND(s.std_ret*SQRT(252)*100, 2)                     AS ann_vol_pct,
               ROUND((s.avg_exc*252)/(s.std_ret*SQRT(252)), 3)       AS sharpe,
               ROUND((s.avg_exc*252)/(s.dn_std*SQRT(252)), 3)        AS sortino,
               ROUND(d.max_dd*100, 2)                                AS max_dd_pct,
               ROUND(s.var_95*100, 3)                                AS var_95_pct,
               ROUND(s.cvar_95*100, 3)                               AS cvar_95_pct
        FROM stats s JOIN dd d ON s.Ticker = d.Ticker
        ORDER BY sharpe DESC
    """).df()

    # Colour-coded dataframe
    def colour_sharpe(val):
        if val >= 1.0:   return "background-color: #C8E6C9; color: black"
        if val >= 0.5:   return "background-color: #FFF9C4; color: black"
        return "background-color: #FFCDD2; color: black"

    styled = (
        risk.style
            .applymap(colour_sharpe, subset=["sharpe"])
            .format({
                "ann_ret_pct": "{:.2f}%",
                "ann_vol_pct": "{:.2f}%",
                "sharpe":      "{:.3f}",
                "sortino":     "{:.3f}",
                "max_dd_pct":  "{:.2f}%",
                "var_95_pct":  "{:.3f}%",
                "cvar_95_pct": "{:.3f}%",
            })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig_s = px.bar(risk.sort_values("sharpe"), x="Ticker", y="sharpe",
                       color="sharpe", color_continuous_scale="RdYlGn",
                       title="Sharpe Ratio", template="plotly_white")
        fig_s.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_b:
        fig_dd = px.bar(risk.sort_values("max_dd_pct"), x="Ticker", y="max_dd_pct",
                        color="max_dd_pct", color_continuous_scale="RdYlGn_r",
                        title="Max Drawdown (%)", template="plotly_white")
        fig_dd.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — SQL Explorer
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Live DuckDB SQL Explorer")
    st.caption(
        "Query `market` (OHLCV + returns), `anomalies` (with ISO/Z flags), "
        "or `backtest` (confirmed trades). DuckDB runs in-memory on the loaded DataFrames."
    )

    con.register("market",    market_df)
    con.register("anomalies", anomaly_df)
    if len(backtest_df) > 0:
        con.register("backtest", backtest_df)

    example_queries = {
        "Annual returns per ticker": """
SELECT
    Ticker,
    YEAR(CAST(Date AS DATE))                          AS yr,
    ROUND(SUM(DailyReturn)*100, 2)                    AS total_ret_pct,
    ROUND(STDDEV(DailyReturn)*SQRT(252)*100, 2)       AS ann_vol_pct
FROM market
WHERE DailyReturn IS NOT NULL
GROUP BY Ticker, yr
ORDER BY Ticker, yr""",
        "Anomaly rate by VIX regime": """
SELECT
    CASE WHEN \"^VIX\" < 15 THEN 'Low (<15)'
         WHEN \"^VIX\" < 25 THEN 'Medium (15-25)'
         WHEN \"^VIX\" < 40 THEN 'High (25-40)'
         ELSE 'Extreme (>40)' END        AS vix_regime,
    COUNT(*)                             AS total_days,
    SUM(IsConfirmed)                     AS confirmed_anomalies,
    ROUND(SUM(IsConfirmed)*100.0/COUNT(*),2) AS anomaly_rate_pct
FROM anomalies
WHERE \"^VIX\" IS NOT NULL
GROUP BY vix_regime
ORDER BY vix_regime""",
        "Top 10 worst days any ticker": """
SELECT Date, Ticker,
       ROUND(DailyReturn*100, 2) AS ret_pct,
       ROUND(\"^VIX\", 1)          AS VIX
FROM market
WHERE DailyReturn IS NOT NULL
ORDER BY DailyReturn ASC
LIMIT 10""",
        "Confirmed anomalies by year": """
SELECT
    YEAR(CAST(Date AS DATE)) AS yr,
    Ticker,
    COUNT(*)                 AS confirmed_anomalies
FROM anomalies
WHERE IsConfirmed = 1
GROUP BY yr, Ticker
ORDER BY yr, Ticker""",
    }

    chosen = st.selectbox("Load example query", ["(custom)"] + list(example_queries.keys()))
    default_sql = example_queries.get(chosen, "SELECT * FROM market LIMIT 10")

    user_sql = st.text_area("SQL Query", value=default_sql, height=160,
                            help="Tables available: market, anomalies, backtest")

    if st.button("▶  Run Query", type="primary"):
        try:
            result = con.execute(user_sql).df()
            st.success(f"Returned {len(result):,} rows × {len(result.columns)} columns")
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Forecast
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    forecast_img = Path("outputs/spy_forecast.png")
    if forecast_img.exists():
        st.subheader("SPY 90-Day Prophet Forecast")
        st.image(str(forecast_img), use_column_width=True)
        st.caption(
            "Model: Prophet(changepoint_prior_scale=0.05, yearly_seasonality=True, "
            "weekly_seasonality=True, interval_width=0.95). "
            "Evaluated on last 6 months of actual data."
        )
    else:
        st.info("Run 05_forecast.ipynb to generate the forecast chart.")

    risk_img = Path("outputs/risk_scorecard.png")
    if risk_img.exists():
        st.subheader("Risk Scorecard Heatmap")
        st.image(str(risk_img), use_column_width=True)

    backtest_img = Path("outputs/backtest_equity_curve.png")
    if backtest_img.exists():
        st.subheader("Anomaly Strategy vs Buy-and-Hold")
        st.image(str(backtest_img), use_column_width=True)
