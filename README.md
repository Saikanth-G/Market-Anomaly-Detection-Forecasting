# Market Anomaly Detection & Forecasting

End-to-end financial data analysis project covering 6 years of daily OHLCV data for 6 tickers (AAPL, JPM, XOM, AMZN, SPY, GLD). Detects market anomalies using Isolation Forest ML combined with rolling Z-score analysis, forecasts SPY 90 days forward with Prophet, and quantifies risk-adjusted performance for each asset.

---

## Live Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes:
- Interactive normalised price chart (log scale) with anomaly markers
- Per-ticker rolling volatility
- Live DuckDB SQL explorer — query the data directly in the browser
- Risk scorecard (Sharpe, Sortino, VaR, CVaR, Max Drawdown)
- Prophet forecast and backtest equity curve

---

## Project Structure

```
├── 01_ingestion.ipynb          # Download OHLCV via yfinance, compute returns, attach VIX
├── 02_eda.ipynb                # EDA: normalised prices, volatility, correlation, SQL aggregations
├── 03_indicators.ipynb         # SMA/EMA/RSI/Bollinger Bands/Z-score (pure pandas)
├── 04_anomaly.ipynb            # Isolation Forest + Z-score anomaly detection
├── 05_forecast.ipynb           # Prophet 90-day SPY forecast + evaluation
├── 06_risk_metrics.ipynb       # Sharpe, Sortino, Max Drawdown, VaR, CVaR
├── 07_backtest.ipynb           # Anomaly signal backtesting vs buy-and-hold
├── app.py                      # Streamlit interactive dashboard
│
├── data/
│   ├── market_data.csv                  # 12,426 rows × 10 cols (2018–2026)
│   ├── market_data_indicators.csv       # + SMA/EMA/RSI/BB/Z-score columns
│   ├── anomaly_results.csv              # + IsoScore / IsAnomaly_ISO / IsConfirmed
│   └── backtest_trades.csv             # Confirmed anomaly trade log
│
└── outputs/
    ├── normalised_price.png
    ├── rolling_volatility.png
    ├── correlation_heatmap.png
    ├── spy_monthly_heatmap.png
    ├── anomaly_{AAPL,AMZN,GLD,JPM,SPY,XOM}.png
    ├── spy_forecast.png
    ├── rolling_sharpe.png
    ├── drawdown_chart.png
    ├── risk_scorecard.png
    └── backtest_equity_curve.png
```

---

## Methodology

### 1. Data Ingestion (`01_ingestion.ipynb`)
- Downloads adjusted OHLCV via `yfinance` for AAPL, JPM, XOM, AMZN, SPY, GLD (Jan 2018 → present)
- Reshapes to long format (one row per date-ticker)
- Computes `DailyReturn` (pct_change), `LogReturn` (log(Pₜ/Pₜ₋₁)), `CumReturn` ((1+r).cumprod()-1)
- Merges VIX (CBOE fear index) on date
- DuckDB SQL: ticker summary table, top-10 worst days query

### 2. Exploratory Data Analysis (`02_eda.ipynb`)
- Rebases all tickers to 100 at Jan 2018, plotted on **log scale** (equal vertical distance = equal % move)
- Per-ticker rolling volatility as 2×3 subplot grid (each ticker has its own y-axis scale)
- Seaborn return correlation heatmap — GLD shows near-zero correlation with equities
- DuckDB SQL: year-over-year returns, best/worst days, monthly return heatmap, worst calendar months

### 3. Technical Indicators (`03_indicators.ipynb`)
All indicators computed from scratch in pandas (no external TA library):

| Indicator | Formula | Flag |
|---|---|---|
| SMA50 / SMA200 | `rolling(n).mean()` | Golden Cross / Death Cross |
| EMA20 | `ewm(span=20)` | — |
| RSI14 | Wilder smoothing | Overbought (>70) / Oversold (<30) |
| Bollinger Bands (20d, 2σ) | `mean ± 2×std` | Upper/Lower breach |
| Rolling Z-score | `(r − μ̄₃₀) / σ̄₃₀` | `\|z\| > 3` → statistical anomaly |

### 4. Anomaly Detection (`04_anomaly.ipynb`)
**Two independent methods → confirmed only when both agree:**

| Method | Features | Contamination |
|---|---|---|
| Isolation Forest | DailyReturn, LogReturn, VolChange, Vol5D, Vol30D, RSI14, VIX | 5% |
| Rolling Z-score | DailyReturn | \|z\| > 3 |

**Result: 74 confirmed anomalies across 6 tickers** (≈0.6% of all trading days)

| Ticker | ISO flags | Z flags | Confirmed |
|---|---|---|---|
| AAPL | ~100 | ~35 | 14 |
| AMZN | ~100 | ~45 | 16 |
| GLD | ~100 | ~25 | 11 |
| JPM | ~100 | ~30 | 13 |
| SPY | ~100 | ~30 | 14 |
| XOM | ~100 | ~20 | 6 |

DuckDB SQL analysis confirms anomaly rate rises sharply with VIX regime: Low VIX (<15) → <0.3% anomaly rate; Extreme VIX (>40) → >4% anomaly rate.

### 5. SPY Forecast (`05_forecast.ipynb`)
- Prophet model with `changepoint_prior_scale=0.05`, yearly + weekly seasonality, 95% confidence interval
- Train/test split: last 6 months held out for evaluation
- **Test MAE = $26.30 | Test MAPE = 3.88%** (excellent for equity forecasting)
- Forecast horizon: 90 business days forward

### 6. Risk Metrics (`06_risk_metrics.ipynb`)
All computed via DuckDB SQL (window functions + aggregations):

- **Sharpe Ratio**: Excess return per unit of total risk (RF = 4.5% p.a.)
- **Sortino Ratio**: Excess return per unit of downside risk only
- **Max Drawdown**: Largest peak-to-trough loss (DuckDB running-window MAX)
- **Calmar Ratio**: Annualised return / |Max Drawdown|
- **VaR 95%**: 5th percentile of daily returns (`PERCENTILE_CONT`)
- **CVaR 95%**: Average return given you are in the worst 5% of days

### 7. Anomaly Backtest (`07_backtest.ipynb`)
- Strategy: buy at close on confirmed anomaly day T, hold N days (5 / 10 / 20)
- Baseline: average N-day forward return on non-anomaly days
- DuckDB SQL computes per-ticker edge (anomaly return minus baseline)
- SPY trade log, win rate, P&L, and equity curve vs buy-and-hold

---

## Key Findings

1. **COVID crash (March 2020)** generated the highest density of confirmed anomalies across all tickers — visible in every anomaly chart as a cluster of red circles
2. **GLD** (gold) has near-zero correlation with equities (confirmed by heatmap) — behaves as a portfolio hedge
3. **Anomaly rate rises with VIX**: during Extreme VIX (>40) periods, the model flags >4× more anomalies than during Low VIX periods
4. **Prophet forecast MAPE = 3.88%** on a 6-month out-of-sample test — well within acceptable range for equity price forecasting
5. **AMZN** has the highest annualised volatility and the most confirmed anomalies; **XOM** has the fewest anomalies (more stable, commodity-anchored)

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data ingestion | `yfinance` |
| Data manipulation | `pandas`, `numpy` |
| SQL analytics | `DuckDB` (in-process, no server) |
| ML anomaly detection | `scikit-learn` IsolationForest |
| Time series forecast | `Prophet` (Facebook) |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `Streamlit` |
| Notebooks | Jupyter (`nbformat 4`) |

---

## Setup

```bash
pip install yfinance prophet scikit-learn duckdb streamlit plotly \
            pandas numpy matplotlib seaborn kaleido
```

Run all notebooks in order (01 → 07), then launch the dashboard:

```bash
streamlit run app.py
```
