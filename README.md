# Machine Learning for Intraday Pairs Trading

> **A benchmark study of nine pair-selection methods for statistical arbitrage on 1-minute S&P 500 data, combining classical criteria (distance, cointegration, correlation) with machine-learning filters (K-means clustering, autoencoder-based neighborhoods).**

![Python](https://img.shields.io/badge/python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-completed-success)

**Author:** Bryan Djoufack · M2 Research Project · April 2026

📄 [**Read the detailed report (PDF, 30 pages)**](reports/pairs_trading_report_detailed.pdf) · 📄 [Short executive summary (8 pages)](reports/pairs_trading_report.pdf)

---

## 🎯 Key Results

| Method | Test Sharpe (net) | % PnL from outliers | Verdict |
|---|---|---|---|
| ae+cointegration | 1.59 | 44% | Event-driven (AMD) |
| ae+correlation | 1.17 | 28% | Modest gain from ML |
| kmeans+cointegration | 1.06 | **77%** | **Event-driven (AMD)** |
| **ae+distance** ⭐ | **1.05** | **-0.5%** | **Broad-based, robust** |
| kmeans+distance | 1.05 | 37% | Comparable to AE |
| correlation (baseline) | 0.79 | 29% | Respectable baseline |
| distance (baseline) | 0.34 | 55% | Modest |
| cointegration (baseline) | -0.05 | n/a | Fails OOS |

**Main finding:** While several methods display high apparent Sharpe ratios, most are driven by a single idiosyncratic event — the AMD-OpenAI partnership (October 2025). **Only `ae+distance` delivers genuine broad-based alpha**, passing all four robustness tests.

### Fama-French factor analysis (train period, FF5)

- Daily alpha = **36.6 bps** (annualized ~92%)
- t-statistic = **3.55** (significant at 0.1%)
- R² = 4.3% (market-neutral confirmed)
- Benchmark: Stübinger & Bredthauer (2017) report 16 bps/day with t = 13.77 on S&P 500 over 1998-2015

Our alpha is **more than twice Stübinger's benchmark in absolute terms**, with similar R² confirming statistical arbitrage nature.

---

## 📋 Methodology at a Glance

- **Data:** 1-minute OHLC bars on 37 S&P 500 large-caps, 2022-01 to 2026-03 (1,063 trading days)
- **Walk-forward:** 10-day formation + 5-day trading, 208 non-overlapping windows
- **Train/Test split:** 73/27 strict temporal split (153 train + 55 test windows)
- **Signals:** Stübinger's **EV strategy** (varying Bollinger bands, rolling 1-day window, k=2.5)
- **Transaction costs:** two-tier model, 5-8 bps round-trip
- **Factor analysis:** Fama-French 5-factor + momentum, Newey-West HAC standard errors

## 🧠 Machine Learning Components

**K-means filter (K=7):** tickers clustered on hand-crafted features (volatility, beta, market correlation, PCA 1-3, sector dummies). Selection restricted to intra-cluster pairs.

**Autoencoder filter (latent dim = 10):** symmetric feedforward autoencoder trained on 120-day rolling returns. Latent dimension chosen automatically via PCA 95% variance threshold. Neighborhoods defined by k=7 nearest neighbors in latent space.

**Key insight:** the autoencoder spontaneously recovers sector structure without being told about sectors — strong evidence that the learned representation captures meaningful ticker dynamics.

---

## 📂 Repository Structure

```
├── notebook/
│   └── pairs_trading_ml.ipynb       # Main notebook (120 cells, 8 steps, all outputs)
├── reports/
│   ├── pairs_trading_report.pdf           # Short report (8 pages, executive summary)
│   ├── pairs_trading_report_detailed.pdf  # Detailed report (30 pages + appendices)
│   ├── pairs_trading_report.tex           # LaTeX source (short)
│   └── pairs_trading_report_detailed.tex  # LaTeX source (detailed)
├── figures/                          # All figures from the notebook
├── requirements.txt                  # Python dependencies
├── .gitignore
├── LICENSE
└── README.md                         # This file
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/bryandjoufack/pairs-trading-ml.git
cd pairs-trading-ml
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Acquire 1-minute OHLC data

See [Data section](#-data) below for format specification and data sources.

### 4. Run the notebook
```bash
jupyter notebook notebook/pairs_trading_ml.ipynb
```

**Runtime:** approximately 60-75 minutes total (CPU-only, no GPU needed).

### 5. Notebook structure

| Step | Description | Runtime |
|---|---|---|
| 0 | Data preparation (cleaning, duplicate detection, universe filtering) | ~2 min |
| 1 | Baseline backtest (distance, cointegration, correlation) | ~6 min |
| 2 | Transaction cost modeling | ~7 min |
| 3 | K-means filtering (6 methods) | ~15 min |
| 4 | Autoencoder filtering (9 methods total) | ~30 min |
| 5 | Robustness analysis (4 experiments) | ~30 min |
| 6 | Summary and synthesis | <1 min |
| 7 | Fama-French factor analysis | ~1 min |

---

## 📊 Data

> ⚠️ **The 1-minute OHLC data used in this study is not included in the repository for licensing reasons.** Market data from brokers and commercial providers is generally subject to redistribution restrictions.

The notebook outputs (graphs, metrics, tables) are preserved in the committed `.ipynb` file, so you can inspect all results without re-running the backtest.

### Expected data format

One CSV per ticker, with columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | 1-minute bar timestamp (ET) |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | int | Volume |

Example filename: `AAPL.csv`, `MSFT.csv`, etc.

### Approximate data size

The full dataset used in this study:
- **Period:** 2 January 2022 to 30 March 2026 (~1,063 trading days)
- **Per-ticker file:** ~40 MB (~380 bars/day × 1,063 days)
- **Full dataset (37 tickers after cleaning):** ~1.5 GB
- **Raw universe (72 tickers before filtering):** ~2.4 GB

### Universe

37 tickers retained after data-quality filtering:

```
Technology (14): AAPL, MSFT, NVDA, TSLA, AMZN, AMD, AVGO, CRM, CSCO,
                 INTC, NFLX, ORCL, QCOM, TXN
Finance (7):     JPM, BAC, WFC, C, MS, SCHW, V
Healthcare (5):  UNH, JNJ, MRK, CVS, GILD
Consumer (5):    NKE, SBUX, TGT, PEP, PG
Energy (5):      XOM, CVX, COP, SLB, OXY
Industrial (1):  BA
```

### Where to acquire compatible data

**Commercial providers** (paid):
- [Polygon.io](https://polygon.io/) — Stocks Starter plan ~$29/month, unlimited queries
- [Databento](https://databento.com/) — pay-as-you-go, high-quality historical
- [QuantQuote](https://quantquote.com/) — academic discounts available
- Broker APIs (Interactive Brokers, Alpaca Markets with free tier)

**Academic access** (free, institutional):
- [WRDS](https://wrds-www.wharton.upenn.edu/) — CRSP, TAQ databases
- University financial-data subscriptions

**Limited free sources:**
- Yahoo Finance 1-minute (last 7-30 days only, via `yfinance`)
- Alpha Vantage (limited API calls)

Once data is acquired, place the CSVs in `data/raw/` (create the folder) and the notebook Step 0 will handle loading and cleaning.

---

## 🔬 Main Findings

### 1. Classical cointegration has lost its edge
Out-of-sample Sharpe of -0.05 confirms the post-2003 decline documented by Do & Faff (2010).

### 2. Machine-learning filtering helps distance-based selection monotonically
- Train Sharpe: baseline (0.68) < K-means (1.73) < autoencoder (**2.09**)
- Test Sharpe: baseline (0.34) < K-means (1.05) = autoencoder (1.05)

### 3. High-Sharpe cointegration methods are event-driven
`kmeans+cointegration` (Sharpe 1.06) derives **77% of its test PnL** from just 6 outlier pair-windows concentrated around the October 2025 AMD-OpenAI announcement. Remove the outliers, and its effective Sharpe collapses toward zero.

`ae+cointegration` (Sharpe 1.59, highest in study) still derives **44%** of its PnL from outliers.

### 4. The autoencoder spontaneously recovers sector structure
Without being told about sectors, the AE's latent space groups tickers by GICS classification. TSLA and BA are spontaneously isolated as outliers — consistent with the K-means singleton clusters.

### 5. Robustness confirms ae+distance is the preferred method
- **Experiment A** — Regularization: 4/4 configs Sharpe > 1.3
- **Experiment B** — Neighborhood size: 3/3 values Sharpe > 1.2
- **Experiment C** — Outlier removal: only -0.5% impact on test PnL (vs 77% for kmeans+cointegration)
- **Experiment D** — 3× transaction costs: 96% of Sharpe retained

### 6. Fama-French alpha analysis confirms skill
Train period delivers **36.6 bps/day alpha** (t = 3.55, significant at 0.1%), more than twice Stübinger's 1998-2015 benchmark of 16 bps/day. Test period alpha remains positive but underpowered due to the short 257-day test window.

---

## 🎨 Illustrative Outputs

Key figures are available in the `figures/` folder:

- `fig_step4_subplots.png` — Nine-method comparison (signature plot)
- `fig_latent_space.png` — Autoencoder latent space with sector coloring
- `fig_kmeans_clusters.png` — K-means clustering on PC space
- `fig_exp_C_outliers.png` — Outlier PnL contribution by method
- `fig_ae_training.png` — Autoencoder training curves
- `fig_pca_variance.png` — PCA variance justifying latent dim = 10

---

## 📚 References

**Primary methodological references:**

- **Stübinger & Bredthauer (2017).** Statistical Arbitrage Pairs Trading with High-Frequency Data. *International Journal of Economics and Financial Issues*, 7(4):650-662. 🎯 *Core methodology*
- **Sarmento & Horta (2020).** Enhancing a Pairs Trading strategy with the application of Machine Learning. *Expert Systems with Applications*, 158. *K-means filter*
- **Jung (2024).** A Nearest-Neighbor Approach to Pair Trading. *Working paper.* *Autoencoder neighborhoods*

**Classical pairs trading:**

- **Gatev, Goetzmann & Rouwenhorst (2006).** Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies*, 19(3):797-827.
- **Do & Faff (2010).** Does Simple Pairs Trading Still Work? *Financial Analysts Journal*, 66(4):83-95.
- **Caldeira & Moura (2013).** Selection of a Portfolio of Pairs Based on Cointegration. *Brazilian Review of Finance*, 11(1):49-80.

**Factor models:**

- **Fama & French (1996).** Multifactor explanations of asset pricing anomalies. *Journal of Finance*, 51(1):55-84.
- **Fama & French (2015).** A five-factor asset pricing model. *Journal of Financial Economics*, 116(1):1-22.

**Microstructure:**

- **Corwin & Schultz (2012).** A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices. *Journal of Finance*, 67(2):719-760.
- **Holden & Jacobsen (2014).** Liquidity Measurement Problems in Fast, Competitive Markets. *Journal of Finance*, 69(4):1747-1785.

Full bibliography: see detailed report.

---

## 📝 License

This project's **code and analysis** are released under the MIT License — see [LICENSE](LICENSE) for details. Market data is not included and subject to its own licensing terms from the original providers.

---

## 🙋 Contact

**Bryan Djoufack**
M2 Research Project · [LinkedIn](https://www.linkedin.com/in/bryan-djoufack-6aa897195/) · [Email](bryan.djoufacknguessong@gmail.com)

Feedback and discussions welcome via GitHub Issues.

---

## 🏷️ Keywords

`pairs-trading` · `statistical-arbitrage` · `machine-learning` · `autoencoder` · `high-frequency-trading` · `fama-french` · `walk-forward` · `python` · `keras` · `scikit-learn`
