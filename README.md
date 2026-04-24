# Intraday Pairs Trading with Machine Learning Filters

**Reproducing Stübinger & Bredthauer (2017) on the 2020-2026 S&P 500 sample, with K-means and autoencoder filters as remediation.**

> M2 research project — Bryan Djoufack, 2026

---

## Executive summary

This project benchmarks nine variants of an intraday statistical arbitrage pairs trading strategy on S&P 500 large caps, faithfully reproducing the methodology of [Stübinger & Bredthauer (2017)](#references) and extending it with two machine learning pair-selection filters: **K-means clustering** on formation-window features and a **rolling autoencoder k-nearest-neighbor filter** à la [Jung et al. (2024)](#references) (Approach B).

The headline finding is consistent with the declining-profitability literature ([Do & Faff, 2012](#references); [Landgraf, 2016](#references)): **over 2020-2026, all nine method variants exhibit a positive gross Sharpe but a negative net Sharpe after realistic transaction costs**. The statistical edge per round-trip has compressed to 1-3 bps on our sample, down from the ~100 bps implied by Stübinger's 1998-2015 annualised 50.5% return. This is not a bug in the implementation — the per-round-trip diagnostic, replicated across every method, confirms that the cost layer (10 bps/round-trip) is ~2× more conservative than the implicit cost scheme in Stübinger's paper (≈ 20 bps/round-trip).

The main result is therefore scientific rather than operational: **modern pair-selection ML improves the gross statistical edge** (kmeans+cointegration lifts gross per RT from 0.65 → 3.69 bps, a 5.7× improvement) but is **insufficient to offset the secular decline of pairs trading profitability** documented by Landgraf (2016) extending from 2008 to the present.

---

## Results at a glance

### 9-method performance — full period (Jan 2020 to Mar 2026)

| Method | Sharpe (gross) | Sharpe (net) | Total PnL (gross) | Total PnL (net) | Max DD (net) |
|---|---:|---:|---:|---:|---:|
| **ae+correlation** | **1.95** | −4.62 | **+34.3%** | −78.6% | −80.1% |
| correlation | 1.79 | −4.30 | +34.5% | −80.2% | −82.2% |
| ae+distance | 1.78 | −6.13 | +27.6% | −92.0% | −93.0% |
| kmeans+distance | 1.74 | −6.25 | +26.6% | −93.0% | −93.4% |
| kmeans+correlation | 1.66 | −4.64 | +30.5% | −82.5% | −84.5% |
| distance | 1.15 | −6.00 | +19.7% | −99.6% | −100.4% |
| **kmeans+cointegration** | 0.69 | **−1.48** | +37.8% | **−79.9%** | −93.7% |
| ae+cointegration | 0.63 | −2.74 | +22.4% | −95.4% | −97.6% |
| cointegration | 0.19 | −2.87 | +7.4% | −110.3% | −112.0% |

**Two "winners" for different questions:**
- Best **gross** signal: `ae+correlation` (Sharpe 1.95) — confirms autoencoders extract a meaningful statistical edge.
- Best **net** performance: `kmeans+cointegration` (Sharpe −1.48) — confirms clustering restricted to fundamentally similar pairs is the most robust to post-cost erosion.

### Per-round-trip diagnostic — the core finding

| Method | Gross / RT (bps) | Cost / RT (bps) | Net / RT (bps) |
|---|---:|---:|---:|
| distance | 1.70 | 10.39 | −8.69 |
| cointegration | 0.65 | 10.39 | −9.74 |
| correlation | 2.83 | 10.38 | −7.54 |
| kmeans+distance | 2.32 | 10.38 | −8.06 |
| kmeans+cointegration | **3.69** | 10.38 | **−6.69** |
| kmeans+correlation | 2.84 | 10.37 | −7.53 |
| ae+distance | 2.40 | 10.38 | −7.99 |
| ae+cointegration | 2.01 | 10.40 | −8.39 |
| ae+correlation | **3.18** | 10.37 | −7.20 |

**Stübinger (2017) benchmark**: gross ≈ 100 bps/RT, cost ≈ 20 bps/RT, net ≈ 80 bps/RT. Our gross/RT is 30-150× smaller. The costs we apply (10 bps/RT) are in fact 50% more conservative than Stübinger's implicit level.

### k-threshold sensitivity (Step 5)

Higher entry thresholds reduce the number of trades (from ~11 500 round-trips at k=2.5 to ~6 000 at k=3.5) and marginally increase gross per round-trip, but **net remains negative for every method**. The best net Sharpe at k=3.5 is `ae+correlation` at −2.91 (vs −4.62 at k=2.5). The cost drag remains the dominant factor at every k.

### Fama-French 5-factor alpha

All nine methods exhibit **statistically significant negative alpha** after costs, with market betas ≈ 0 (confirming the market-neutral design) and R² ≈ 1-2% (confirming the strategy is not loading on systematic factors):

| Method | Alpha ann. | t-statistic | R² |
|---|---:|---:|---:|
| kmeans+cointegration | −17.1% | −4.13 | 0.010 |
| ae+correlation | −17.8% | −12.78 | 0.010 |
| distance | −20.7% | −15.63 | 0.011 |
| cointegration | −22.0% | −7.25 | 0.006 |

For comparison, Stübinger's EV top-20 strategy over 1998-2015 reports **+40.3% annualised alpha with t = +13.77**.

---

## Methodology

### Data

- **Universe**: 39 S&P 500 large caps after clean-day filtering (Tukey outliers + 85% completeness floor)
- **Frequency**: M1 mid-prices from ICMarkets MetaTrader
- **Period**: 2020-01-16 → 2026-03-30 (1 558 trading days)
- **Sector coverage**: 11 GICS sectors

### Walk-forward protocol

Following Stübinger (2017) §4.2:
- 10-day formation window (pair selection + hedge ratio estimation)
- 5-day trading window
- 50-window warm-up before any trade (for ML filter stability)
- Non-overlapping windows → ~258 formation/trading pairs

### Pair selection methods (3 × 3 = 9 variants)

**Selection rules:**
- `distance` — minimum sum of squared normalised price distances ([Gatev et al., 2006](#references))
- `cointegration` — Engle-Granger ADF test, p < 0.05 ([Vidyamurthy, 2004](#references))
- `correlation` — highest Pearson correlation above 0.7 ([Chen et al., 2019](#references))

**Filter regimes:**
- `baseline` — no filter, top-N across the whole universe
- `kmeans` — K-means clustering (K=7, silhouette-optimal) on formation-window features; pairs restricted to intra-cluster
- `ae+` — autoencoder embeds formation prices to a 5-dim latent space; pairs are drawn from k-NN neighborhoods in latent space ([Jung et al., 2024](#references), Approach B)

The ML filters are **re-fitted every 10 windows** to adapt to regime shifts (Jaccard stability of AE neighborhoods: 0.51; K-means ARI: 0.94).

### Trading rule

Varying Bollinger bands with entry threshold k=2.5σ (robustness to k ∈ {2.5, 3.0, 3.5} in Step 5), mean exit on z-score crossing zero, as in [Stübinger & Bredthauer (2017)](#references) Table 4.

### Return computation

**Gatev (2006) eq (2)-(3)** with compounded leg weights, reporting **return on committed capital** (divides by TOP_N=10 pairs at every bar, even when some are flat). Method-specific capital allocation:
- distance & correlation: $1/$1 dollar-neutral (w_A = w_B = 0.5)
- cointegration: β-weighted (w_A = 1/(1+|β|), w_B = |β|/(1+|β|))

### Transaction costs

**Stübinger (2017) §4.5 convention: 5 bps per share per half-turn, all-in** (spread + commission + impact). Applied uniformly; the decomposition (2.5 bps half-spread + 1.5 bps commission + 1.0 bps slippage) is shown for pedagogical transparency but constrained to sum to 5 bps. Round-trip cost per pair ≈ 10 bps.

---

## Repository structure

```
├── data/
│   ├── README.md     
├── notebook/
|   ├── pairs_trading_ml_gatev.ipynb       
├── reports/
|   ├── pairs_trading_report_detailed.pdf                
└── README.md
```

### Notebook structure

| Step | Content | Output |
|---|---|---|
| 0 | Data loading, ticker filtering, time-grid alignment | `close_m1`, `close_d`, `day_flags` |
| 1 | Walk-forward baseline (3 methods, no costs) | `pairs_log`, baseline equity curves |
| 2 | Transaction cost model, gross/net PnL, per-RT diagnostic | `pairs_log_costs`, Bloc 2.4bis verdict |
| 3 | Sector-aware K-means rolling clustering | `clusters_history`, K-means pair lists |
| 4 | Autoencoder rolling fit + full 9-method backtest | `result_full`, signature equity curves |
| 5 | k-threshold sensitivity (k ∈ {2.5, 3.0, 3.5}) | `k_summary`, method × k pivots |
| 6 | Master comparison, sub-period stability, rolling Sharpe | `master_summary.csv` |
| 7 | Fama-French FF3 / FF3+MOM / FF5 regressions | Alpha t-stats across 9 methods |

---

## How to reproduce

### Prerequisites

```bash
python >= 3.10
pip install numpy pandas matplotlib scikit-learn statsmodels jupyter
```

Note: the autoencoder is implemented in **pure NumPy** (no PyTorch/TensorFlow dependency) for transparency and determinism.

### Run

```bash
jupyter notebook pairs_trading_ml_gatev.ipynb
# Run all cells. Full run-time ≈ 90-120 minutes on a standard laptop
```

Steps 1-4 share a common random seed; Steps 5-7 depend on the output dictionaries of Step 4.

---

## Key design decisions

1. **Method-specific PnL convention**: distance and correlation use naive $1/$1 allocation (no natural β); cointegration uses β-weighted allocation (Vidyamurthy 2004). This keeps total gross capital at $1 per pair regardless of method, so returns are directly comparable across families.

2. **Unified cost model**: 5 bps per share per half-turn all-in, matching Stübinger (2017) §4.5 exactly. This is **more favorable** than Landgraf (2016) who applies 50 bps/RT (4 commissions + 2 spread crossings).

3. **Return on committed capital**: we report this (rather than employed capital) as it is the conservative investor-perspective metric. Stübinger Table 7 shows the two are very close when the pair activity rate is high (EV: 50.50% vs 50.21%).

4. **Rolling ML re-fit**: both K-means and AE are re-fitted every 10 windows, not trained on the full sample. This prevents look-ahead bias and lets the filters adapt to regime shifts.

5. **Fama-French evaluation in Step 7**: we follow Stübinger's Table 9 convention to make alpha comparable with the literature.

---

## Interpretation

The negative net Sharpe across all nine methods is **consistent with three independent findings from the literature**:

1. **[Do & Faff (2012)](#references)**: "pairs trading profits decline over time, and high-frequency costs substantially erode returns".
2. **[Landgraf (2016)](#references)** §4.5.5: "From 2008-2016, all strategies become unprofitable with negative average returns" — our 2021-2026 period extends this finding.
3. **[Bowen & Hutchinson (2016)](#references)**: limited profitability post-decimalization in FTSE markets.

The ML filters partially mitigate but do not reverse this dynamic:
- K-means on cointegration lifts gross/RT from 0.65 → 3.69 bps (5.7×)
- The AE filter on correlation produces the best gross Sharpe (1.95)
- But no filter pushes gross/RT above the 10 bps cost floor

This suggests that **modern pair selection is necessary but not sufficient**. Potential avenues for further research — outside the scope of this project — would include: tighter timing rules (Velayutham 2nd-crossing filter, cited in Stübinger §4.2.3), broader universes (S&P 500 full coverage rather than our 39-ticker sample), or alternative microstructure-aware execution (Krauss et al., 2017, deep-learning approaches).

---

## References

- Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761-782.
- Bowen, D., Hutchinson, M. C., & O'Sullivan, N. (2010). High frequency equity pairs trading: transaction costs, speed of execution, and patterns in returns. *Journal of Trading*.
- Bowen, D., & Hutchinson, M. C. (2016). Pairs trading in the UK equity market: risk and return. *The European Journal of Finance*, 22(14), 1363-1387.
- Chen, H., Chen, S., Chen, Z., & Li, F. (2019). Empirical investigation of an equity pairs trading strategy. *Management Science*, 65(1), 370-389.
- Do, B., & Faff, R. (2010). Does simple pairs trading still work? *Financial Analysts Journal*, 66(4), 83-95.
- Do, B., & Faff, R. (2012). Are pairs trading profits robust to trading costs? *Journal of Financial Research*, 35(2), 261-287.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797-827.
- Jung, N., Oh, T., & Kim, K. (2024). Pairs trading using clustering and deep learning. *Journal of the Korea Data & Information Science Society*.
- Kishore, V. (2012). Optimizing pairs trading of US equities in a high frequency setting. Working paper, University of Pennsylvania.
- Krauss, C. (2017). Statistical arbitrage pairs trading strategies: review and outlook. *Journal of Economic Surveys*, 31(2), 513-545.
- Landgraf, N. (2016). High-frequency pairs trading on NASDAQ: an empirical analysis of transaction cost effects. Master's thesis, University of Hamburg.
- Stübinger, J., & Bredthauer, J. (2017). Statistical arbitrage pairs trading with high-frequency data. *International Journal of Economics and Financial Issues*, 7(4), 650-662.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.

---

## Author

**Bryan Djoufack** — M2 Financial Engineering, 2026. This project was developed as part of a portfolio for junior trading positions, with emphasis on scientific rigor, academic traceability, and honest reporting of negative-but-informative results.

For questions or discussion, open an issue or see accompanying research report (in separate file).
