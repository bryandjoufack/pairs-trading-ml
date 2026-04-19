# Data Directory

> **This directory is intentionally empty in the public repository.**
> 1-minute OHLC data from brokers/commercial providers cannot be redistributed.

## How to populate this directory

Place your 1-minute OHLC CSV files in `data/raw/` with the following structure:

```
data/
└── raw/
    ├── AAPL.csv
    ├── MSFT.csv
    ├── NVDA.csv
    ├── ...
    └── (37 ticker files total after cleaning)
```

## Expected CSV format

Each file should contain 1-minute bars with these columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime (ET) | 1-minute bar timestamp |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | int | Volume |

## Example file head

```
timestamp,open,high,low,close,volume
2022-01-03 09:35:00,182.45,182.52,182.38,182.50,125000
2022-01-03 09:36:00,182.50,182.58,182.45,182.55,87000
2022-01-03 09:37:00,182.55,182.60,182.50,182.57,91000
...
```

## Expected date range

- Start: 2 January 2022
- End: 30 March 2026 (or latest available)
- Session: 9:35 to 15:54 ET (380 bars per complete day)

## Universe (37 tickers)

```
AAPL, AMD, AMZN, AVGO, BA, BAC, C, COP, CRM, CSCO, CVS, CVX,
GILD, INTC, JNJ, JPM, MRK, MS, MSFT, NFLX, NKE, NVDA, ORCL,
OXY, PEP, PG, QCOM, SBUX, SCHW, SLB, TGT, TSLA, TXN, UNH, V,
WFC, XOM
```

## Data providers

See the [main README](../README.md#-data) for a list of compatible data sources.

