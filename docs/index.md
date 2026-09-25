# Beacon

Beacon (***Be***t***a*** ***Con***structor) is a Python toolkit for end-to-end
index, ETF, and Delta-1 derivatives development — from defining an index
methodology, through calculating its historical levels, to backtesting a
tracking portfolio and analysing the result.

!!! info "Status"
    Under active development. The distribution name is `py-beacon`; the
    import package is `beacon`.

## The three-layer pipeline

Beacon is organised around a three-layer pipeline. Each layer has a single
responsibility and depends only on the layer(s) below it, which keeps the
methodology, the calculation, and the simulation cleanly separated.

```
        ┌──────────────────────────────────────────────┐
        │  Methodology                                   │
        │  eligibility rules + weighting schemes         │
        │  (what belongs in the index and at what weight)│
        └───────────────────────┬────────────────────────┘
                                │  defines
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Calculator                                    │
        │  IndexCalculator.run() -> IndexResult          │
        │  (levels, divisor, constituent/weight history) │
        └───────────────────────┬────────────────────────┘
                                │  target weights
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Backtest                                      │
        │  BacktestEngine.run() -> BacktestResult        │
        │  (NAV, trades, tracking error vs. the index)   │
        └──────────────────────────────────────────────┘
```

1. **Methodology** (`beacon.index.methodology`, `beacon.index.constructor`) —
   `IndexDefinition` holds the static rules (universe, currency, base date,
   rebalance frequency); eligibility rules and weighting schemes
   (`EqualWeighted`, `MarketCapWeighted`) decide membership and weights.
2. **Calculator** (`beacon.index.calculation`) — `IndexCalculator.run()`
   iterates business days and returns an `IndexResult` holding index levels,
   divisor history, and constituent/weight snapshots keyed by rebalance date.
3. **Backtest** (`beacon.backtest.engine`) — `BacktestEngine` consumes a
   target weight schedule (an `IndexResult` **or** a raw
   `{Timestamp: {asset_id: weight}}` dict), simulates trading (sells before
   buys, costs in bps on notional), and returns a `BacktestResult` with NAV,
   transactions, and tracking metrics versus the target index.

Funds (`IndexFund`, `ETF`) compose the Calculator and Backtest layers, and the
Derivatives layer prices instruments off the levels an `IndexResult`
produces.

## Modules

- **`index`** — Index construction and calculation. `IndexDefinition`
  captures the static rules; `methodology` provides the eligibility rules and
  weighting schemes (e.g. `EqualWeighted`, `MarketCapWeighted`);
  `IndexCalculator` runs the day-by-day calculation and returns an
  `IndexResult` with index levels, divisor history, and constituent/weight
  snapshots.
- **`backtest`** — Portfolio simulation. `BacktestEngine` consumes a target
  weight schedule, simulates trading with configurable transaction costs, and
  returns a `BacktestResult` exposing NAV, cash and weight history,
  transactions, and tracking metrics.
- **`portfolio`** — The `Portfolio` accounting primitive: holdings, cash,
  transactions, valuation and weights, plus Excel reporting helpers. It has
  no dependency on assets or data sources — callers pass identifiers and
  prices.
- **`fund`** — Investable vehicles. `IndexFund` composes an `IndexCalculator`
  and a `BacktestEngine` to track an index (with management-fee accrual);
  `ETF` extends it with a ticker, creation-unit size, market-price
  simulation, and tracking-performance analysis.
- **`derivatives`** — Delta-1 instruments referencing indices/ETFs/equities:
  `IndexFuture`, `ETFFuture`, and `TotalReturnSwap`, built on a
  `DerivativeBase` ABC, plus pure `pricing` functions (cost-of-carry,
  discrete-dividend forward, implied repo, roll return, TRS breakeven
  spread).
- **`analysis`** — Performance and risk analytics, including ETF tracking
  metrics (`analysis.etf`), attribution, and risk measures.
- **`data`** — Market and reference data access. `MarketData`/`ReferenceData`
  wrap tabular sources and `DataFetcher` provides a unified query interface
  used throughout the calculation and backtest layers.
- **`environment`** — The `Environment` configuration object that
  centralises run-level settings.

See [Concepts](concepts/overview.md) for a narrative walkthrough of each
layer, or the [Reference](reference/index.md) section for the generated API
documentation.

## Installation

Beacon targets Python 3.11+. Clone the repository and install it in
editable mode (a virtual environment is recommended):

```bash
git clone https://github.com/karanbh01/py-beacon.git
cd py-beacon
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

The distribution is named `py-beacon-kit`; the import package is `beacon`. Core
dependencies (pandas, numpy, pydantic) are installed automatically.

Everything beyond the core pipeline lives behind an extra, so a plain install
stays light:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `data` | yfinance | Downloading market data |
| `excel` | openpyxl | `ReportGenerator` Excel output |
| `optimise` | scipy | Portfolio optimisation |
| `plot` | matplotlib | Chart accessors on result objects |
| `plot-interactive` | plotly | Interactive charts (planned) |
| `server` | fastapi, uvicorn, orjson, websockets | The local API server |
| `docs` | mkdocs-material, mkdocstrings[python] | Building this documentation site |
| `dev` | pytest, ruff, mypy, pre-commit, hypothesis | Contributing |

Install one with `pip install "py-beacon-kit[plot]"`, or several with
`pip install "py-beacon-kit[plot,data]"`. Using a feature without its extra
raises an error naming the extra to install.

## Quickstart

Define an index, calculate it, backtest a portfolio that tracks it, and view
the results. This snippet is fully self-contained (synthetic data, no
external dependencies) and copy-paste runnable:

```python
import logging

import pandas as pd

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.schedule import sessions

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

# 1. Data: two stocks priced on every New York Stock Exchange session.
days = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-28"), "XNYS")
growth = {"AAA": 0.10, "BBB": 0.20}  # each stock's rise over the period

market = MarketData.from_dataframe(pd.DataFrame([
    {"IDENTIFIER": name, "DATE": day, "SHARES_OUTSTANDING": 1_000,
     "CLOSE": 100 * (1 + rise) ** (step / (len(days) - 1))}
    for name, rise in growth.items()
    for step, day in enumerate(days)
]))
reference = ReferenceData.from_dataframe(pd.DataFrame([
    {"IDENTIFIER": name, "NAME": name, "CURRENCY": "USD",
     "EXCHANGE": "XNYS", "DATE_FROM": "2020-01-01"}
    for name in growth
]))
data = DataFetcher(market, reference)

# 2. The index: equal weight, rebalanced monthly on the NYSE calendar.
definition = IndexDefinition(
    index_id="DEMO", index_name="Demo Equal-Weight Index",
    base_date="2024-01-02", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=EqualWeighted(),
    rebalancing_frequency="MONTHLY", calendar="XNYS",
    universe_identifiers=list(growth),
)
index_result = IndexCalculator(definition, data).run(end_date="2024-03-28")
print("Final index level:", round(index_result.index_levels.iloc[-1], 2))

# 3. A backtest of a portfolio that trades to the index's weights.
backtest = BacktestEngine(
    start_date="2024-01-02", end_date="2024-03-28",
    initial_capital=1_000_000.0, data_provider=data,
    index_result=index_result, calendar="XNYS",
).run()

summary = backtest.summary()
print("Total return:  ", round(summary["total_return"], 4))
print("Tracking error:", round(summary["tracking_error"], 6))
```

The [example notebooks](https://github.com/karanbh01/py-beacon/tree/main/examples)
go further: backtest analysis, index futures, and optimised indices.
