# Beacon

![Beacon logo](./logo.svg)

Beacon (**Be**t**a** **Con**structor) is an end-to-end Python toolkit for building, calculating, and backtesting systematic indices and ETF-style investment products.

## Installation

Beacon supports Python 3.9 and newer. From a local checkout, install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and test work, install pytest as well:

```bash
pip install -e . pytest
pytest
```

## Architecture overview

Beacon is organized around a three-layer workflow:

```text
+-----------------------------+
| Methodology                 |
| Index definitions, asset    |
| universes, selection rules, |
| and target weights          |
+-------------+---------------+
              |
              v
+-----------------------------+
| Calculator                  |
| IndexCalculator transforms  |
| the methodology into an     |
| IndexResult with levels,    |
| constituents, weights, and  |
| divisor history             |
+-------------+---------------+
              |
              v
+-----------------------------+
| Backtest                    |
| BacktestEngine applies the  |
| calculated index weights to |
| a Portfolio and returns a   |
| BacktestResult              |
+-----------------------------+
```

The output of each layer is designed to feed the next layer directly. For example, an `IndexResult` produced by the calculator can be passed to `BacktestEngine` as the target index, and the resulting `BacktestResult` can be used for performance and tracking analysis.

## Modules

- **`beacon.index`**: Index methodology, calculation, and result objects. This layer defines index construction inputs and produces `IndexResult` objects containing levels, divisor history, constituent snapshots, and weight snapshots.
- **`beacon.backtest`**: Portfolio simulation engine and result analytics. `BacktestEngine` consumes an `IndexResult` or explicit target weights, executes rebalances, and returns a `BacktestResult` with NAV, cash, transactions, tracking error, tracking difference, and summary statistics.
- **`beacon.portfolio`**: Portfolio accounting primitives. This module tracks cash, holdings, transactions, market value, realized/unrealized P&L, and target-weight rebalancing.
- **`beacon.fund`**: Fund wrappers for index-tracking products. `IndexFund` and `ETF` connect index definitions, portfolios, data providers, and fund-level metrics such as NAV and tracking performance.
- **`beacon.derivatives`**: Delta-1 derivative instruments and pricing helpers. This module is intended for index futures, ETF futures, swaps, and related pricing/mark-to-market utilities.
- **`beacon.analysis`**: Analytical helpers for returns, risk, attribution, and tracking metrics. Use this layer to compare portfolio outcomes against targets and summarize performance.
- **`beacon.data`**: Data-provider abstractions used by calculators and backtests. Providers expose market data to higher-level workflows while keeping examples and tests synthetic-friendly.
- **`beacon.environment`**: Runtime environment and configuration support for reproducible research workflows.

## Quickstart

The example below is fully synthetic: it defines target index weights, supplies in-memory prices through a small data provider, runs a backtest, and prints the resulting summary.

```python
import pandas as pd

from beacon.backtest.engine import BacktestEngine
from beacon.index.result import IndexResult


class SyntheticDataProvider:
    def __init__(self, prices):
        self.prices = prices

    def fetch_market_data(self, identifier, start=None, end=None, columns=None):
        price = self.prices.get(identifier, {}).get(start)
        if price is None:
            return pd.DataFrame()
        return pd.DataFrame({"CLOSE": [price]})


# 1. Define a simple methodology: monthly equal-weight target weights.
dates = pd.bdate_range("2025-01-02", periods=5)
weights = {dates[0]: {"AAA": 0.5, "BBB": 0.5}}

# 2. Build an IndexResult. In production this would usually come from
#    IndexCalculator; this minimal example creates one directly.
index_result = IndexResult(
    index_id="equal_weight_demo",
    index_levels=pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates),
    divisor_history=pd.Series(1.0, index=dates),
    constituent_snapshots={dates[0]: ["AAA", "BBB"]},
    weight_snapshots=weights,
)

# 3. Provide synthetic market prices for the portfolio backtest.
prices = {
    "AAA": {date.strftime("%Y-%m-%d"): 100.0 + i for i, date in enumerate(dates)},
    "BBB": {date.strftime("%Y-%m-%d"): 50.0 + 0.5 * i for i, date in enumerate(dates)},
}
data_provider = SyntheticDataProvider(prices)

# 4. Backtest the index-tracking portfolio.
engine = BacktestEngine(
    start_date=str(dates[0].date()),
    end_date=str(dates[-1].date()),
    initial_capital=10_000.0,
    data_provider,
    target_index_result=index_result,
    transaction_cost_bps=0.0,
)
result = engine.run()

# 5. View portfolio and tracking results.
print(result.portfolio_nav.tail())
print(result.summary())
```

## Current status

Beacon is under active development. The core index, portfolio, and backtest layers are covered by tests, while fund, derivative, and analysis workflows are being expanded.
