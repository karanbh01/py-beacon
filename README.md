# py-beacon

[![CI](https://github.com/karanbh01/py-beacon/actions/workflows/ci.yml/badge.svg)](https://github.com/karanbh01/py-beacon/actions/workflows/ci.yml)

![Beacon logo](https://raw.githubusercontent.com/karanbh01/py-beacon/main/logo.svg)

py-beacon is a Python library for building indices, ETFs and Delta-1
derivatives. You define an index methodology, calculate the index's history
on a real exchange calendar, backtest a portfolio that tracks it, and analyse
the result.

**Documentation:** [pybeacon.dev/library](https://pybeacon.dev/library/)
&middot; **Changelog:** [pybeacon.dev/changelog](https://pybeacon.dev/changelog/)
&middot; **Source:** [github.com/karanbh01/py-beacon](https://github.com/karanbh01/py-beacon)

## Install

py-beacon needs Python 3.11 or later.

```bash
pip install py-beacon-kit
```

In code, import it as `beacon`. The core installs pandas, numpy, pydantic and
exchange_calendars. Optional features are extras: `data` (Yahoo Finance
downloads), `excel`, `postgres`, `optimise`, `plot`, `pdf` and `server`.
Install them as `pip install "py-beacon-kit[plot,optimise]"`. The
[documentation](https://pybeacon.dev/library/#install) lists what each one
adds.

## Quickstart

Define an index, calculate it, and backtest a portfolio that tracks it. The
example builds its own small dataset, so it runs as it is. Each block is a
cell, as in a notebook, and the charts need the `plot` extra
(`pip install "py-beacon-kit[plot]"`).

```python
import logging

import numpy as np
import pandas as pd

from beacon.backtest import Backtest
from beacon.data import DataFetcher, MarketData, ReferenceData
from beacon.index import EqualWeighted, IndexDefinition
from beacon.index.schedule import sessions

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short
```

**Data.** Four stocks priced on every New York Stock Exchange session in
2024, from a seeded random walk, and the reference data that lists them.

```python
days = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
names = ["AAA", "BBB", "CCC", "DDD"]

rng = np.random.default_rng(28)
returns = rng.normal(0.0004, 0.015, size=(len(days), len(names)))
closes = pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)),
                      index=days,
                      columns=names)

prices = [{"IDENTIFIER": name,
           "DATE": day,
           "CLOSE": closes.at[day, name],
           "SHARES_OUTSTANDING": 1_000_000}
          for day in days
          for name in names]
listings = [{"IDENTIFIER": name,
             "NAME": name,
             "CURRENCY": "USD",
             "EXCHANGE": "XNYS",
             "DATE_FROM": "2020-01-01"}
            for name in names]

data = DataFetcher(MarketData.from_dataframe(pd.DataFrame(prices)),
                   ReferenceData.from_dataframe(pd.DataFrame(listings)))
```

**The index.** Equal weight across the four, rebalanced monthly on the NYSE
calendar.

```python
definition = IndexDefinition(index_id="DEMO",
                             index_name="Demo Equal-Weight Index",
                             base_date="2024-01-02",
                             base_value=1000.0,
                             currency="USD",
                             eligibility_rules=[],
                             weighting_scheme=EqualWeighted(),
                             rebalancing_frequency="MONTHLY",
                             calendar="XNYS",
                             universe_identifiers=names)
```

**The backtest.** Calculate the index, then simulate a portfolio trading to
its weights, paying 5 basis points on every trade.

```python
backtest = Backtest(initial_capital=100_000.0,
                    transaction_cost_bps=5.0,
                    data_provider=data)
result = backtest.run(definition,
                      end="2024-12-31")
```

**Results.** The index level and the portfolio's NAV each day, and the
portfolio's summary metrics against the index.

```python
daily = pd.DataFrame({"Index level": result.index.target.levels,
                      "Portfolio NAV": result.portfolio.nav})
daily.tail().round(2)
```

| | Index level | Portfolio NAV |
|---|---:|---:|
| 2024-12-24 | 1088.28 | 108747.08 |
| 2024-12-26 | 1091.06 | 109025.21 |
| 2024-12-27 | 1097.57 | 109675.70 |
| 2024-12-30 | 1119.11 | 111828.46 |
| 2024-12-31 | 1133.80 | 113295.48 |

```python
pd.DataFrame({"Portfolio": result.summary()}).round(4)
```

| | Portfolio |
|---|---:|
| total_return | 0.1330 |
| annualised_return | 0.1330 |
| volatility | 0.1214 |
| sharpe_ratio | 1.0950 |
| max_drawdown | -0.1148 |
| tracking_error | 0.0005 |
| tracking_difference | -0.0008 |

**Charts.** The index against its four stocks, then the portfolio's growth
and drawdown.

```python
from beacon.plot import use

use("light")  # the beacon chart style

rebased = closes.assign(Index=result.index.target.levels)
rebased = rebased / rebased.iloc[0] * 100

ax = rebased[names].plot(alpha=0.45,
                         title="The index and its stocks, rebased to 100")
rebased["Index"].plot(ax=ax,
                      linewidth=2.5,
                      legend=True)
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/karanbh01/py-beacon/main/docs/images/quickstart-stocks.dark.png">
  <img alt="The four stocks and the equal-weight index, each rebased to 100 over 2024. The index ends at 113.4." src="https://raw.githubusercontent.com/karanbh01/py-beacon/main/docs/images/quickstart-stocks.light.png" width="720">
</picture>

```python
result.plot.performance()
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/karanbh01/py-beacon/main/docs/images/quickstart-performance.dark.png">
  <img alt="The portfolio's growth of 100 over 2024, ending at 113.4, with its drawdown beneath, deepest at about 11 percent." src="https://raw.githubusercontent.com/karanbh01/py-beacon/main/docs/images/quickstart-performance.light.png" width="720">
</picture>

`result.index.target` holds the index the run aimed at, and
`result.portfolio` holds the simulated portfolio: its NAV, positions, cash and
transactions. The portfolio ends a little behind the index because it pays
for its trades.

## How it fits together

Data, the index and the backtest are separate layers. A `DataFetcher` serves
every calculation its data. An `IndexDefinition` says what the index holds,
and `IndexCalculator` computes its history as an `IndexResult`. A backtest
trades to the index's weights, and a derivative is priced off its levels.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/karanbh01/py-beacon/main/.github/assets/architecture-dark.png">
  <img alt="Data and the methodology feed the calculation, which produces an IndexResult. Derivatives are priced off it, and a backtest trades to its weights and produces a BacktestResult for analysis." src="https://raw.githubusercontent.com/karanbh01/py-beacon/main/.github/assets/architecture-light.png" width="464">
</picture>

A backtest has five parts. The strategy decides what is held and the
implementation how it is traded. Flows move money in and out, which the
vehicle deals at its price and charges its fees on. The modelling
assumptions are what the run takes as given.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/karanbh01/py-beacon/main/.github/assets/backtest-dark.png">
  <img alt="The strategy's weights go through the implementation as trades, and flows are dealt through the vehicle as units and fees. Both, with the modelling assumptions, feed Backtest.run(), which produces a BacktestResult." src="https://raw.githubusercontent.com/karanbh01/py-beacon/main/.github/assets/backtest-light.png" width="634">
</picture>

The strategy can hold the index in full, track it with fewer names, or try
to beat it. Each gives the backtest its target weights.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/karanbh01/py-beacon/main/.github/assets/strategies-dark.png">
  <img alt="From the index, an IndexDefinition holds every name at its index weight, IndexTracking holds fewer names, optimised or sampled, and an ActiveStrategy builds weights from a signal and a construction. Each gives target weights." src="https://raw.githubusercontent.com/karanbh01/py-beacon/main/.github/assets/strategies-light.png" width="670">
</picture>

## What it does

- **Indices.** Select constituents with eligibility rules, including
  expressions such as `data.market.market_cap > 1e9`; weight them equally or
  by (free-float) market cap; cap any one name's weight; and rebalance on a
  schedule against a real exchange calendar. Indices can be price, total
  return or net total return, and the divisor handles rebalances, special
  dividends and delistings.
  ([Methodology](https://pybeacon.dev/library/concepts/methodology/))
- **Backtests.** Simulate a portfolio trading towards the index's weights,
  with transaction costs, drift thresholds and a benchmark, then read its
  NAV, trades and tracking error.
  ([Backtest](https://pybeacon.dev/library/concepts/backtest/))
- **Data.** Load prices and reference data from CSV or Excel files, a data
  store on disk, a Postgres database or Yahoo Finance, or generate a
  realistic synthetic market. Names in several currencies are converted with
  FX rates. ([Data](https://pybeacon.dev/library/concepts/data/))
- **Funds and derivatives.** Index funds and ETFs with management fees;
  index and ETF futures and total return swaps, with pricing functions.
  ([Funds](https://pybeacon.dev/library/concepts/funds/),
  [Derivatives](https://pybeacon.dev/library/concepts/derivatives/))
- **Optimisation and risk.** Optimised indices and portfolios under
  constraints, covariance estimation with shrinkage, factor models, and risk
  contributions.
  ([Optimiser](https://pybeacon.dev/library/concepts/optimiser/),
  [Risk model](https://pybeacon.dev/library/concepts/risk-model/))
- **Analysis and output.** Return and risk attribution, concentration and
  drift, charts drawn from results, and PDF and Excel reports.
  ([Attribution](https://pybeacon.dev/library/concepts/attribution/),
  [Charts](https://pybeacon.dev/library/concepts/charts/),
  [Reports](https://pybeacon.dev/library/concepts/reports/))

## The local server

py-beacon also runs as a local API server, which is how the Beacon desktop
app uses it. This generates a synthetic dataset and serves it:

```bash
pip install "py-beacon-kit[server]"
python -m beacon.synthetic --seed 42
python -m beacon.server --port 0 --token dev
```

See the [server guide](https://pybeacon.dev/library/server/) and
[serving data](https://pybeacon.dev/library/serving-data/).

## Versioning

py-beacon follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0 the API is still settling: a breaking change raises the minor
version (0.1 to 0.2), and additions and fixes raise the patch. The full
policy is in
[CONTRIBUTING.md](https://github.com/karanbh01/py-beacon/blob/main/CONTRIBUTING.md#versioning-and-deprecation-policy).

## Contributing

See [CONTRIBUTING.md](https://github.com/karanbh01/py-beacon/blob/main/CONTRIBUTING.md)
for setting up a development install, the checks every change must pass, and
the release process.

py-beacon is released under the MIT licence.
