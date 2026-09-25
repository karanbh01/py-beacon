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
example builds its own small dataset, so it runs as it is:

```python
import logging

import pandas as pd

from beacon.backtest import Backtest
from beacon.data import DataFetcher, MarketData, ReferenceData
from beacon.index import EqualWeighted, IndexDefinition
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

# 3. Calculate the index, then simulate a portfolio trading to its weights,
#    paying 5 basis points on every trade.
backtest = Backtest(initial_capital=1_000_000.0,
                    transaction_cost_bps=5.0,
                    data_provider=data)
result = backtest.run(definition,
                      end="2024-03-28")

print("Final index level:", round(result.index.target.levels.iloc[-1], 2))
print("Final NAV:        ", round(result.portfolio.nav.iloc[-1], 2))
print("Tracking error:   ", round(result.summary()["tracking_error"], 6))
```

`result.index.target` holds the index the run aimed at, and
`result.portfolio` holds the simulated portfolio: its NAV, positions, cash and
transactions. The portfolio ends a little behind the index because it pays
for its trades.

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
