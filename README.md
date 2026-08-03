# Beacon

[![CI](https://github.com/karanbh01/py-beacon/actions/workflows/ci.yml/badge.svg)](https://github.com/karanbh01/py-beacon/actions/workflows/ci.yml)

![Beacon logo](./logo.svg)

Beacon (***Be***t***a*** ***Con***structor) is a Python toolkit for end-to-end
index, ETF, and Delta-1 derivatives development — from defining an index
methodology, through calculating its historical levels, to backtesting a
tracking portfolio and analysing the result.

> Status: under active development.

## Architecture

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

Funds (`IndexFund`, `ETF`) compose the Calculator and Backtest layers, and the
Derivatives layer prices instruments off the levels an `IndexResult` produces.

## Modules

- **`index`** — Index construction and calculation. `IndexDefinition` captures
  the static rules (universe, currency, base date, rebalance frequency);
  `methodology` provides the eligibility rules and weighting schemes (e.g.
  `EqualWeighted`, `MarketCapWeighted`); `IndexCalculator` runs the day-by-day
  calculation and returns an `IndexResult` with index levels, divisor history,
  and constituent/weight snapshots.
- **`backtest`** — Portfolio simulation. `BacktestEngine` consumes a target
  weight schedule (an `IndexResult` or a custom weight dict), simulates trading
  with configurable transaction costs, and returns a `BacktestResult` exposing
  NAV, cash and weight history, transactions, and tracking metrics.
- **`portfolio`** — The `Portfolio` accounting primitive: holdings, cash,
  transactions, valuation and weights, plus Excel reporting helpers. It has no
  dependency on assets or data sources — callers pass identifiers and prices.
- **`fund`** — Investable vehicles. `IndexFund` composes an `IndexCalculator`
  and a `BacktestEngine` to track an index (with management-fee accrual); `ETF`
  extends it with a ticker, creation-unit size, market-price simulation, and
  tracking-performance analysis.
- **`derivatives`** — Delta-1 instruments referencing indices/ETFs/equities:
  `IndexFuture`, `ETFFuture`, and `TotalReturnSwap`, built on a `DerivativeBase`
  ABC, plus pure `pricing` functions (cost-of-carry, discrete-dividend forward,
  implied repo, roll return, TRS breakeven spread).
- **`analysis`** — Performance and risk analytics, including ETF tracking
  metrics (`analysis.etf`), attribution, and risk measures.
- **`data`** — Market and reference data access. `MarketData`/`ReferenceData`
  wrap tabular sources and `DataFetcher` provides a unified query interface used
  throughout the calculation and backtest layers. `data.store` persists a
  fetcher to disk so a spawned server can find one at startup.
- **`synthetic`** — A generator for market-like data at demo scale: a factor
  model with GJR-GARCH volatility and Student-t innovations, plus the reference
  data, shares outstanding, free float and corporate actions that agree with
  the prices it produces.
- **`environment`** — The `Environment` configuration object that centralises
  run-level settings.

## Installation

Beacon targets Python 3.11+. Clone the repository and install it in editable
mode (a virtual environment is recommended):

```bash
git clone https://github.com/karanbh01/py-beacon.git
cd py-beacon
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

The distribution is named `py-beacon`; the import package is `beacon`. Core
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
| `dev` | pytest, ruff, mypy, pre-commit, hypothesis | Contributing |

Install one with `pip install "py-beacon[plot]"`, or several with
`pip install "py-beacon[plot,data]"`. Using a feature without its extra raises
an error naming the extra to install. To run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

Contributors should also install the git hooks once per clone:

```bash
pre-commit install                        # lint + whitespace checks on commit
pre-commit install --hook-type pre-push   # strict type check on push
```

`ruff check` and `mypy` are the enforced gates. Code formatting is not
tool-enforced — signature layout follows a reviewed convention rather than a
formatter, so `ruff format` is deliberately not part of the hook set.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the conventions, the issue and
commit format, and the release process, and [CHANGELOG.md](./CHANGELOG.md) for
what has changed.

## Running the API server

```bash
pip install -e ".[server]"
python -m beacon.server --port 0 --token dev
```

The process binds first and prints `BEACON_PORT=<n>` on stdout before serving,
so a parent process launching it with `--port 0` can read back the port the OS
chose. Every later stdout line is ordinary logging.

### Where the server gets its data

A data source is resolved at startup, in this order:

1. `--data <path>` — an explicit store directory
2. `$BEACON_DATA_PATH`
3. the app-data store, auto-loaded if one has been written there
4. nothing — the server starts data-less and the data endpoints report
   `CONFIGURATION_ERROR` until a sync populates one

The branch that ran is logged immediately after the port announcement, so an
empty client is diagnosed by reading the log rather than by guessing. The first
two branches fail loudly: asking for a store that cannot be read stops startup,
because starting empty instead would disguise the mistake. Auto-load only
warns, so a corrupt store cannot leave the client unable to start the server
that would replace it.

### Which origins may call it

`localhost` on any port is always allowed, so a dev build needs no
configuration. Beyond that the defaults are `beacon://app` (the packaged
renderer's origin) and `app://`. To set them explicitly:

```bash
python -m beacon.server --cors-origin beacon://app --cors-origin app://custom
BEACON_CORS_ORIGINS="beacon://app,app://custom" python -m beacon.server
```

Explicit origins **replace** the defaults rather than adding to them — an
operator narrowing what may call the server should not find extras still
permitted. The allowed set is logged at startup, because a CORS failure
otherwise appears only in a browser console on the far side of the process
boundary.

### Discovering what a methodology can contain

`GET /indices/rule-types` publishes the eligibility rules and weighting schemes
the library provides, with enough detail to render an editor: each parameter's
name, display type, whether it is required, its default, a label, its position
in the form, and any closed set of choices.

`GET /optimise/constraint-types` serves the optimiser's constraints in the same
shape under `specs`, so one client component can render both editors. Its
original `types` field is unchanged.

Both come from a registry the classes populate themselves
(`beacon.catalogue`). Names, types, defaults and required-ness are read from
the constructors, so they cannot drift from what the code accepts; only labels
and ordering are declared, because a signature cannot carry them. A rule class
that exists without a catalogue entry fails a completeness test — the symptom
otherwise is silent, since the rule still works and the editor simply never
offers it.

### Looking up many instruments at once

`GET /data/reference` is the batch form of `/data/reference/{identifier}`:

```
/data/reference?identifiers=AAA,BBB,CCC&fields=NAME,SECTOR,adv_3m
```

Entries come back in the order the request named them, one per identifier, so
a table renders straight down the list. An unknown identifier is an entry with
`found: false` rather than a failed batch — one bad ticker in five hundred
should not lose the other 499. At most 1000 identifiers per call.

`fields` selects stored reference columns and may also name a derived field.
`adv_3m` is mean daily volume over the trailing three *calendar* months,
computed server-side from held prices; it is opt-in, because computing it means
slicing price history for every identifier in the batch.

### Generating data to serve

`beacon.synthetic` produces a universe at demo scale — hundreds of anonymised
companies with years of history — and writes it straight to the location above:

```bash
python -m beacon.synthetic --assets 512 --start 2019-12-31 --seed 42
python -m beacon.server --port 0 --token dev      # picks it up automatically
```

Prices reproduce the stylized facts of equity returns rather than being a
random walk: volatility clustering (GJR-GARCH), fat tails (Student-t
innovations), negative skew, and a market/sector factor structure that puts
average pairwise correlation near 0.39 with same-sector pairs above
cross-sector ones. Shares outstanding, free float, dividends and splits are
generated alongside the prices and agree with them — undoing the splits and
adding the dividends back recovers the return path exactly.

Nothing resembles a real company: names are `Company A` … and every ticker
carries a `CMP` prefix. The same seed and dates always produce the same store,
byte for byte.

It is importable too, which is what examples and integration tests use:

```python
from beacon.synthetic import SyntheticConfig, generate

dataset = generate(SyntheticConfig(assets=64, seed=1))
fetcher = dataset.fetcher()
```

This is not `beacon.testing.dataset`, which stays a tiny frozen fixture whose
exact values the chart baselines depend on.

### The store format

A store is a directory of gzipped CSV written by `beacon.data.store`:

```python
from pathlib import Path
from beacon.data import store

store.save(fetcher, Path("~/beacon-data").expanduser(), source="local")
```

`store.default_path()` is the app-data location branch 3 reads.

## Versioning

Beacon follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0` the public API may change in any release — the
surface is still settling. Breaking changes are recorded under **Changed** or
**Removed** in the changelog. From 1.0 onward, a deprecated name keeps working
for at least one minor release with a `DeprecationWarning` naming its
replacement, and removals only land in a major release; the full policy is in
[CONTRIBUTING.md](./CONTRIBUTING.md#versioning-and-deprecation-policy).

## Quickstart

Define an index, calculate it, backtest a portfolio that tracks it, and view
the results. This snippet is fully self-contained (synthetic data, no external
dependencies) and copy-paste runnable:

```python
import logging
import pandas as pd
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.calculation import IndexCalculator
from beacon.backtest.engine import BacktestEngine

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the demo output clean

# --- 1. Synthetic market data: two assets over ~3 months of business days ---
ASSETS = ["AAA", "BBB"]
DAYS = pd.bdate_range("2024-01-02", "2024-03-29")

def price(asset,
          day):
    frac = DAYS.get_loc(day) / (len(DAYS) - 1)
    return (100 * 1.10 ** frac) if asset == "AAA" else (50 * 1.20 ** frac)

class QuickData:
    """Tiny in-memory provider satisfying the calculator + engine data APIs."""
    def fetch_reference_data(self,
                             identifier,
                             date=None):
        return pd.DataFrame(
            {"NAME": [identifier], "CURRENCY": ["USD"], "EXCHANGE": ["NYSE"]},
            index=pd.Index([identifier], name="IDENTIFIER"))
    def fetch_market_data(self,
                          identifier,
                          start=None,
                          end=None,
                          columns=None):
        p = price(identifier, pd.Timestamp(start))
        return pd.DataFrame({"CLOSE": [p]}, index=pd.Index([pd.Timestamp(start)], name="DATE"))
    def fetch_shares_outstanding(self,
                                 ticker,
                                 date):
        return 1_000

data = QuickData()

# --- 2. Define the index: equal-weight, rebalanced monthly ---
definition = IndexDefinition(
    index_id="DEMO", index_name="Demo Equal-Weight Index",
    base_date="2024-01-02", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=EqualWeighted(),
    rebalancing_frequency="MONTHLY", universe_identifiers=ASSETS,
)

# --- 3. Calculate the index ---
index_result = IndexCalculator(definition, data).run(end_date="2024-03-29")
print("Final index level:", round(index_result.index_levels.iloc[-1], 2))

# --- 4. Backtest a portfolio that tracks the index ---
backtest = BacktestEngine(
    start_date="2024-01-02", end_date="2024-03-29",
    initial_capital=1_000_000.0, data_provider=data,
    target_index_result=index_result,
).run()

# --- 5. View results ---
summary = backtest.summary()
print("Total return:      ", round(summary["total_return"], 4))
print("Annualised return: ", round(summary["annualised_return"], 4))
print("Tracking error:    ", round(summary["tracking_error"], 6))
```

For a derivatives walkthrough — pricing an `IndexFuture` off an `IndexResult` —
see [`examples/futures_pricing_example.py`](./examples/futures_pricing_example.py).
