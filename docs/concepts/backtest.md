# Backtest

The backtest layer (`src/beacon/backtest/engine.py`) simulates a real,
tradeable portfolio against a target weight schedule — either an
`IndexResult` produced by the Calculator layer (see
[Methodology](methodology.md)), or a raw
`{pd.Timestamp: {asset_id: weight}}` dict. These two inputs are mutually
exclusive constructor arguments on `BacktestEngine`.

## What the engine does

`BacktestEngine.run()` walks business days from `start_date` to `end_date`
and, for each day:

1. Fetches current prices for every held asset and updates the internal
   `Portfolio`'s market values.
2. If the date is a key in the weight schedule (a rebalance date), generates
   and executes trades to move the portfolio toward the target weights, then
   re-prices.
3. Records the day's NAV, cash balance, and per-asset weight.

Rebalance dates come from the weight schedule's own keys — when driven by an
`IndexResult`, that means `IndexResult.weight_snapshots`, which in turn come
from `IndexDefinition.get_rebalance_dates()`. The engine only trades on
schedule dates that also fall on a simulated business day.

## Trade generation and costs

`_generate_trades()` produces a sells-before-buys list of `TradeInstruction`
objects:

- **Sells** — any held asset absent from the target weights, or overweight
  relative to its target, is trimmed (or fully liquidated).
- **Buys** — any target asset that is new or underweight is topped up, using
  cash freed by the sells.

Each trade's cost is `notional * (transaction_cost_bps / 10_000)`. A buy is
only executed if the portfolio has sufficient cash after costs; otherwise it
is skipped with a warning.

## Modifiers

`BacktestModifier` (in `rules.py`) is an optional hook point with two
extension methods:

- `should_skip_rebalance(date, portfolio, target_weights)` — veto a
  scheduled rebalance entirely.
- `adjust_trades(trades, date, portfolio)` — modify the generated trade list
  before execution.

`DriftThresholdModifier(threshold)` ships as an example: it skips a
rebalance entirely when every asset's drift from target is within
`threshold`, avoiding unnecessary turnover.

## Reading the result

`BacktestEngine.run()` returns a `BacktestResult` holding `portfolio_nav`,
`cash_history`, `transactions`, and `actual_weight_history`. When a target
`IndexResult` was supplied, `get_tracking_error()` and
`get_tracking_difference()` compare the portfolio's return series against
the index's; `summary()` bundles total/annualised return, volatility,
Sharpe ratio, max drawdown, and (if available) the two tracking metrics into
one dict.

## The index level is not the backtest NAV

This is a real, measured gotcha, not a bug: **the index level and the
backtest NAV are two different numbers, even at zero transaction cost.**

`IndexCalculator` computes a market-cap-style level — the sum of
`price * shares` (adjusted for FX and free float) across constituents,
divided by the divisor. `BacktestEngine`, by contrast, holds an actual
weight-rebalanced portfolio: on each rebalance date it trades to hit target
*weights*, and between rebalances those weights drift with each asset's own
price return.

The two only track exactly when constituent price paths are proportional
between rebalance dates (i.e. every asset returns the same percentage each
day) — in general they are not, so a small amount of tracking difference
between rebalances is expected and not itself evidence of a bug in either
layer. When writing tests or validating a new methodology, assert loose,
*measured* tracking bounds rather than assuming zero difference, or build an
independent reference level rather than comparing directly against the
`BacktestResult` NAV.
