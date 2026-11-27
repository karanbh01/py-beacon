---
title: Backtest
description: "Simulating a portfolio that trades to an index's target weights: the Backtest front door, BacktestEngine, and BacktestResult."
---

# Backtest

Simulating a portfolio that trades to an index's target weights: the `Backtest` front door, `BacktestEngine`, and `BacktestResult`. See [Backtest](../concepts/backtest.md).

::: beacon.backtest

## The fund's and the engine's records

`result.fund`, the vehicle's recorded data (its NAV, units, fees, flows, dividends and share classes, and the parts only some vehicles have), and `result.trading`, what the engine did (its rebalance steps, pricing, unfilled orders and costs). See [Moving to 0.11.0](../moving-to-0.11.md) for where each 0.10.0 name went.

::: beacon.backtest.fund_record

::: beacon.backtest.trading_record

## Daily breakdown

What a run cost each day, by kind (fixed trading costs, market impact, futures rolls and creation fees), and where its cash went: balances, cash that moved, and what was charged or owed without cash moving.

::: beacon.backtest.breakdown
