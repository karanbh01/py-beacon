---
title: Analysis
description: "A run's performance, portfolio, risk and attribution analytics, and the helpers beneath them."
---

# Analysis

A run's performance, portfolio, risk and attribution analytics, and the older helpers for concentration, liquidity and ETF tracking. See [Performance](../concepts/performance.md), [Portfolio](../concepts/portfolio.md), [Risk analysis](../concepts/risk-analysis.md) and [Attribution](../concepts/attribution.md).

::: beacon.analysis

## A run's data

Everything a run's analysis reads, from a finished run or a stored one, read back exactly as it was written.

::: beacon.analysis.data

## A run's tables

The large tables a run keeps beside its record, such as the fund's daily weights in every name, as Parquet files.

::: beacon.analysis.panels

## Snapshots at each rebalance

What a run's names looked like at each rebalance, such as their sector and country, kept with the run so a later rerun of its analysis gives the same answer.

::: beacon.analysis.snapshots

## Performance

A run's performance, its figures against its comparators, in crises and in each kind of market, and the formulas each figure is computed with. See [Performance](../concepts/performance.md).

::: beacon.analysis.performance.analytics

::: beacon.analysis.performance.settings

::: beacon.analysis.performance.relative_analytics

::: beacon.analysis.performance.crises

::: beacon.analysis.performance.regimes

::: beacon.analysis.performance.formulas

## Portfolio

What a run held and traded, and what it cost. See [Portfolio](../concepts/portfolio.md).

::: beacon.analysis.performance.portfolio

## Risk

A run's factors, value at risk, risk over time, concentration and liquidity. See [Risk analysis](../concepts/risk-analysis.md).

::: beacon.analysis.risk.analytics

::: beacon.analysis.risk.model

::: beacon.analysis.risk.forecast

## Attribution

A run's attribution: by name and group, Brinson-Fachler, currency, factors and implementation. See [Attribution](../concepts/attribution.md).

::: beacon.analysis.attribution.analytics

## Analysis requests

What a run's analysis should give: the reports, the statistics added to or taken from them, and every choice the analysis makes. See [Reports](../concepts/reports.md).

::: beacon.analysis.request
