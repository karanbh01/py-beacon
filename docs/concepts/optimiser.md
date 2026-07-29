# Optimiser

!!! warning "Not yet implemented"
    There is no optimiser anywhere in `src/beacon` today. This page
    describes the concept and where it is expected to sit in the pipeline,
    not a working feature. It will be delivered by **BN-72**
    ([GitHub issue #88](https://github.com/karanbh01/py-beacon/issues/88)).

## What it will be responsible for

The `optimise` extra already exists in `pyproject.toml` (it installs
`scipy`), reserving the dependency for this component ahead of its
implementation. Conceptually, an optimiser sits alongside the
[Methodology](methodology.md) layer's Weighting step: where `EqualWeighted`
and `MarketCapWeighted` are closed-form rules, an optimiser would derive
weights numerically — for example, minimising tracking error to a benchmark
subject to constraints such as position limits, sector exposure caps, or a
turnover budget.

## How it will fit the existing pipeline

Nothing about the surrounding pipeline is expected to change shape:

- It would most likely implement the same `WeightingSchemeBase` contract
  (`calculate_weights(constituents, current_date, market_data_provider,
  context=None) -> dict[Asset, float]`) that `EqualWeighted` and
  `MarketCapWeighted` implement today, so `IndexCalculator` could drive it
  exactly as it drives any other weighting scheme.
- Its output would still flow into `IndexCalculator.run()` -> `IndexResult`
  -> `BacktestEngine.run()` -> `BacktestResult`, unchanged.
- It would depend on `scipy` behind `beacon._optional.require("scipy",
  ...)`, matching how every other optional dependency in Beacon is guarded
  (see `EXTRA_FOR_MODULE` in `src/beacon/_optional.py`).

No API for this exists yet, so there is nothing to import or call — check
back once BN-72 lands.
