# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing has been released yet — the repository carries no tags, so everything
below is unreleased work against version `0.0.2`. While the major version is
`0`, the public API may change in any release; see the versioning policy in the
README.

### Added

- **Packaging**: PEP 621 metadata with `hatchling`, a `src/` layout, and
  `__version__` single-sourced from `src/beacon/__init__.py`. The distribution
  is `py-beacon`; the import package is `beacon`.
- **Optional dependency extras**: `data`, `excel`, `optimise`, `plot`,
  `plot-interactive`, `server`, `docs`, `dev`. The core imports with only
  pandas, numpy and pydantic; anything else is reached through
  `beacon._optional.require()`, which raises `MissingDependencyError` naming
  the extra to install.
- **Type information**: `py.typed` ships in the wheel, so downstream `mypy`
  sees Beacon's types.
- **Tooling**: `ruff` lint and `mypy --strict` as enforced gates, pre-commit
  hooks (lint on commit, types on push), a CI matrix over Python 3.11–3.13 and
  Linux/macOS/Windows, a tag-triggered release workflow producing draft
  releases with wheel and sdist, and a mkdocs-material documentation site.
- **Property tests**: `tests/test_properties.py` covers weighting-scheme
  invariants, divisor continuity, portfolio cash accounting, return/level
  recompounding, and cost-of-carry pricing identities. Coverage is gated at
  80%.
- **Derivatives layer**: `DerivativeBase`, `IndexFuture`, `ETFFuture` and
  `TotalReturnSwap`, plus a dependency-free `pricing` module (cost of carry,
  discrete-dividend forward, implied repo, roll return, TRS breakeven spread).
- **Result objects**: `IndexResult` and `BacktestResult`, each with opt-in data
  binding via `.with_data()` and queryable per-asset views (`IndexAssetView`,
  `BacktestAssetView`).
- **Asset types**: `Bond` and `Commodity` alongside `Equity`, and a base
  `AssetView` pairing an identifier with a `DataFetcher`.
- `IndexDefinition.get_rebalance_dates()` for MONTHLY/QUARTERLY/SEMI-ANNUAL/
  ANNUAL schedules, and a `universe_identifiers` field defining the selectable
  universe.
- `BacktestModifier` with `DriftThresholdModifier`, replacing the former rule
  system.

### Changed

- `Asset` and `Equity` are frozen dataclasses — immutable metadata containers
  rather than data fetchers. Data access belongs to `DataFetcher`.
- `IndexCalculationAgent` renamed to `IndexCalculator`, and made stateless:
  `run()` holds its own working state, so repeated calls are side-effect free
  and idempotent.
- `Portfolio` no longer depends on `DataFetcher`. Callers pass prices in via
  `update_prices()`, holdings are keyed by string `asset_id`, and
  `add_transaction` split into `execute_buy` / `execute_sell`.
- `BacktestEngine` consumes a target weight schedule — an `IndexResult` or a
  raw `{Timestamp: {asset_id: weight}}` dict — instead of recalculating an
  index. It manages its own portfolio and executes sells before buys.
- `IndexFund` composes `IndexCalculator` and `BacktestEngine` and holds no
  buy/sell logic of its own.
- The whole pipeline drives off `DataFetcher.fetch_market_data()`; the legacy
  `fetch_prices(...)['Adj Close']` interface was removed.
- Logging levels standardised: INFO for lifecycle, DEBUG for per-trade detail,
  WARNING for missing data, ERROR for result-affecting failures.
- Docstrings normalised to Google style so the API reference builds under
  `mkdocs build --strict`.

### Removed

- `MethodologyRule` base class and its `.apply()` proxy methods.
- `requirements.txt`, superseded by the pyproject extras. Its pins contradicted
  the project metadata and nothing referenced it.
- Circular-import workarounds throughout the codebase. One intentional
  `TYPE_CHECKING` import remains in `backtest/engine.py`; see CLAUDE.md.

### Fixed

- `ReportGenerator.generate_holdings_report_excel` called
  `Portfolio.get_holdings_summary()` with two arguments against a zero-argument
  method, raising `TypeError` on every call. The Excel holdings report had
  never worked. Found by `mypy --strict`.
- `market_data` on the derivatives pricing methods was annotated
  `dict[str, float]` while callers pass a list of dividend tuples under
  `discrete_dividends`.
- `ETF.get_tracking_performance` declared `dict[str, float]` but returns an
  error string when no target index is present.
- Two `Optional` values were dereferenced without narrowing, in
  `backtest/engine.py` and `index/calculation/corporate_actions.py`.

### Deprecated

- Nothing is deprecated. See the deprecation policy in
  [CONTRIBUTING.md](CONTRIBUTING.md#deprecation-policy) for how removals will
  be handled once the API is public.

[Unreleased]: https://github.com/karanbh01/py-beacon/commits/main
