# Universe

The universe is the pool of identifiers an index is allowed to draw
constituents from. It is the input to Selection (see
[Methodology](methodology.md)) — not every universe member ends up in the
index, but every index constituent must first come from the universe.

## Declaring a universe

`IndexDefinition.universe_identifiers` (`src/beacon/index/constructor.py`) is
an optional list of string identifiers (tickers, ISINs, etc.). If it is
`None`, the calculator treats the universe as empty on every date and logs a
warning.

```python
definition = IndexDefinition(
    index_id="DEMO",
    index_name="Demo Equal-Weight Index",
    base_date="2024-01-02",
    base_value=1000.0,
    currency="USD",
    eligibility_rules=[],
    weighting_scheme=EqualWeighted(),
    rebalancing_frequency="MONTHLY",
    universe_identifiers=["AAA", "BBB"],
)
```

## Resolving identifiers into assets

`IndexCalculator._get_universe(date)` (in
`src/beacon/index/calculation/calculator.py`) turns each identifier into an
`Equity` object:

1. For each identifier, it calls
   `self.data.fetch_reference_data(identifier, date_str)` on the bound
   `DataFetcher`.
2. If the lookup returns an empty frame, the identifier is skipped with a
   warning — it simply does not appear in that date's universe.
3. Otherwise a `beacon.asset.equity.Equity` is built from the `NAME`,
   `CURRENCY`, and `EXCHANGE` columns of the reference-data row, with the
   identifier itself used as the ticker.

This resolution runs on the base date and on every rebalance date, so a
universe member whose reference data has since changed (e.g. a currency
correction) or whose reference data disappears (e.g. a delisting) is
re-evaluated at each reconstitution, not just once at inception.

## Universe vs. constituents

The universe is deliberately broader than the constituent list: it is
whatever `select_constituents()` is given to filter, via the eligibility
rules described on the [Methodology](methodology.md) page. A `MarketCapRule`
or `LiquidityRule` can only ever narrow the universe down — Beacon has no
mechanism for adding names to the index that are not in
`universe_identifiers`.
