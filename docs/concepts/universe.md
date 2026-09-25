# Universe

The universe is the pool of identifiers an index may draw constituents from.
Every constituent comes from the universe, but not every universe member
becomes a constituent: [selection](methodology.md#selection) narrows it on
each rebalance. Nothing can add a name to the index that is not in the
universe.

## Declaring a universe

`IndexDefinition.universe_identifiers` is a list of identifiers (tickers,
ISINs or whatever the data uses). An empty list raises `ValueError` when the
definition is built. `None` is accepted by the definition, but the
calculation refuses it: resolving the universe raises `CalculationError`,
because an index over an unspecified universe has nothing to select from.

The examples on this page use the frozen sample dataset in
`beacon.testing.dataset`: six names on the XNYS calendar, `AAA` to `EEE`
quoted in USD and `FFF` in GBP, with a `GBPUSD` pair.

```python
import pandas as pd

from beacon.exceptions import CalculationError
from beacon.index import EqualWeighted, IndexCalculator, IndexDefinition
from beacon.testing import dataset

fetcher = dataset.data_fetcher()


def demo_index(universe_identifiers):
    return IndexDefinition(index_id="DEMO",
                           index_name="Demo Equal-Weight",
                           base_date="2024-01-02",
                           base_value=1000.0,
                           currency="USD",
                           eligibility_rules=[],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="QUARTERLY",
                           calendar="XNYS",
                           universe_identifiers=universe_identifiers)


try:
    IndexCalculator(demo_index(None), fetcher).run(end_date="2024-03-28")
except CalculationError as error:
    print(error)
```

```text
Error in calculation 'UniverseResolution': index 'Demo Equal-Weight' has no universe_identifiers, so there is nothing to select constituents from. An index cannot be calculated over an unspecified universe.
```

## Resolving identifiers into assets

On the base date and on every rebalance, the calculator turns the
identifiers into `Equity` objects. `IndexCalculator.resolve_universe(date)`
does the same thing on its own:

1. It reads the reference data for the whole universe in one call,
   `fetch_reference_data(identifiers, date)`. The read is point in time: for
   each name it returns the record valid on that date (`DATE_FROM` on or
   before it, and `DATE_TO` empty or on or after it).
2. An identifier with no valid record is skipped, with a warning. It is
   simply not in that date's universe.
3. Every other identifier becomes an `Equity` whose ticker is the identifier
   and whose name, currency and exchange come from its record's `NAME`,
   `CURRENCY` and `EXCHANGE` columns. If a column is absent, the name falls
   back to the identifier, the currency to the index currency and the
   exchange to `"UNKNOWN"`.

The assets come back in the order the definition lists them.

```python
calculator = IndexCalculator(demo_index(["AAA", "FFF", "ZZZ"]), fetcher)
assets = calculator.resolve_universe(pd.Timestamp("2024-01-02"))

print([(asset.asset_id, asset.currency, asset.exchange) for asset in assets])
```

```text
[('AAA', 'USD', 'NYSE'), ('FFF', 'GBP', 'LSE')]
```

`ZZZ` has no reference record, so it is dropped with the warning
`No reference data for 'ZZZ' on 2024-01-02. Skipping.`

Because resolution repeats on every rebalance, a change in a name's
reference data (a corrected currency, a new listing, a delisting) takes
effect at the next reconstitution, not only at inception. An index whose
universe resolves to no eligible names on its base date raises
`CalculationError`, saying how many identifiers were named, how many
resolved and how many passed the rules.

## Delistings

A name leaves the universe when its reference record ends: the last record
for an identifier has a `DATE_TO`, and no record for it is open-ended. That
date, not a gap in the prices, decides. A missing price is a data-quality
problem, and a name with an ended record is gone.

A delisted name leaves the index in one of two ways:

- **Between rebalances**, it is removed on the first session after its
  `DATE_TO`, and the divisor is rescaled so the level does not move. The
  remaining holdings are unchanged, so their weights grow in proportion,
  which is what reinvesting the proceeds across them would do.
- **At a rebalance**, the point-in-time read no longer returns it, so it is
  not in the new universe.

See [Methodology](methodology.md#treatment) for the divisor arithmetic, and
for the case where the first session after a delisting is itself a
rebalance, which currently loses the leaver's weight.

The sample dataset has no delistings, so this example uses a small dataset
from the synthetic generator.

```python
from beacon.synthetic import SyntheticConfig, generate

synthetic = generate(SyntheticConfig(assets=20, start="2023-01-03",
                                     end="2024-12-31", seed=3))
store = synthetic.fetcher()

print({name: date.strftime("%Y-%m-%d")
       for name, date in sorted(store.delisting_dates().items())})

definition = IndexDefinition(index_id="SYN",
                             index_name="Synthetic",
                             base_date="2024-01-02",
                             base_value=100.0,
                             currency="USD",
                             eligibility_rules=[],
                             weighting_scheme=EqualWeighted(),
                             rebalancing_frequency="QUARTERLY",
                             calendar="XNYS",
                             universe_identifiers=store.reference_identifiers)
result = IndexCalculator(definition, store).run(end_date="2024-12-31")
divisor = result.divisor_history

print([date.strftime("%Y-%m-%d") for date in divisor.index[divisor.diff() != 0]])
```

```text
{'CMPG': '2023-12-14', 'CMPJ': '2024-02-29', 'CMPP': '2024-04-10', 'CMPS': '2023-04-13'}
['2024-01-02', '2024-03-01', '2024-04-01', '2024-04-11', '2024-07-01', '2024-10-01']
```

`CMPG` and `CMPS` were gone before the base date, so they never enter the
index. After it is set on the base date, the divisor moves on each quarterly
rebalance and on the first sessions after `CMPJ` and `CMPP` leave (1 March
and 11 April).

## Stale prices

A name can stay listed and stop trading. `DataFetcher` takes
`max_price_staleness_days`: when it is set, a name whose last bar is more
than that many calendar days before the selection date is dropped before any
eligibility rule runs, so no rule evaluates a stale close. The default,
`None`, keeps every name however long ago it traded. The setting belongs to
the data fetcher rather than to any index, so it applies to every index and
backtest that uses it.

A stale name shows up in the selection funnel as its own rung, at position
`-1` with the name `StalePrice`. A name with no price at all is not stale;
it is a different problem, which the calculation refuses by name.

```python
from beacon.data import DataFetcher, MarketData

frame = dataset.market_frame()
frame = frame[~((frame["IDENTIFIER"] == "EEE")
                & (frame["DATE"] > "2023-11-15"))]

strict = DataFetcher(MarketData.from_dataframe(frame),
                     dataset.reference_data(),
                     max_price_staleness_days=30)
calculator = IndexCalculator(demo_index(list(dataset.UNIVERSE)), strict)
date = pd.Timestamp("2024-01-02")
selection = calculator.select_with_provenance(calculator.resolve_universe(date),
                                              date)

for step in selection.steps:
    print(step.position, step.rule_name or "universe", step.remaining,
          step.excluded)
```

```text
0 universe 6 []
-1 StalePrice 5 ['EEE']
```

## Currencies

A universe can mix currencies. Each resolved asset carries the currency from
its reference record, and everything the index adds up is converted into the
index currency first: holding values, market caps for `MarketCapRule` and
`MarketCapWeighted`, special dividends and reinvested distributions.
`LiquidityRule` is the exception: its volume and traded-value averages stay
in each name's own currency.

Rates come from the data fetcher's FX pairs, stored as market-data
identifiers named `"{FROM}{TO}"` (for example `GBPUSD`). A pair can also be
found as the inverse of the stored reverse pair or as a cross through USD.
Whether a day with no rate uses the last one published or refuses is the
fetcher's `fx_policy` (see [Data](data.md)). When no rate can be found, the
calculation raises `CalculationError` instead of treating the amount as if
it were already in the index currency.

```python
without_fx = DataFetcher(
    MarketData.from_dataframe(
        dataset.market_frame().query("IDENTIFIER != 'GBPUSD'")),
    dataset.reference_data())

try:
    IndexCalculator(demo_index(list(dataset.UNIVERSE)),
                    without_fx).run(end_date="2024-03-28")
except CalculationError as error:
    print(str(error)[:95])
```

```text
Error in calculation 'ConstituentMarketValues': no GBP/USD rate on or before 2024-01-02, so FFF
```

## Filtered universes

Instead of listing identifiers by hand, you can build the list from an
[expression](expressions.md) with `beacon.universe.where`. It evaluates the
expression for each candidate, point in time, and returns the identifiers
that match, in candidate order.

```python
from beacon import universe
from beacon.expressions import data

print(universe.where(data.reference.sector == "Technology", fetcher))
print(universe.where(data.market.market_cap > 80e9, fetcher,
                     date="2024-01-02"))
```

```text
['AAA', 'BBB']
['AAA', 'CCC', 'DDD']
```

`where(expression, fetcher, date=None, identifiers=None, on_missing=False)`:

- `date` defaults to the last date in the loaded data, not today.
- `identifiers` defaults to every instrument with reference data (or every
  market-data identifier when there is no reference data), so FX pairs are
  not offered as members.
- `on_missing=True` includes a name that has no value for a field the
  expression reads; by default it is excluded.
- `data.market.market_cap` is in USD whatever the index currency, so `FFF`
  (about 63bn USD on that date) misses the 80bn cut.

The result is a plain list, so it can go straight into
`universe_identifiers`. That list is then fixed: the index re-resolves its
reference data at every rebalance, but it does not re-run the filter. To keep
the question as well as the answer, `universe.build(expression, fetcher,
date=None, mode="live")` returns a `FilteredUniverse` holding both, with
`as_of` and a `mode`: `"live"` means re-evaluate the filter when read,
`"frozen"` means keep the stored members. `universe.resolve_document()`
answers a stored universe document according to its mode.

See the [universe reference](../reference/universe.md) for every function.
