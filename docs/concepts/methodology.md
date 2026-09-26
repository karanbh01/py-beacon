# Methodology

An index methodology is an `IndexDefinition`: the rules that decide which
names the index holds (selection), in what proportions (weighting and
capping), when the composition changes (scheduling), what return it measures
(return type), and how the level stays continuous while all of that happens
(treatment). `IndexCalculator` applies the definition to a `DataFetcher` and
returns an `IndexResult`. The [backtest](backtest.md) then trades those
weights.

The examples on this page run on the frozen sample dataset in
`beacon.testing.dataset`: six names, five quoted in USD and `FFF` in GBP,
with a `GBPUSD` pair, on the XNYS calendar from January 2023 to December
2025.

```python
import pandas as pd

from beacon.index import (IndexCalculator, IndexDefinition, LiquidityRule,
                          MarketCapRule, MarketCapWeighted)
from beacon.testing import dataset

fetcher = dataset.data_fetcher()

definition = IndexDefinition(
    index_id="DEMO",
    index_name="Demo Cap-Weighted",
    base_date="2024-01-02",
    base_value=1000.0,
    currency="USD",
    eligibility_rules=[MarketCapRule(min_market_cap=60e9),
                       LiquidityRule(min_avg_daily_value=60e6)],
    weighting_scheme=MarketCapWeighted(use_free_float=True),
    rebalancing_frequency="QUARTERLY",
    calendar="XNYS",
    universe_identifiers=list(dataset.UNIVERSE),
    max_constituent_weight=0.35,
)
calculator = IndexCalculator(definition, fetcher)
```

`IndexDefinition` validates its own fields when it is built: the base value
must be positive, the calendar, currency and frequency must be given, a
universe list must not be empty, and the cap, day rule, return type,
withholding rate and announcement lag must be in range. An empty rule list is
allowed (it logs a warning).

## Running a definition

`IndexCalculator.run(start_date=None, end_date=...)` calculates the index.
Before it does any work it checks two things:

- **The data has every column the definition reads.** Each rule and scheme
  declares its market-data columns (`required_columns()`), and the run checks
  them, plus the price column it values holdings with (`CLOSE` unless
  `IndexCalculator(price_column=...)` says otherwise), against the dataset. A
  missing column raises a `CalculationError` naming the column and what needs
  it. The check is skipped when the data provider cannot list its columns.
- **The calendar covers the window.** A window wholly outside the range of
  dates the calendar knows raises. A window it covers only partly is
  calculated over the covered part, and `result.calendar_coverage` says what
  was asked for and what was covered. A window inside the calendar's range
  that holds no session (a single weekend) gives an empty result.

The run then walks the sessions of the index's own calendar, not Monday to
Friday, from the base date to `end_date`. A base date that is not a session
(1 January, say) moves forward to the first session after it. On the base
date and on each rebalance it resolves the [universe](universe.md), selects,
weights and caps; on every other session it values the holdings and sets the
level.

!!! note "Share counts are always read"
    Whatever the weighting, the calculation sizes its holdings from the
    constituents' total market value (price times `SHARES_OUTSTANDING`) on
    the base date and at each rebalance. An index over data with no share
    counts therefore fails on its base date, even with `EqualWeighted`, which
    declares no columns of its own.

## Selection

Selection narrows the universe to the names eligible on a date. Before any
rule runs, names that have not traded recently enough are dropped, if the
data fetcher sets `max_price_staleness_days` (see
[Universe](universe.md#stale-prices)). Then the rules in
`eligibility_rules` run in order, and each rule sees only the names that
passed the ones before it. A name must pass every rule.

Four rules ship with py-beacon:

| Rule | Screens on | Columns it needs |
|---|---|---|
| `MarketCapRule(min_market_cap, max_market_cap)` | price times shares outstanding, in the index currency | `CLOSE`, `SHARES_OUTSTANDING` |
| `LiquidityRule(min_avg_daily_volume, min_avg_daily_value, lookback_days=60)` | average daily volume and/or traded value | `VOLUME`, plus `CLOSE` for a value floor |
| `FeatureRule(field, comparison="gt", threshold=0.0, ...)` | one feature value against a threshold | none (features live in their own table) |
| `ExpressionRule.from_expression(expression, ...)` | any [expression](expressions.md) | whatever market columns the expression names |

**`MarketCapRule`** reads the price and share count on one session: the last
session on or before the date that the data has. A name with no price or no
positive share count on that session is excluded, with a warning. The cap is
converted into the index currency at that session's rate before it meets the
bounds, so the bounds are always in the index currency. Either bound may be
`None`; a minimum above the maximum raises `ValueError`.

**`LiquidityRule`** averages over the last `lookback_days` rows on or before
the date. A name with fewer than 80% of that many rows, or with the column it
needs empty, is excluded, with a warning. Its values are in the currency the
name trades in and are not converted.

**`FeatureRule`** reads a feature such as `pe_ratio` from the data's feature
table, point in time: only a value published on or before the date counts,
and one older than `max_age_days` (default 730) counts as missing.
`comparison` is one of `"gt"`, `"ge"`, `"lt"`, `"le"`, `"eq"` or `"ne"`.
`feature_type` names the dataset to read from; left as `None`, it searches
all of them. A name with no value is excluded unless `on_missing="include"`.

**`ExpressionRule`** screens on an [expression](expressions.md), resolved
point in time, and stores the expression as a serialisable tree, so a rule
written in Python and one built in the app are the same document. A malformed
tree is refused when the rule is built (`InvalidRuleError`). Missing values
and stale features work as they do for `FeatureRule`. The derived field
`data.market.market_cap` is always in USD, whatever the index currency, while
`MarketCapRule` compares in the index currency.

`FeatureRule` and `ExpressionRule` need data with features, so they are
shown under [Return types](#return-types) on a synthetic dataset. Here is the
funnel for the definition above, on the base date:

```python
date = pd.Timestamp("2024-01-02")
selection = calculator.select_with_provenance(calculator.resolve_universe(date),
                                              date)

for step in selection.steps:
    print(step.position, step.rule_name or "universe", step.remaining,
          step.excluded)
```

```text
0 universe 6 []
1 MarketCapRule 5 ['EEE']
2 LiquidityRule 4 ['CCC']
```

`FFF` passes the 60bn floor although its cap is about 46bn in GBP: converted
at that day's `GBPUSD` rate it is about 63bn USD. `EEE` (about 52bn) does
not.

`select_with_provenance()` records, for each rung, which names it removed.
Each excluded name is attributed to the first rule that removed it, so the
counts add up: a name that would fail three rules is excluded once.
`selection.excluded_by("EEE")` returns the rung that removed a name. The
stale-price rung, when present, has position `-1` and the name `StalePrice`.
`select_constituents()` runs the same walk and returns only the survivors.

**Excluded is not the same as failed.** A rule that cannot evaluate raises,
and the error reaches the caller instead of being recorded as an exclusion.
`MarketCapRule` raises for a date outside the data's coverage and for a
missing FX pair. `MarketCapRule` and `LiquidityRule` raise `CalculationError`
for an asset that is not an `Equity`. (The calculator builds every universe
member as an `Equity`, so this only matters when you pass assets of your
own.)

A rule of your own subclasses `EligibilityRuleBase` and implements
`is_eligible(asset, current_date, market_data_provider, context=None)`,
returning `True` or `False`. `context` is an `IndexContext` carrying the
index currency. Override `required_columns()` so the up-front column check
covers it; a rule that declares nothing fails at its first read instead.

## Weighting

The weighting scheme turns the selected names into weights. Two ship with
py-beacon:

- **`EqualWeighted()`** gives each of `n` constituents `1 / n`. It reads no
  market data.
- **`MarketCapWeighted(use_free_float=False)`** weights each name by its
  market cap (price times shares outstanding, times the free-float factor
  when `use_free_float=True`) over the sum of all of them. Each name is
  priced at its last close on or before the rebalance session, and its
  shares, free float and FX rate are read on that same day. A universe quoted
  in one currency needs no conversion; one spanning currencies is converted
  into the index currency first.

`MarketCapWeighted` has no fallback. It raises `CalculationError` if the date
lies outside the data's coverage, if a constituent has no close at all, no
positive share count, no usable free float (when float-adjusted) or no FX
pair, if a constituent is not an `Equity`, or if the caps sum to zero.

The free float in force on a date is the last value reported on or before
it, if that value is no more than `free_float_backfill_days` calendar days
old. That setting lives on the `DataFetcher` (default 90; 0 uses only a
value dated that day). A value dated after the date is never used, and a
value outside 0 to 1 is refused. The same rule applies wherever a free float
is read. See [Data](data.md).

`calculate_constituent_weights()` calls the scheme and checks that the
weights sum to 1 within 1e-9. A scheme of your own subclasses
`WeightingSchemeBase` and implements `calculate_weights()`; weights that do
not sum to 1 raise `CalculationError` rather than being rescaled. A scheme
that raises anything other than `CalculationError` is reported as an
`UnexpectedCalculationError`.

```python
weights = calculator.calculate_constituent_weights(selection.survivors, date)
print({asset.asset_id: round(weight, 4) for asset, weight in weights.items()})
```

```text
{'AAA': 0.4121, 'BBB': 0.2122, 'DDD': 0.2079, 'FFF': 0.1678}
```

### Capping

`max_constituent_weight` caps any single name, as a fraction in (0, 1].
`IndexCalculator.cap_weights()` applies it after the scheme, so it works with
every scheme. The weight taken off a capped name is spread over the uncapped
names in proportion to their weights, and because that can push another name
over the cap, the pass repeats until nothing breaches.

```python
capped, report = calculator.cap_weights(weights)
print({asset.asset_id: round(weight, 4) for asset, weight in capped.items()})
print(report.capped, round(report.redistributed, 4), report.passes)
```

```text
{'AAA': 0.35, 'BBB': 0.2346, 'DDD': 0.2298, 'FFF': 0.1855}
{'AAA': 0.4120794401670447} 0.0621 1
```

The `CapReport` maps each capped name to its weight before capping,
`redistributed` is the total weight moved, and `passes` counts the
iterations. A capped name's weight before capping is not always above the
cap: a name pushed over in the first pass is capped in the second, and its
original weight sits below the cap it ends at. What always holds is that a
capped name ends at the cap, and that the weight given up sums to
`redistributed`.

`result.cap_reports` holds a report only for the rebalances where the cap
bound, keyed by date. Each report keeps `uncapped_weights`, the full weight
vector before capping, so the counterfactual (what the index would have held
without the cap) is stored rather than estimated. On a rebalance with no
report, the stored weights are already the uncapped ones. This is what makes
cap drag computable in [attribution](attribution.md).

A cap is infeasible when `cap * n < 1`: 15% across six names can distribute
at most 90%. The run raises `CalculationError` at the first rebalance where
that happens, and the server reports it as a validation finding while a
definition is still being edited.

```python
from beacon.exceptions import CalculationError
from beacon.index import EqualWeighted

too_tight = IndexDefinition(
    index_id="TIGHT",
    index_name="Too Tight",
    base_date="2024-01-02",
    base_value=1000.0,
    currency="USD",
    eligibility_rules=[],
    weighting_scheme=EqualWeighted(),
    rebalancing_frequency="QUARTERLY",
    calendar="XNYS",
    universe_identifiers=list(dataset.UNIVERSE),
    max_constituent_weight=0.15,
)

try:
    IndexCalculator(too_tight, fetcher).run(end_date="2024-03-28")
except CalculationError as error:
    print(error)
```

```text
Error in calculation 'WeightCapping': a cap of 15.0000% cannot be satisfied by 6 constituents: the total would reach at most 90.0000%. The smallest feasible cap here is 16.6667%.
```

## Scheduling

**`calendar`** is an exchange MIC such as `XNYS`, from the
`exchange_calendars` package, and every `IndexDefinition` needs one: there is
no default, because a default would schedule a European index on New York's
holidays without saying so. The calendar decides which days the index has a
level on and which days a rebalance can fall on. `GET /indices/calendars`
lists every MIC a server accepts, with its name, timezone and region. A stored
definition without a calendar is read as `XNYS`.

**`rebalancing_frequency`** is `MONTHLY`, `QUARTERLY`, `SEMI-ANNUAL` or
`ANNUAL`. The cadence is anchored on the first scheduled date in the range,
not on the calendar year: a quarterly index starting in February rebalances
in February, May, August and November.

**`rebalance_day_rule`** picks the day within a scheduled month:

- `FIRST_BUSINESS_DAY` (the default): the month's first session.
- `LAST_BUSINESS_DAY`: the month's last session.
- `THIRD_FRIDAY`: the month's third Friday by the calendar. If that day is
  not a session, the rebalance rolls back to the session before it, which
  keeps it inside its month. The first two rules pick a session directly, so
  they never need to roll.

```python
from beacon.index.schedule import rebalance_dates

for rule in ["FIRST_BUSINESS_DAY", "LAST_BUSINESS_DAY", "THIRD_FRIDAY"]:
    dates = rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31", "XNYS",
                            rule)
    print(rule, [date.strftime("%Y-%m-%d") for date in dates])
```

```text
FIRST_BUSINESS_DAY ['2025-01-02', '2025-04-01', '2025-07-01', '2025-10-01']
LAST_BUSINESS_DAY ['2025-01-31', '2025-04-30', '2025-07-31', '2025-10-31']
THIRD_FRIDAY ['2025-01-17', '2025-04-17', '2025-07-18', '2025-10-17']
```

The third Friday of April 2025 is Good Friday, when NYSE was closed, so that
rebalance falls on Thursday 17 April.

`IndexDefinition.get_rebalance_dates(start, end)` returns the same dates for
a definition, and `next_rebalance(as_of)` returns the first one after a date,
anchored on the base date. `GET /indices/{id}/schedule` serves the next
rebalance and the days until it.

**`effective_lag_sessions`** separates announcement from effect. A real index
publishes its new constituents before they take effect, which gives tracking
funds time to trade. With a lag of `n`, the composition is selected and
weighted as of the scheduled (announcement) date and applied `n` sessions
later, at that day's prices. Snapshots are keyed by the effective date, and
`result.announcement_dates` maps each effective date to its announcement
date (only where the two differ). If the data ends before the lag has
elapsed, the rebalance is applied on its announcement date, with a warning.

```python
result = calculator.run(end_date="2024-12-31")

print(round(result.index_levels.iloc[-1], 2))
print([date.strftime("%Y-%m-%d") for date in result.weight_snapshots])
print([date.strftime("%Y-%m-%d") for date in result.cap_reports])
```

```text
1167.54
['2024-01-02', '2024-04-01', '2024-07-01', '2024-10-01']
['2024-01-02', '2024-04-01', '2024-07-01', '2024-10-01']
```

## Return types

`return_type` decides what happens to cash distributions:

- **`PRICE`** (the default) ignores them. The price drops on the ex-date and
  the level drops with it.
- **`TOTAL_RETURN`** reinvests them across the whole index by shrinking the
  divisor on the ex-date.
- **`NET_TOTAL_RETURN`** does the same after withholding a flat
  `withholding_tax_rate` (a fraction in [0, 1)) from each distribution. The
  rate is ignored for the other two types.

Distributions come from the data's corporate-action history, and only cash
actions count: a split distributes nothing. A distribution paid in another
currency is converted into the index currency first, and a missing FX pair
raises. Reinvestment assumes the price series is unadjusted, so it already
contains the ex-date drop; on a dividend-adjusted series it would count each
distribution twice.

The sample dataset has no dividends, so this example uses a small dataset
from the synthetic generator, which does. It also has features, so it shows
the other two rules.

```python
from beacon.expressions import data
from beacon.index import ExpressionRule, FeatureRule
from beacon.synthetic import SyntheticConfig, generate

synthetic = generate(SyntheticConfig(assets=20, start="2023-01-03",
                                     end="2024-12-31", seed=3))
store = synthetic.fetcher()


def synthetic_index(rules,
                    return_type="PRICE"):
    return IndexDefinition(index_id="SYN",
                           index_name="Synthetic",
                           base_date="2024-01-02",
                           base_value=100.0,
                           currency="USD",
                           eligibility_rules=rules,
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="QUARTERLY",
                           calendar="XNYS",
                           universe_identifiers=store.reference_identifiers,
                           return_type=return_type,
                           withholding_tax_rate=0.15)


rules = [FeatureRule("pe_ratio", comparison="lt", threshold=25.0,
                     feature_type="fundamentals"),
         ExpressionRule.from_expression(data.market.market_cap > 1e9)]
screened = IndexCalculator(synthetic_index(rules), store)
selection = screened.select_with_provenance(screened.resolve_universe(date),
                                            date)

for step in selection.steps:
    print(step.position, step.rule_name or "universe", step.remaining,
          step.excluded)

for return_type in ["PRICE", "TOTAL_RETURN", "NET_TOTAL_RETURN"]:
    levels = IndexCalculator(synthetic_index([], return_type),
                             store).run(end_date="2024-12-31").index_levels
    print(return_type, round(levels.iloc[-1], 2))
```

```text
0 universe 18 []
1 FeatureRule 12 ['CMPC', 'CMPD', 'CMPF', 'CMPH', 'CMPK', 'CMPN']
2 ExpressionRule 7 ['CMPB', 'CMPE', 'CMPL', 'CMPM', 'CMPP']
PRICE 106.27
TOTAL_RETURN 107.49
NET_TOTAL_RETURN 107.3
```

(The universe starts at 18 of 20 names because two were delisted before the
base date; see [Universe](universe.md#delistings).)

## Treatment

The **divisor** converts the value of the index's holdings into a level:
`level = aggregate value / divisor`. On the base date it is set to the
constituents' total market value (in the index currency) divided by
`base_value`, so the level starts at `base_value`. Between rebalances the
index holds a fixed number of units of each name, so weights drift with
prices.

Whenever the holdings change for a reason other than a price move, the
divisor is rescaled so the level is the same immediately before and after.
Four things trigger that:

- **A rebalance.** The outgoing holdings and the new ones are both valued at
  the day's prices, and
  `new_divisor = old_divisor * new_value / old_value`
  (`IndexCalculator.adjust_divisor_for_rebalance`).
- **A delisting.** A holding whose reference record ends (`DATE_TO`) is
  removed on the first session after that date, and the divisor is rescaled
  by the book's value without it over its value with it, both on the last
  day it had a price. The survivors' weights grow in proportion. On a
  rebalance day this happens before the outgoing holdings are valued. See
  [Universe](universe.md#delistings).
- **A cash distribution**, for `TOTAL_RETURN` and `NET_TOTAL_RETURN` indices:
  `new_divisor = old_divisor * A / (A + D)`, where `A` is the holdings' value
  on the ex-date and `D` the cash received, net of withholding. On a
  rebalance date the reinvestment happens first, into the outgoing holdings.
- **A corporate action passed to `handle_corporate_action()`.** Only
  `SPECIAL_DIVIDEND` is implemented: the market-value reduction (dividend per
  share times shares outstanding, times free float for a float-adjusted
  index, converted at the ex-date rate) rescales the divisor. `RIGHTS_ISSUE`,
  `SPIN_OFF`, `STOCK_DIVIDEND` and `MERGER` are recognised but not
  implemented by this method, and they raise `CalculationError`, as does an
  unknown type or an action missing its ex-date, asset or value. An action on
  a name the index does not hold leaves the divisor unchanged.

Splits, reverse splits and stock dividends do not move the divisor. On the
ex-date the stored close moves by the ratio, and the units the index holds are
multiplied by the same ratio (the action's `VALUE`: 2.0 for a 2-for-1 split),
so the level is unchanged. A cancelled action is ignored.

!!! warning "`run()` does not call `handle_corporate_action()`"
    It is a public method you can call, but the calculation loop does not
    apply the action history through it. A `PRICE` index therefore does not
    adjust for special dividends during a run. The only use a run makes of
    the action history is total-return reinvestment, so a special dividend is
    never counted twice.

Two data gaps do not move the divisor:

- A holding with no price on a session is valued at zero for that session,
  with a warning, so the level dips by its share and recovers when the price
  returns. Only if every holding is unpriced is the previous level carried
  forward.
- If every holding would be delisted at once, the holdings are kept and an
  error is logged, rather than emptying the index. If the book cannot be
  valued on the leaver's last day, the leaver is removed without a divisor
  adjustment and the level steps, with a warning.

See the [indices reference](../reference/indices.md) for every class and
parameter on this page.
