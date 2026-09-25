# Expressions and screens

An expression names a datapoint and says something about it: "sector is
Technology", "market cap above a billion", "P/E below 20". The same expression
works as an index eligibility rule, a universe filter and a backtest screen,
so there is one way to write a condition and it means the same thing
everywhere.

```python
from beacon.expressions import data
```

Always import `data` from `beacon.expressions`. `from beacon import data`
gives you `beacon.data`, the data package, which is a different thing: asking
it for `market`, `reference` or `actions` raises an `AttributeError` that
points back to the right import.

## Fields

`data` has four namespaces, and each attribute inside one is a field:

| Namespace | Fields | Read from |
| --- | --- | --- |
| `data.market` | `open`, `high`, `low`, `close`, `volume`, `shares_outstanding`, `free_float`, and the derived `adv_3m`, `market_cap`, `free_float_market_cap` | Market data |
| `data.reference` | `name`, `sector`, `sub_industry`, `region`, `exchange`, `currency`, `country_listing`, `country_domicile` | Reference data |
| `data.actions` | `type`, `kind`, `value`, `ex_date`, `pay_date`, `status` | Corporate actions |
| `data.features.<type>` | Whatever the loaded features carry, such as `data.features.fundamentals.revenue` | Features, by `TYPE` |

The fields listed are declared, so they autocomplete in Jupyter, IPython and
`dir()`. They are not the only ones allowed: `data.reference.isin` works if the
loaded reference data has an `ISIN` column. Fields are lower case in Python and
upper case in storage (`data.reference.sector` reads `SECTOR`); feature fields
keep their case, because a feature's `FIELD` is whatever was loaded.

Feature fields nest by dataset (`data.features.fundamentals.pe_ratio`, never
`data.features.pe_ratio`), so two datasets that both carry `revenue` stay
distinct.

The derived market fields are computed when they are read, not stored:

- `market_cap` is `CLOSE` times `SHARES_OUTSTANDING`, converted into USD at
  the name's `CURRENCY` rate, so names quoted in different currencies compare
  on size. It is missing when the price, the share count or the rate is.
- `free_float_market_cap` is `market_cap` times `FREE_FLOAT`.
- `adv_3m` is the mean daily `VOLUME` over the trailing three calendar months.

A misspelt namespace fails at once: `data.refrence` raises
`UnknownDatasetError`. A misspelt field inside a namespace does not, because
the namespace is open; [validation](#validation) catches it.

Autocompletion in an editor such as VS Code or PyCharm reads declarations
rather than live objects, so it cannot see feature datasets. To complete them
too, write a stub file from a data store:

```bash
python -m beacon.expressions.stubs --store my-store --out beacon-data.pyi
```

## Comparisons

Compare a field to a value with an operator or the equivalent method:

| Operator | Method | Meaning |
| --- | --- | --- |
| `>`, `>=`, `<`, `<=` | `gt`, `ge`, `lt`, `le` | Order |
| `==`, `!=` | `eq`, `ne` | Equality |
| | `is_in(values)` | One of a list |
| | `between(low, high)` | Inclusive at both ends |

Combine comparisons with `&` (and), `|` (or) and `~` (not). Parenthesise each
comparison, because `&` and `|` bind more tightly than `>` and `==` in Python:

```python
import logging

import pandas as pd

from beacon import universe
from beacon.data import DataFetcher
from beacon.data.features import FeatureData
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

large_tech = (data.reference.sector == "Technology") & (data.market.market_cap > 1e11)
not_utilities = ~(data.reference.sector == "Utilities")
mid_price = data.market.close.between(50, 150) | data.reference.currency.is_in(["GBP"])

print(large_tech)
# ((reference.sector eq 'Technology') & (market.market_cap gt 100000000000.0))
```

`and`, `or`, `not`, `if` and `assert` do not work on an expression, because
it has no truth value until it is resolved for a name on a date. Rather than
quietly dropping half the condition, they raise `ExpressionError`, saying to
use `&` and `|`.

For the same reason `==` builds a comparison rather than answering whether two
fields are the same. Use `field.same_as(other)` for that, and `field.key` (a
plain tuple) when a field must be a dictionary key or set member.

An expression is a tree, not a value. `to_dict()` turns it into plain JSON
and `from_dict()` rebuilds it exactly, which is how an expression is saved in
an index definition or a universe. `fields_in(expression)` lists the fields it
reads.

```python
from beacon.expressions import fields_in, from_dict

stored = large_tech.to_dict()
print(from_dict(stored))                          # the same expression again
print([field.path for field in fields_in(large_tech)])
# ['reference.sector', 'market.market_cap']
```

## How an expression is answered

An expression is answered for one name on one date, and every read is point
in time: it uses only what was known on that date.

- **Market fields** take the last value on or before the date, looking back
  at most 10 days, so a date the name did not trade still sees its latest
  price.
- **Reference fields** use the reference row in force on the date, so a name
  that changed sector in June was in its old sector in March.
- **Feature fields** use the latest value dated on or before the date. A
  value dated D is invisible before D, however long ago the period it
  describes ended. A value more than 730 days old counts as missing; index
  rules take a `max_age_days` argument to change that.

A name with no value for a field is **missing**, which is not the same as
zero. Zero fails `> 0` honestly; missing has nothing to compare. Each use of
an expression has an `on_missing` setting that decides what a comparison on a
missing value answers, and it excludes the name by default. That way a screen
for "revenue above a billion" does not admit every name the dataset has never
heard of.

!!! warning "Corporate-action fields"
    `data.actions` fields pass validation and appear in the field catalogue,
    but a screen cannot read them yet: every name is missing a value, so the
    comparison answers whatever `on_missing` says.

Here the sample dataset gets a small feature table. A P/E published on
20 February is invisible to a screen on 1 February:

```python
features = FeatureData.from_dataframe(pd.DataFrame({
    "IDENTIFIER": ["CCC", "AAA", "BBB", "FFF", "AAA"],
    "DATE": ["2023-11-10", "2024-02-15", "2024-02-20", "2024-03-20", "2024-05-15"],
    "TYPE": "fundamentals",
    "FIELD": "pe_ratio",
    "VALUE": [12.0, 28.0, 16.0, 9.0, 18.0],
}))
fetcher = DataFetcher(dataset.market_data(), dataset.reference_data(),
                      features=features)

cheap = data.features.fundamentals.pe_ratio < 20

print(universe.where(cheap, fetcher, "2024-02-01"))   # ['CCC']
print(universe.where(cheap, fetcher, "2024-03-01"))   # ['BBB', 'CCC']
print(universe.where(cheap, fetcher, "2024-06-03"))   # ['AAA', 'BBB', 'CCC', 'FFF']
```

AAA's first P/E (28, published 15 February) fails the screen; its second (18,
published 15 May) passes, so AAA joins only in June.

## Where expressions are used

### Filtered universes

`beacon.universe.where(expression, fetcher, date=None, identifiers=None,
on_missing=False)` returns the names that pass on a date, in the order the
candidates were given. With no date it answers as of the last date in the
loaded data (not today). With no candidates it tests every name in the
reference data, so FX pairs are never offered as members.

`universe.build` resolves an expression and keeps the expression beside the
answer, with a `mode`: `LIVE` means the saved universe is the question, and
re-evaluates when it is read; `FROZEN` means it is the list of names, fixed as
of the date it was built. `resolve_document` reads a saved universe either
way. See [Universe](universe.md).

```python
frozen = universe.build(cheap, fetcher, "2024-04-01", mode=universe.FROZEN)
print(frozen.identifiers)   # ['BBB', 'CCC', 'FFF']
print(universe.resolve_document(frozen.as_document(), fetcher, "2024-06-03"))
# ['BBB', 'CCC', 'FFF']: a frozen universe keeps its names
```

### Index rules

`ExpressionRule` makes an expression an eligibility rule in an index
[methodology](methodology.md). It is evaluated at every rebalance, through the
same point-in-time reads, so the index can only select on what was known on
the rebalance date.

```python
from beacon.index import EqualWeighted, ExpressionRule, IndexCalculator, IndexDefinition

rule = ExpressionRule.from_expression(
    cheap & (data.reference.sector != "Financials"))

definition = IndexDefinition(
    index_id="VALUE", index_name="Cheap, excluding financials",
    base_date="2024-01-02", base_value=1000.0, currency="USD",
    eligibility_rules=[rule], weighting_scheme=EqualWeighted(),
    rebalancing_frequency="MONTHLY", calendar="XNYS",
    universe_identifiers=list(dataset.UNIVERSE),
)
result = IndexCalculator(definition, fetcher).run(end_date="2024-06-28")

for date, names in result.constituent_snapshots.items():
    print(date.date(), names)
# 2024-01-02 ['CCC']
# 2024-02-01 ['CCC']
# 2024-03-01 ['BBB', 'CCC']
# 2024-04-01 ['BBB', 'CCC']
# 2024-05-01 ['BBB', 'CCC']
# 2024-06-03 ['AAA', 'BBB', 'CCC']
```

`ExpressionRule.from_expression(expression, on_missing="exclude",
max_age_days=730)` stores the expression's `to_dict()` form as the rule's
parameters, so a rule written in Python and one built in a client are the
same document. `on_missing` is `"exclude"` or `"include"`. The rule tells the
index which market columns it needs (a `market_cap` screen needs `CLOSE` and
`SHARES_OUTSTANDING`), so a dataset without one is refused with a
`CalculationError` naming it before the run does any work.

`FeatureRule(field, comparison="gt", threshold=0.0, feature_type=None,
on_missing="exclude", max_age_days=730)` is the simpler form for one feature
against one threshold, such as `FeatureRule("pe_ratio", "lt", 20,
feature_type="fundamentals")`. With no `feature_type` it searches every
dataset, and picks arbitrarily between two that carry the same field name.

### Backtest screens

`ExpressionScreen(expression, fetcher, on_missing=False)`, from
`beacon.backtest.rules`, applies an expression inside a backtest. Pass it in
the engine's `modifiers`. The target weights still come from the index; the
screen removes what should not be held. At every rebalance it resolves the
expression for each name being traded or held, as of the rebalance date:

- a buy of a name that fails is dropped, and its weight stays in cash;
- a name held that fails is sold, at the price the portfolio was last marked
  at, with no transaction cost;
- if the rebalance already sells part of a failing name, that sell stands and
  no further exit is added;
- a failing name with no price to sell at is kept, with a warning.

```python
from beacon.backtest import BacktestEngine
from beacon.backtest.rules import ExpressionScreen

screen = ExpressionScreen(data.market.close < 150, fetcher)

backtest = BacktestEngine(start_date="2024-01-02", end_date="2024-06-28",
                          initial_capital=1_000_000.0, data_provider=fetcher,
                          index_result=result, calendar="XNYS",
                          modifiers=[screen]).run()

print(sorted(backtest.portfolio.holdings))   # ['BBB', 'CCC']: AAA closed above 150 in June
```

See [Backtest](backtest.md) for the engine and its other modifiers.

## Validation

Nothing checks an expression before it runs: `universe.where`,
`ExpressionRule` and `ExpressionScreen` resolve whatever they are given, and a
misspelt field is simply missing for every name, so the screen selects nothing
(or everything, with `on_missing` set to include). Check an expression against
the data first with `beacon.expressions.validation`:

- `validate(expression, fetcher)` returns every problem as a list of
  `Finding`s, each with the field's `path`, a stable `code` and a `message`.
  An empty list means the expression is valid.
- `errors_in(expression, fetcher)` returns only the findings that block, and
  `is_valid(expression, fetcher)` is true when there are none.

The codes are `UNKNOWN_FIELD` (the namespace or feature dataset has no such
field), `UNKNOWN_FEATURE_TYPE` (no such feature dataset, or no features loaded
at all) and `UNKNOWN_NAMESPACE`. A message suggests up to three close matches
from what is loaded, or lists a few available names when nothing is close.

```python
from beacon.expressions.validation import is_valid, validate

typo = (data.reference.sectr == "Energy") & (data.features.fundamentals.pe_ratoi < 15)

for finding in validate(typo, fetcher):
    print(finding.code, finding.path, finding.message)
# UNKNOWN_FIELD reference.sectr 'sectr' is not in reference. Did you mean 'sector'?
# UNKNOWN_FIELD features.fundamentals.pe_ratoi 'pe_ratoi' is not in the 'fundamentals' dataset. Did you mean 'pe_ratio'?

print(is_valid(cheap, fetcher))   # True
```

The loaded data decides what exists, not the declared field list: a reference
column you loaded yourself is valid, and a declared column your data lacks is
not. The derived market fields are always valid, since they are computed, and
action fields are checked against the declared list.

## The field catalogue

`beacon.expressions.catalogue.describe_fields(fetcher)` lists every field an
expression can name in the loaded data, for building a field picker. Each
entry gives the field's `path`, `namespace`, `name`, feature `dataset`, and
whether it is `derived`. It lists the market and reference columns actually
loaded (leaving out keys such as `IDENTIFIER` and `DATE_FROM`), the derived
market fields, the declared action fields, and every loaded feature field by
dataset. The engine serves the same list at `GET /data/fields`.

```python
from beacon.expressions.catalogue import describe_fields

fields = describe_fields(fetcher)
print([entry["path"] for entry in fields if entry["namespace"] == "reference"])
# ['reference.name', 'reference.currency', 'reference.exchange', 'reference.sector']
print([entry["path"] for entry in fields if entry["derived"]])
# ['market.adv_3m', 'market.market_cap', 'market.free_float_market_cap']
```

## See also

- [Data](data.md): the datasets expressions read, and point-in-time storage.
- Reference: [Expressions](../reference/expressions.md),
  [Universes](../reference/universe.md), [Indices](../reference/indices.md),
  [Backtest](../reference/backtest.md).
