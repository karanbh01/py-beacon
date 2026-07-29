# Methodology

The methodology layer (`src/beacon/index/methodology.py` and
`constructor.py`) is the rules layer: `IndexDefinition` holds the static
configuration, and two families of pluggable rules — eligibility rules and
weighting schemes — decide, on any given date, which assets are in the index
and at what weight. `IndexCalculator` (the Calculator layer) is what
actually invokes these rules as it walks business days; see
[Backtest](backtest.md) for what happens downstream of the weights it
produces.

A methodology has three concerns, covered in turn below: **Selection**,
**Weighting**, and **Treatment** (of corporate actions and divisor
continuity).

## Selection

Selection narrows the [universe](universe.md) down to the assets actually
eligible for inclusion, via `EligibilityRuleBase` subclasses.

`EligibilityRuleBase.is_eligible(asset, current_date, market_data_provider,
context=None)` is the abstract contract every rule implements, returning a
plain `bool`. `IndexDefinition.eligibility_rules` holds a list of these; an
asset must pass **every** rule to be selected.

Two concrete rules ship today:

- **`MarketCapRule(min_market_cap=None, max_market_cap=None)`** — computes
  `price * shares_outstanding` for the asset on `current_date` and checks it
  against the configured bounds. Non-`Equity` assets pass through
  unconditionally.
- **`LiquidityRule(min_avg_daily_volume=None, min_avg_daily_value=None,
  lookback_days=60)`** — looks back over a trailing window of prices/volume
  and checks average daily volume and/or average daily traded value.
  Requires at least 80% of the lookback window's trading days to be present
  before it will pass an asset.

`IndexCalculator.select_constituents(universe, current_date)` runs each
universe member through every configured rule (via the internal
`_passes_all_rules` helper) and returns the assets that pass all of them. A
rule that raises an exception is treated as a failure for that asset (logged
as an error), not propagated.

## Weighting

Weighting assigns a float weight to each selected constituent, via
`WeightingSchemeBase` subclasses.

`WeightingSchemeBase.calculate_weights(constituents, current_date,
market_data_provider, context=None)` returns a `dict[Asset, float]` whose
values should sum to 1.0. Two schemes ship today:

- **`EqualWeighted()`** — `1 / n` for each of the `n` constituents.
- **`MarketCapWeighted(use_free_float=False)`** — each constituent's weight
  is its market cap (`price * shares_outstanding`, optionally scaled by a
  free-float factor from `DataFetcher.fetch_free_float_factor`) divided by
  the total market cap of all constituents. If every constituent's market
  cap comes back zero, it falls back to equal weighting.

`IndexCalculator.calculate_constituent_weights(constituents, current_date)`
calls the configured scheme and then **normalises** the result to sum to
1.0 if it does not already (within a small tolerance), logging a warning
when it has to.

!!! note "No weight capping"
    Beacon does not currently enforce any maximum single-name or sector
    weight. `calculate_weights()` returns whatever the scheme computes
    (normalised to sum to 1.0); there is no post-processing step that caps
    and redistributes weight. If a methodology needs capping, it has to be
    built into a custom `WeightingSchemeBase` subclass today.

## Treatment

Treatment covers what happens to the index's **divisor** — the value that
converts raw aggregate market value into an index level — when composition
or corporate actions change the underlying market value without a real
change in the economic value of the index.

Two situations trigger a divisor adjustment, both implemented on
`IndexCalculator` (mixed in from `market_values.py` and
`corporate_actions.py`):

- **Rebalance** — `IndexCalculator.adjust_divisor_for_rebalance(old_divisor,
  old_market_value, new_market_value)` rescales the divisor so that the
  index level is identical immediately before and after a reconstitution:
  `new_divisor = old_divisor * (new_market_value / old_market_value)`.
- **Corporate actions** — `CorporateActionsMixin.handle_corporate_action(...)`
  adjusts the divisor for a single event. **`SPECIAL_DIVIDEND`** is fully
  implemented: the market-value reduction
  (`dividend_per_share * shares_outstanding`, adjusted for free float and
  FX) is used to rescale the divisor the same way a rebalance does. Four
  other action types — `RIGHTS_ISSUE`, `SPIN_OFF`, `STOCK_DIVIDEND`, and
  `MERGER` — are recognised but are **stubs**: calling
  `handle_corporate_action` with one of these types logs a warning and
  returns the divisor unchanged.

Both mechanisms exist to keep the index level continuous — a change in
composition or a dividend payment should not, by itself, cause the index to
jump.
