# Methodology

The methodology layer (`src/beacon/index/methodology.py` and
`constructor.py`) is the rules layer: `IndexDefinition` holds the static
configuration, and two families of pluggable rules — eligibility rules and
weighting schemes — decide, on any given date, which assets are in the index
and at what weight. `IndexCalculator` (the Calculator layer) is what
actually invokes these rules as it walks business days; see
[Backtest](backtest.md) for what happens downstream of the weights it
produces.

A methodology has four concerns, covered in turn below: **Selection**,
**Weighting**, **Scheduling** (when the composition changes), and
**Treatment** (of corporate actions and divisor continuity).

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
universe member through every configured rule and returns the assets that
pass all of them. A rule that raises an exception is treated as a failure for
that asset (logged as an error), not propagated.

`select_with_provenance()` answers the harder question: *why* a name was
excluded. It walks the same rules once and records, per rule, which assets it
removed — so an editor can show the waterfall from universe to constituents
rather than only its two ends. Exclusions are attributed to the **first** rule
that dropped a name, which is what makes the counts sum: a name failing three
rules is excluded once, not three times.

`GET /indices/rule-types` publishes every registered rule with its parameters,
types, defaults and labels, derived from the classes themselves. A rule ships
with its catalogue entry by construction, so an editor cannot fall behind the
library.

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

### Capping

`IndexDefinition.max_constituent_weight` caps any single name, as a fraction.
`IndexCalculator.cap_weights()` applies it after the scheme and **iterates**:
weight pushed off a breaching name can carry another over the cap, so the pass
repeats until nothing breaches.

That iteration has a consequence worth knowing before you render it. A capped
name's *raw* weight is not necessarily above the cap — a name pulled over by
the first round's redistribution is capped in the second, and its pre-cap
weight sits below the cap it ends at. What holds is that a capped name ends
*at* the cap, and that the weight given up sums to `redistributed`.

Every rebalance keeps its `uncapped_weights`, so the counterfactual — what the
index would have held without the cap — is stored rather than estimated. That
is what makes cap drag computable in [attribution](attribution.md).

An infeasible cap is arithmetic, not a solver outcome: 5% across ten names
distributes at most 50%, and the server reports it as a validation finding
while a user is still editing.

## Scheduling

`rebalancing_frequency` (MONTHLY, QUARTERLY, SEMI-ANNUAL, ANNUAL) is the
cadence. Two further fields decide the rest.

**`rebalance_day_rule`** picks the day within a scheduled month:
`FIRST_BUSINESS_DAY` (the default, and what every index defined before this
existed used), `LAST_BUSINESS_DAY`, or `THIRD_FRIDAY` — the S&P and FTSE
convention.

**`calendar`** is an exchange MIC such as `XNYS`. Naming one backs the
arithmetic with real holidays; leaving it null means Monday to Friday, which
treats Christmas Day as a trading day. A rebalance landing on a holiday **rolls
back** to the previous session, not forward, which is the convention and also
the only choice that keeps a month-end rebalance inside its month.

Declaring a calendar requires the `calendars` extra. That is deliberately an
error rather than a silent fallback: two installations must not compute
different indices from the same definition.

**`effective_lag_sessions`** separates announcement from effect. Real indices
publish a constituent list before it takes effect, which is what gives tracking
funds time to trade. Selection and weighting happen as of the *announcement*;
only the units and the divisor use the effective date's prices. Snapshots are
keyed by the effective date, since that is when the weights are in force.

**`return_type`** decides whether distributions accumulate: `PRICE` ignores
them, `TOTAL_RETURN` reinvests them across the index by shrinking the divisor,
and `NET_TOTAL_RETURN` does the same after a flat `withholding_tax_rate`. See
[attribution](attribution.md) for what reinvestment does to the level.

`GET /indices/{id}/schedule` computes the next rebalance and the days until it
from the schedule and the calendar — derived rather than stored, since a stored
date silently expires.

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
  adjusts the divisor for a single event it is *handed*. `SPECIAL_DIVIDEND`
  is implemented: the market-value reduction (`dividend_per_share *
  shares_outstanding`, adjusted for free float and FX) rescales the divisor
  the same way a rebalance does. `RIGHTS_ISSUE`, `SPIN_OFF`, `STOCK_DIVIDEND`
  and `MERGER` are recognised but are **stubs** — they log a warning and
  return the divisor unchanged.

!!! warning "`handle_corporate_action` is not called by `run()`"
    It is a public method a caller can invoke, but the run loop does not
    consult the action history through it. So a price index does **not**
    currently adjust for special dividends during a run, despite the method
    existing.

    The one thing that does read the history during a run is total-return
    reinvestment, added later and independently — which is why there is no
    double-counting between the two. Worth knowing before assuming a price
    index handles distributions.

Both mechanisms exist to keep the index level continuous — a change in
composition should not, by itself, cause the index to jump.
