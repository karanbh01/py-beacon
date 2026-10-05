# 6. Backtest implementation, AUM, vehicles and strategies

Date: 2026-09-29

## Status

Accepted. Built in phases, tracked as BN-263 to BN-272 (#276 to #285), and
BN-276 (#289) for modelling assumptions, added 2026-10-02. Amended 2026-10-05:
flows are a fifth part of a run rather than a vehicle setting, the cash buffer
belongs to the implementation, and dividends are reinvested by default.

## Context

A backtest today trades a fixed amount of capital to an index's weights, at a
flat cost in basis points. Three things it cannot express came up together:

- **Implementation limits that depend on size.** A broad index run at 100
  million and at 100 billion should differ: the larger fund cannot own much of
  a small company or trade it quickly. The only hook edits the finished trade
  list, so a screen cannot even say "do not hold this name" cleanly (#268).
- **Money moving in and out.** No flows, no units, no fund accounting.
- **What kind of fund it is.** `IndexFund` and `ETF` are thin wrappers (a fee;
  a market price equal to NAV) and cannot grow into OEICs, SICAVs, ETFs dealt
  in kind, or active funds without a subclass per combination.

The engine also receives no dividends, so a backtest trails a total-return
index by about the dividend yield.

## Decision

A backtest run has five independent parts:

1. **Strategy**: what is held. An index definition (tracked in full), an
   index with a replication method, or an active strategy. Always passed to
   `run()`.
2. **Implementation**: screens, capacity caps, costs, execution limits, the
   cash buffer and how flows are invested. It applies to every vehicle, since
   a plain backtest at 100 billion needs capacity limits too.
3. **Flows**: the money arriving and leaving, as a scenario. None for a fixed
   amount of capital.
4. **Vehicle**: how the money is held: fees, and the rules a structure applies
   to flows. None for a plain backtest, or a `Vehicle` built from a structure
   preset.
5. **Modelling assumptions**: how the simulation models markets and data,
   whatever the strategy, size or vehicle. A `ModellingAssumptions` object.

```python
backtest = Backtest(modelling_assumptions=ModellingAssumptions(...),
                    implementation=..., flows=..., vehicle=etf,
                    data_provider=data)

backtest.run(my_index, start, end)                     # full physical
backtest.run(IndexTracking(my_index, replication="optimised"), start, end)
backtest.run(ActiveStrategy(...), start, end)
```

`run()` has one form with or without a vehicle.

Keeping them apart is what allows a fair comparison: the same index, the same
limits and the same flows, only the vehicle changed (an index fund against an
ETF, say).

Flows are not a modelling assumption. The assumptions describe how markets and
data behave and have a process-wide default; flows are a scenario chosen per
run, and one default schedule for every backtest would mean nothing.

### Modelling assumptions

What the simulation takes as given about the world, gathered in one object
rather than spread across data-source settings, module constants and fixed
engine behaviour:

- **Data treatment**, shared with the index calculation so the index and the
  backtest never assume different things: the FX policy on a day with no
  rate, the stale-price threshold, the free-float backfill window, how old a
  feature value may be, and a missing bar carried at its last close.
- **Simulation conventions**: when and at what price trades happen (today the
  rebalance day's close, sells before buys), what cash earns (today
  nothing), the risk-free rate for Sharpe (today 0), periods per year (252),
  how a delisting settles, and dividend timing and withholding (#276).

The defaults reproduce today's behaviour exactly, so adding the object
changes no result. `ModellingAssumptions` owns the data-treatment settings;
a data source's own settings remain the fallback, so existing code keeps
working. A process-wide default can be set once, as `beacon.use()` sets the
data source, and a `Backtest` can override it. Every result records the
assumptions it was produced under, and the index cache keys on the ones that
affect a calculation.

### The backtest as stages

Each stage has its own hook, in this order:

1. **Target weights**, from the strategy (an index stays exactly as
   defined).
2. **Screens**: which names may be held. Expression, market cap, liquidity,
   minimum price, listing age, exclusion lists, and the stale-price setting,
   which keeps the one definition index construction uses.
3. **Capacity caps**: how much of each may be held, against the book's size at
   each rebalance. Ownership (a share of free-float market cap), days to
   liquidate, minimum position. **Caps, not exclusions**, so a position shrinks
   smoothly as the fund grows rather than jumping to zero at a threshold.
4. **Redistribution**: removed or capped weight is spread across the remaining
   names pro rata, iterated so none breaks its own cap. Cash is an option.
5. **Trades**, generated as today.
6. **Execution limits**: at most a set share of a day's volume, the rest
   carried to later days.
7. **Costs**: fixed costs plus **market impact** (square-root law), which is
   what makes size matter. Without it a backtest's return does not depend on
   its size.

Screens use entry and exit thresholds (buffers), so names near a threshold do
not flip at every rebalance. All thresholds are in the book's currency, which
follows the index (#241), and every price read converts into it (#266).

### Dividends

Received on the pay date for the shares held on the ex-date, converted into the
book's currency. **Reinvested in the current holdings by default**, held as
cash until the next rebalance, or distributed, for every fund type. The first
two keep the income in the book, as an accumulating share class does; the
third pays it out, as a distributing class does. A flat withholding rate
first; rates by domicile later.

### AUM

- **Units.** Flows create or cancel units at the day's NAV per unit.
  Performance uses NAV per unit (time-weighted); a money-weighted return is
  reported beside it. With no flows every number is as it is today.
- **Flows**, every form, the user's choice: dated amounts, periodic amounts or
  percentages of AUM, stochastic (seeded), and performance-chasing. They are
  their own part of the run, the same whatever the vehicle.
- **Inflows are invested immediately**, toward the current target weights by
  default or pro rata to the holdings, as the implementation says. A flow too
  large for a day's participation limit is worked over several days.
- **Outflows** sell pro rata, drawing on the implementation's cash buffer
  first where one is set.
- **The management fee** is the vehicle's, accrued daily on AUM over calendar
  days (ACT/365) and taken from cash.

### Vehicles and funds

The **vehicle** and the **strategy** are separate. A `Vehicle` is how the money
is held: a structure preset and its settings (fees, pricing and dilution,
dealing and gates, creations, distributions). It says nothing about what is
held, and does not originate flows: it applies its structure's rules to the
run's flows. An ETF rounds them to creation units and deals them in cash or in
kind, an interval fund deals only on set dates behind gates, a closed-ended
fund takes none, and pricing or a levy decides who bears their trading cost. That keeps `run()` uniform, matches the app (the strategy chosen in
one place, the vehicle in another), and allows the comparisons that matter:
one ETF wrapper tracking two indices, or one index in an OEIC and in an ETF.

A **`Fund`** is what a real fund product is: a name, currency, share classes
and documents, with one vehicle and one strategy, as a prospectus has. It is a
record for saved products, factsheets and reports, not what the engine runs:

```python
fund = Fund(name="Sample UCITS ETF", vehicle=etf,
            strategy=IndexTracking(my_index))
fund.backtest(start, end)   # Backtest(vehicle=fund.vehicle).run(fund.strategy, ...)
```

`IndexFund` and `ETF` are deprecated over a release as shortcuts that build a
`Fund`.

**Structures are presets, not classes.** What changes a simulation is
behaviour, and global wrappers fall into six behavioural archetypes:

| Archetype | Wrappers | What the model does |
|---|---|---|
| Open-ended, dealt at NAV | UK OEIC, AUT, ACS; SICAV, FCP; ICAV, Irish plc; US mutual fund, CIT; Japanese investment trust; Australian MIS; HK OFC; Singapore VCC | Daily cash flows at NAV; the fund trades for flows (dilution), offset by single, dual or swing pricing or a levy |
| Exchange-traded | ETFs in those wrappers | Creations and redemptions in units, **cash by default, in kind as an option**; authorised-participant fees; market price with premium or discount and spread |
| Closed-ended, listed | Investment trust, US closed-end fund, REIT | Fixed capital; buybacks and issuance only; persistent discount; gearing |
| Interval | US interval fund, UK LTAF, ELTIF | Dealing on set dates, with redemption gates |
| Unit investment trust | US UIT | Full replication; dividends held as cash until paid; no lending |
| Segregated mandate | SMA | One client's flows; effectively the plain backtest |

A preset fills in the archetype's rules: pricing and dilution, dealing
frequency and gates, diversification limits (UCITS 5/10/40, 20% for index
trackers; 1940 Act 75/5/10), which feed the capacity caps, distribution rules,
and domicile. First presets: UK OEIC, Luxembourg SICAV, Irish ICAV (UCITS),
UCITS ETF, US mutual fund, US ETF.

### Strategies

A strategy is the source of the target weights, whatever the vehicle:

- **`IndexTracking(index, replication)`**: full physical, or optimised or
  sampled physical first; synthetic (a substitute basket swapped for the
  index's return, reset at a counterparty exposure limit) and futures-based
  (cash plus rolled futures, and futures to equitise a cash buffer) later.
- **`ActiveStrategy`**: a universe and screens, a signal from expressions or
  features, construction by the optimiser with benchmark-relative constraints,
  and a rebalance schedule. Almost every part exists already; what is new is
  assembling them and reporting against a benchmark (active share,
  information ratio, attribution).

### In the app

One Backtest window: the strategy (an index, with its replication, or an
active strategy) chosen in one place, and a vehicle choice (Backtest, a
structure preset, or an ETF) in another. Choosing a vehicle reveals only its
settings; implementation settings apply to all; the results pane is shared,
with the vehicle's own metrics added. The engine's backtest request gains
optional `strategy`, `implementation` and `vehicle` objects, additively, and
saved runs record them.

## Phases

| Phase | Issue |
|---|---|
| 1. Dividends in the backtest | BN-263 (#276) |
| 2. Staged backtest and screens (with #266 currency, #268, #241 book currency) | BN-264 (#277) |
| 3. Capacity caps | BN-265 (#278) |
| 4. Market impact and execution limits (with #265) | BN-266 (#279) |
| 5. AUM: units, flows, and a minimal vehicle with the fee | BN-267 (#280) |
| 6. Vehicles and structure presets; Fund as a product record | BN-268 (#281) |
| 7. Strategies: replication and active | BN-269 (#282) |
| 8. Synthetic and futures replication; withholding tax by domicile | BN-270, BN-271 (#283, #284) |
| 9. Engine API, then the app's vehicle picker | BN-272 (#285) |
| Alongside 2: modelling assumptions | BN-276 (#289) |

## Consequences

- The index stays pure. A backtest's tracking difference against it now
  measures what implementation, size and the vehicle cost, which is the
  question being asked.
- Metrics move to NAV per unit. Identical without flows, but any code that
  treats total NAV as performance needs checking when flows arrive.
- Behaviour changes to backtest results (dividends, net-of-cost sizing, impact)
  are breaking before 1.0 and raise the minor version.
- Structures as data rather than classes means a new wrapper is a preset, not
  code, and the app can list them from the engine.
- The engine only ever sees a strategy and a vehicle. A `Fund` bundles one of
  each with an identity, so a saved product and a what-if run are the same
  computation.
