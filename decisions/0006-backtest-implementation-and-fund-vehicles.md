# 6. Backtest implementation, AUM, and funds as vehicles

Date: 2026-09-29

## Status

Accepted. Built in phases, tracked as BN-263 to BN-272 (#276 to #285).

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

A backtest run has three independent parts:

1. **What is held**: an index definition for a plain backtest, or a fund's
   strategy.
2. **Implementation**: screens, capacity caps, costs, execution limits. It
   applies to every vehicle, since a plain backtest at 100 billion needs
   capacity limits too.
3. **Vehicle**: who holds the portfolio and how money moves. None for a plain
   backtest, or a `Fund`.

Keeping them apart is what allows a fair comparison: the same index and the
same limits, only the vehicle changed (an index fund against an ETF on the
same flows, say).

### The backtest as stages

Each stage has its own hook, in this order:

1. **Target weights**, from the index (which stays exactly as defined) or the
   strategy.
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
book's currency. **Accumulated by default, distributed as an option**, for
every fund type. A flat withholding rate first; rates by domicile later.

### AUM

- **Units.** Flows create or cancel units at the day's NAV per unit.
  Performance uses NAV per unit (time-weighted); a money-weighted return is
  reported beside it. With no flows every number is as it is today.
- **Flows**, every form, the user's choice: dated amounts, periodic amounts or
  percentages of AUM, stochastic (seeded), performance-chasing, and creation
  units for exchange-traded funds.
- **Inflows are invested immediately.** A flow too large for a day's
  participation limit is worked over several days.
- **Outflows** sell pro rata, drawing on a cash buffer first where one is set.

### Funds

A **`Fund`** has an identity (name, currency, share classes, fees), a
**structure** and a **strategy**, and is passed to a backtest as its vehicle:
`Backtest(vehicle=fund)`, or `fund.backtest(...)` as a shortcut. A fund carries
its strategy, so its run takes no index; a plain backtest keeps
`run(definition, ...)`. `IndexFund` and `ETF` are deprecated over a release as
shortcuts that build a `Fund`.

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

The strategy is the source of a fund's target weights:

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

One Backtest window with a vehicle choice: Backtest, a structure preset, or an
ETF. Choosing one reveals only that vehicle's settings; implementation settings
apply to all; the results pane is shared, with the vehicle's own metrics added.
The engine's backtest request gains optional `implementation`, `vehicle` and
`strategy` objects, additively, and saved runs record them.

## Phases

| Phase | Issue |
|---|---|
| 1. Dividends in the backtest | BN-263 (#276) |
| 2. Staged backtest and screens (with #266 currency, #268, #241 book currency) | BN-264 (#277) |
| 3. Capacity caps | BN-265 (#278) |
| 4. Market impact and execution limits (with #265) | BN-266 (#279) |
| 5. AUM: units and flows | BN-267 (#280) |
| 6. Fund, structure presets, vehicles | BN-268 (#281) |
| 7. Strategies: replication and active | BN-269 (#282) |
| 8. Synthetic and futures replication; withholding tax by domicile | BN-270, BN-271 (#283, #284) |
| 9. Engine API, then the app's vehicle picker | BN-272 (#285) |

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
