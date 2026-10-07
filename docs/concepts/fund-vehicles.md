---
title: Fund vehicles and pricing
description: "How Beacon models fund structures: the structure presets, the four ways a fund prices its dealing, and how an ETF's market price, spread and premium or discount are simulated."
---

# Fund vehicles and pricing

A [backtest](backtest.md) has five parts: the **strategy** (what is held),
the **implementation** (screens, caps, costs and execution), the **flows**
(money arriving and leaving), the **vehicle** (how the money is held) and the
**modelling assumptions**. This page is about the vehicle: the structure a
fund is built in, how it prices the units investors buy and sell, and, for an
exchange-traded fund, how its shares trade on the exchange.

Keeping the vehicle separate is what makes fair comparisons possible: the
same index, limits and flows run through a UK OEIC and a UCITS ETF differ
only by what the structure does.

## Structures and presets

Fund structures differ by country and by wrapper, but what changes a
simulation is behaviour, and the world's wrappers fall into six behavioural
archetypes:

| Archetype | Examples | What the model does |
| --- | --- | --- |
| Open-ended, dealt at NAV | UK OEIC, AUT and ACS; SICAV and FCP; ICAV; US mutual fund; Japanese investment trust; Australian MIS | Investors deal with the fund at its price each day; the fund trades for their flows, and its pricing method decides who pays for that trading |
| Exchange-traded | ETFs in any of those wrappers | Only authorised participants deal with the fund, in creation units; everyone else trades shares on an exchange at a market price |
| Closed-ended, listed | Investment trust, US closed-end fund, REIT | Fixed capital, changed only by buybacks and issuance; the shares trade at a persistent premium or discount |
| Interval | US interval fund, UK LTAF, ELTIF | Dealing on set dates, with limits on how much can leave |
| Unit investment trust | US UIT | A fixed portfolio, dividends held as cash until paid |
| Segregated mandate | SMA | One client's money; effectively the plain backtest |

A **preset** is a named set of vehicle settings for one wrapper: how it
deals, how it prices, its diversification limits and how it handles
distributions. A preset is a starting point, and any setting can be changed.
The first six presets cover the first two archetypes:

| Preset | Dealing | Pricing | Diversification limits |
| --- | --- | --- | --- |
| UK OEIC | Daily, at the next valuation point | Swing pricing | UCITS |
| Luxembourg SICAV (UCITS) | Daily, at the next valuation point | Partial swing pricing | UCITS |
| Irish ICAV (UCITS) | Daily, at the next valuation point | Anti-dilution levy | UCITS |
| UCITS ETF | Creation units, cash by default | ETF market price | UCITS |
| US mutual fund | Daily, at the next NAV | Single pricing, with an optional redemption fee | 1940 Act diversified |
| US ETF | Creation units, in kind by default | ETF market price | 1940 Act diversified |

The pricing each preset starts with is a common choice for that wrapper, not
a rule of it: most of these wrappers allow several methods.

**Dealing.** Every preset deals at the price set at the close of the day the
flow arrives (forward pricing), so an investor never deals at a price that
was already known.

**Diversification limits** become [capacity caps](backtest.md#capacity), so a
fund near a limit holds less of a name rather than breaking it:

- **UCITS**: at most 10% of the fund in one issuer, and the holdings above 5%
  may add up to at most 40%. A fund replicating an index may hold up to 20%
  of one issuer, or 35% of a single one where the index is unusually
  concentrated.
- **1940 Act diversified**: for 75% of the fund, at most 5% in one issuer
  and at most 10% of an issuer's voting shares. Beacon applies this as: the
  holdings above 5% may add up to at most 25% of the fund, and no holding
  exceeds 10% of the company's shares outstanding.

**Distributions** follow the run's dividend policy: reinvested by default,
as an accumulating share class does, or paid out, as a distributing class
does.

### Using a preset

Pass a preset as the backtest's `vehicle`, by its function or by name, and
change any setting by passing it:

```python
import logging

from beacon.backtest import (
    Backtest,
    DatedFlows,
    DilutionLevy,
    PeriodicFlows,
    preset,
    uk_oeic,
)
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

fetcher = dataset.data_fetcher()
definition = IndexDefinition(
    index_id="SAMPLE", index_name="Sample Market-Cap Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)
flows = [PeriodicFlows(fraction=0.02), DatedFlows({"2024-03-01": -2_000_000.0})]

for vehicle in (uk_oeic(management_fee_bps=15),
                preset("irish_icav", pricing=DilutionLevy(rate_bps=25))):
    result = Backtest(initial_capital=10_000_000.0, transaction_cost_bps=10.0,
                      flows=flows, vehicle=vehicle, data_provider=fetcher,
                      ).run(definition, start="2023-01-03", end="2024-12-31")
    paid_in = sum(flow.adjustment for flow in result.flows)
    print(vehicle.name, round(result.nav_per_unit.iloc[-1], 4),
          "paid in by dealing investors:", round(paid_in, 2))
```

`PRESETS` lists every preset by name.

## Dilution, and the four pricing methods

When investors buy into or sell out of an open-ended fund, the fund trades to
invest or raise the money. That trading costs money: the bid-ask spread on
each holding, commissions, taxes such as stamp duty, and the market impact of
the trade. If nothing else happens, the cost comes out of the fund, so the
investors who stayed pay for the ones who came or went. That loss is called
**dilution**. A pricing method decides who pays.

In each method the fund's NAV per unit is first calculated at the close as
usual. The method then sets the **dealing price** units are created or
cancelled at, or adds a charge. Whatever a dealing investor pays above NAV,
or receives below it, stays in the fund and offsets the trading cost.

### Single pricing

Everyone deals at NAV per unit, and nothing is adjusted:

```text
dealing price = NAV per unit
```

The trading cost of the day's flows is borne by the whole fund. This is the
baseline the other three are measured against.

### Dual pricing

The fund quotes two prices. Buyers pay the **offer** price: the value of the
holdings at the prices they could be bought at, plus the cost of buying them.
Sellers receive the **bid** price: the holdings at the prices they could be
sold at, less the cost of selling them.

```text
offer price = NAV per unit x (1 + offer adjustment)
bid price   = NAV per unit x (1 - bid adjustment)
```

Each investor pays the cost of their own dealing, so the fund is not diluted,
but every buyer and seller pays the spread whether or not the fund had to
trade much that day. This was the traditional method for UK unit trusts.

### Swing pricing

The fund keeps a single price, which moves ("swings") with the day's net
flow. On a day of net inflows the price swings up, so buyers pay the cost of
investing their money; on a day of net outflows it swings down, so sellers
pay the cost of raising theirs:

```text
net inflow:  dealing price = NAV per unit x (1 + swing factor)
net outflow: dealing price = NAV per unit x (1 - swing factor)
```

Under **full swing** the price swings every day there is a net flow. Under
**partial swing** it swings only when the net flow is larger than a
threshold, such as 2% of the fund, because small flows can be met from cash
and cost little. A partial swing leaves small days diluted and protects
against the large ones, which do the damage.

The **swing factor** is meant to be the cost of trading the day's net flow.
It can be set as a number of basis points, or estimated from the run's own
cost model: the fixed cost plus [market impact](backtest.md#costs-and-execution)
on the trades the flow actually needs. An estimated factor grows on days the
flow is large or the holdings are hard to trade.

Swing pricing is common for Luxembourg SICAVs and Irish ICAVs and allowed for
UK OEICs. US rules have permitted it since 2018, though US funds have not
adopted it.

### Dilution levy

The price stays at NAV, and a deal larger than a threshold pays a separate
charge into the fund instead:

```text
levy = levy rate x deal amount        (deals above the threshold)
units bought    = (amount paid - levy) / NAV per unit
amount received = units sold x NAV per unit - levy
```

Only the investors whose deals cause meaningful trading pay, and the price
everyone else sees does not move. UK OEICs and Irish funds use levies,
mostly for large deals. A US mutual fund's **redemption fee**, charged to
investors who sell soon after buying, works the same way; Beacon does not
track how long each investor held, so it applies the fee to every redemption.

### Measuring the protection

The run's flow record shows, for every flow, the dealing price, the units
created or cancelled and the amount the dealing investor paid into the fund
through the price or levy. Comparing a fund's NAV per unit under each method
with the same flows shows how much dilution each one prevented.

## Exchange-traded funds

An ETF has two markets.

- In the **primary market**, authorised participants (APs) create and redeem
  shares with the fund, in blocks called **creation units** (50,000 shares,
  say), at NAV.
- In the **secondary market**, everyone else buys and sells existing shares on
  an exchange, from each other and from market makers, at a market price.

An investor who buys on the exchange pays the market price, which can sit
above NAV (a **premium**) or below it (a **discount**), plus half the
**bid-ask spread**. Beacon simulates both.

### Creations and redemptions

The run's flows are the APs' net demand for creations (positive) or
redemptions (negative). Each day they are rounded down to whole creation
units, and what is left over carries to the next day, as an AP waits until a
full unit is worth creating.

- A **cash** creation delivers money; the fund buys the holdings and pays the
  trading costs. A variable creation fee, in basis points, charges those
  costs back to the AP, so the fund is not diluted.
- An **in-kind** creation delivers the holdings themselves; the fund does not
  trade, and the AP bears the cost of assembling the basket.

Redemptions work the same way in reverse. UCITS ETFs create in cash by
default and US ETFs in kind, as is usual for each.

### The arbitrage band

APs keep the market price close to NAV. When the ETF trades at a premium
larger than the cost of creating, an AP can buy the holdings, create shares
at NAV and sell them at the market price for a profit; the new shares push
the price down. At a discount larger than the cost of redeeming, an AP buys
shares cheaply, redeems them for the holdings and sells those. So the
premium is bounded by what creating and redeeming cost:

```text
create cost = AP fee per unit + the cost of buying one creation unit's basket
redeem cost = AP fee per unit + the cost of selling one creation unit's basket

- redeem cost / unit value  <=  premium  <=  create cost / unit value
```

The basket's cost comes from the run's cost model: the fixed cost plus the
market impact of trading one creation unit's worth of every holding. The band
is narrow for an ETF of large, liquid shares and wide for one holding small or
illiquid ones, and it widens on days the basket becomes expensive to trade.

### The premium or discount

Inside the band the premium is not zero. It persists from day to day, it is
pushed by buying and selling pressure, and in a sharp fall the market price
tends to fall faster than the holdings' closing prices reflect, opening a
discount (as bond ETFs showed in March 2020). Beacon models the premium, as a
fraction of NAV, as:

```text
premium(t) = persistence x premium(t-1)
           + flow sensitivity x net flow(t) / AUM
           + return sensitivity x NAV return(t)
           + noise x a standard normal draw
             then held inside the arbitrage band
```

- **Persistence** (between 0 and 1) sets how long a premium lasts: 0.5 halves
  it each day.
- **Flow sensitivity** turns buying pressure into a premium and selling
  pressure into a discount, before APs close it.
- **Return sensitivity** lets the price lead NAV on large moves, so a sharp
  fall opens a discount and a sharp rise a premium.
- **Noise** is the day-to-day wander, seeded so a run repeats exactly.

### The bid-ask spread

Market makers quote a spread that pays them for the cost of hedging their
inventory in the holdings and for the risk of holding it. ETF spreads are
known to be wider when the holdings are less liquid, wider when markets are
volatile, much wider in stress, and never narrower than one price tick.
Beacon models the full spread, as a fraction of the market price, as:

```text
spread(t) = max(tick / price,
                floor
                + basket sensitivity x the cost of trading one creation unit's basket(t)
                + volatility sensitivity x the NAV's recent daily volatility(t))
```

The basket term makes the spread follow the holdings' liquidity, and the
volatility term widens it in turbulent markets. Both rise together in a
crisis, which is what produces the wide spreads seen then.

### What the run reports

```text
market price = NAV per unit x (1 + premium)
ask          = market price x (1 + spread / 2)
bid          = market price x (1 - spread / 2)
```

An ETF run reports, for each day, the market price, the premium or discount,
the spread, the bid and the ask, and the creation units issued or redeemed.
Beside the NAV return it reports the return an exchange investor made: buying
at the ask on the first day and selling at the bid on the last, with the
market price in between. The difference between the two is what the ETF's
trading cost its secondary-market investors.

### In code

`ucits_etf()` and `us_etf()` are the two ETF presets, and `EtfVehicle` builds
any other. `EtfMarket` holds the quote model's settings, all of which have
defaults:

```python
from beacon.backtest import EtfMarket, MarketImpact, Implementation, ucits_etf

etf = ucits_etf(creation_unit=50_000, ap_fee=500.0,
                market=EtfMarket(spread_floor_bps=2.0, persistence=0.5,
                                 flow_sensitivity=0.1, return_sensitivity=0.05,
                                 noise_bps=2.0, seed=7))

listed = Backtest(initial_capital=50_000_000.0, transaction_cost_bps=5.0,
                  implementation=Implementation(impact=MarketImpact()),
                  flows=[PeriodicFlows(fraction=0.02)], vehicle=etf,
                  data_provider=fetcher,
                  ).run(definition, start="2023-01-03", end="2024-12-31")

print(listed.market[["premium", "spread", "bid", "ask"]].tail(3))
print({name: round(value, 6) for name, value in listed.market_summary().items()})
```

`result.market` holds each day's quote, and `market_summary()` (also part of
`summary()`) gives the exchange investor's return and the average premium and
spread. Each flow's `creation_units` says how many units it was.

### What the model leaves out

- **Stale holdings prices.** When an ETF's holdings trade in a market that is
  closed while the ETF trades (a US-listed ETF of Japanese shares, say), the
  market price moves on news the NAV has not seen yet, so part of its
  apparent premium is the NAV being out of date. The model works on one daily
  close and does not separate this out.
- **Intraday trading.** Prices are daily closes; the spread is the typical
  closing spread, not the range seen through a day.
