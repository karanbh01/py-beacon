# src/beacon/strategy/futures.py
"""
Futures-based replication: cash, with the index held through rolled futures.

    IndexTracking(my_index, replication=FuturesReplication(roll_cost_bps=2.0))

The fund holds no securities. Its cash earns the run's cash rate, and a long
position in the index's futures, sized to its net assets, carries the index's
return. Each contract is priced at fair value from the index level, the cash
rate and the index's dividend yield:

    F = S x exp((r - q) x T)

with T the time to expiry over calendar days (ACT/365). Profit and loss is
settled in cash every day, as variation margin is. Because the futures give
up the cash rate and are paid the dividend yield, the fund earns roughly the
index's price return plus its dividend yield: its total return, before the
cost of rolling.

**Contracts** expire quarterly, on the third Friday of March, June,
September and December, and the position rolls to the next one *roll_days*
sessions before expiry, paying *roll_cost_bps* on its notional each time:
the richness that real rolls pay over fair value.

**Prices.** Fair value is priced off the fund's own index unless
*underlying* names another for the futures: a market series by its
identifier, an index built in Beacon (its `IndexResult`), or a level series.
Futures on another index than the fund tracks carry the gap between the two
as well as their own. With *prices* naming a series in the data, each day's
close there is the price of the contract held (a front-month series rolled
on the same days); on a day it has none, and with no series, the price is
fair value moved by an assumed mispricing: mean-reverting around 0, its
typical size *mispricing_bps* (its standard deviation, in basis points of
fair value; 0 by default, so the price is fair value), halving its distance
from 0 in *mispricing_half_life* sessions on average, drawn from
*mispricing_seed* so a run repeats.

**Trading costs.** Every futures trade pays *cost_per_contract* on each
contract it trades (a roll trades two: the old contract sold and the new one
bought), and each trade between rolls (opening the position and each resize)
pays *trading_cost_bps* on the notional it trades as well.

**The dividend yield.** A total return or net total return index already
counts its dividends, so its futures carry none. A price index's are
estimated at each roll from its constituents' dividends over the trailing
twelve months, at their index weights, unless *dividend_yield* gives one.

**Sizing.** A contract is worth its price times its *multiplier*, the
value of one index point (50 by default, as an S&P 500 e-mini's is in
dollars). The position is sized to the fund's net assets when it opens, at
each rebalance and roll, on any day money flows in or out, and on any other
day its exposure is further from that than *band* of the net assets.
Contracts are whole by default, so the exposure is the nearest a whole
number of contracts comes; fractional contracts carry exactly what was asked
for. Initial margin is not modelled, since it is the fund's own cash, held
with a broker.

The same futures can equitise a physical fund's cash buffer instead:
`Implementation(cash_buffer=0.03, equitise=IndexFutures())` holds futures on
the cash the fund keeps uninvested, so the buffer earns the index's return.
"""
# BN-305, phase 8 of decisions/0006. Decided with Karan on 2026-10-11: fair
# value plus a roll cost rather than market futures prices, which few
# datasets have.
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..catalogue import BPS, COUNT, FRACTION, MONEY, REPLICATION, Display, register
from .base import StrategyContext
from .tracking import Replication, ReplicationStep

# Index futures expire in these months, on the third Friday.
EXPIRY_MONTHS = (3, 6, 9, 12)

# Sessions before expiry a position rolls, a common practice for index
# futures, whose liquidity moves to the next contract in the last week.
DEFAULT_ROLL_DAYS = 5

# The value of one index point per contract, as an S&P 500 e-mini's is $50.
DEFAULT_MULTIPLIER = 50.0

# How far the exposure may drift from its target, as a share of the net
# assets, before the position is resized between rebalances.
DEFAULT_BAND = 0.0025


@dataclass(frozen=True)
class IndexFutures:
    """The index futures a fund holds.

    Attributes:
        roll_days: Sessions before expiry the position rolls to the next
            contract.
        roll_cost_bps: What each roll costs, in basis points of the notional.
        dividend_yield: The index's annual dividend yield the futures are
            priced with. None estimates it (zero for an index that reinvests
            its dividends).
        multiplier: The value of one index point per contract, in the
            book's currency.
        whole_contracts: Hold a whole number of contracts (the default), or
            fractional ones that carry exactly the target.
        band: How far the exposure may drift from its target, as a share of
            the net assets, before the position is resized between
            rebalances.
        cost_per_contract: What each contract traded costs, in the book's
            currency: commission and fees.
        trading_cost_bps: What a trade between rolls costs, in basis points
            of the notional it trades: the spread paid.
        underlying: The index the futures are on, when not the fund's: a
            market series' identifier, an `IndexResult` or a level series.
        prices: A series in the data whose close is the price of the
            contract held each day.
        mispricing_bps: The typical size of the assumed mispricing when the
            data has no price: its standard deviation, in basis points of
            fair value.
        mispricing_half_life: The sessions the mispricing takes on average
            to halve.
        mispricing_seed: The seed its draws are made from.
    """
    roll_days: int = DEFAULT_ROLL_DAYS
    roll_cost_bps: float = 0.0
    dividend_yield: float | None = None
    # BN-358.
    multiplier: float = DEFAULT_MULTIPLIER
    whole_contracts: bool = True
    band: float = DEFAULT_BAND
    # BN-409: decided with Karan on 2026-10-27.
    cost_per_contract: float = 0.0
    trading_cost_bps: float = 0.0
    # A level series does not compare as a value, so it is left out of
    # equality.
    underlying: Any = field(default=None, compare=False)
    prices: str | None = None
    mispricing_bps: float = 0.0
    mispricing_half_life: float = 10.0
    mispricing_seed: int = 0

    def __post_init__(self) -> None:
        if self.roll_days < 0 or self.roll_days != int(self.roll_days):
            raise ValueError(f"roll_days is a whole number of sessions, zero or more, "
                             f"got {self.roll_days!r}.")

        # A whole number that arrived as 5.0, as JSON numbers can.
        object.__setattr__(self, "roll_days", int(self.roll_days))

        if not math.isfinite(self.roll_cost_bps) or self.roll_cost_bps < 0.0:
            raise ValueError(f"roll_cost_bps must be zero or more, got "
                             f"{self.roll_cost_bps!r}.")

        if self.dividend_yield is not None and not 0.0 <= self.dividend_yield < 1.0:
            raise ValueError(f"dividend_yield is an annual share from 0 up to 1, "
                             f"got {self.dividend_yield!r}.")

        if not math.isfinite(self.multiplier) or self.multiplier <= 0.0:
            raise ValueError(f"multiplier must be above zero, got {self.multiplier!r}.")

        if not isinstance(self.whole_contracts, bool):
            raise ValueError(f"whole_contracts is true or false, got "
                             f"{self.whole_contracts!r}.")

        if not math.isfinite(self.band) or not 0.0 <= self.band < 1.0:
            raise ValueError(f"band is a share of the net assets from 0 up to 1, "
                             f"got {self.band!r}.")

        for name in ("cost_per_contract", "trading_cost_bps", "mispricing_bps"):
            value = getattr(self, name)

            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be zero or more, got {value!r}.")

        if not math.isfinite(self.mispricing_half_life) or self.mispricing_half_life <= 0.0:
            raise ValueError(f"mispricing_half_life must be above zero, got "
                             f"{self.mispricing_half_life!r}.")

        if self.mispricing_seed != int(self.mispricing_seed):
            raise ValueError(f"mispricing_seed is a whole number, got "
                             f"{self.mispricing_seed!r}.")

        object.__setattr__(self, "mispricing_seed", int(self.mispricing_seed))

        if not (self.underlying is None or isinstance(self.underlying, str | pd.Series)
                or hasattr(self.underlying, "index_levels")):
            raise ValueError(f"underlying is an identifier, an IndexResult or a level "
                             f"series, got {type(self.underlying).__name__}.")

    def underlying_levels(self) -> pd.Series | None:
        """The underlying's levels when given as an index or a series; None
        for an identifier, which the data is asked for, or for none."""
        if isinstance(self.underlying, pd.Series):
            return self.underlying.sort_index()

        if hasattr(self.underlying, "index_levels"):
            return pd.Series(self.underlying.index_levels).sort_index()

        return None

    def trade_cost(self,
                   contracts: float,
                   price: float,
                   rolling: bool = False) -> float:
        """What trading *contracts* at *price* costs: per contract, and in
        basis points of the notional for a trade between rolls."""
        traded = abs(contracts)
        cost = traded * self.cost_per_contract

        if not rolling:
            cost += traded * price * self.multiplier * self.trading_cost_bps / 10_000.0

        return float(cost)

    def contracts(self,
                  target: float,
                  price: float) -> float:
        """How many contracts carry *target* at *price*: whole, the nearest
        number, or fractional, exactly."""
        value = price * self.multiplier

        if value <= 0.0:
            return 0.0

        held = target / value

        return float(math.floor(held + 0.5)) if self.whole_contracts else held


@register(REPLICATION, "Futures", fields={
    "roll_days": Display("Days before expiry to roll", unit=COUNT, minimum=0),
    "roll_cost_bps": Display("Cost of each roll"),
    "dividend_yield": Display("Dividend yield", unit=FRACTION, minimum=0.0,
                              help="Leave blank to estimate it from the constituents."),
    "multiplier": Display("Value of an index point", unit=MONEY, minimum=0.0,
                          help="Per contract, in the book's currency."),
    "whole_contracts": Display("Whole contracts"),
    "band": Display("Resize when exposure drifts by", unit=FRACTION, minimum=0.0,
                    maximum=1.0, help="A share of the net assets."),
    "cost_per_contract": Display("Cost per contract", unit=MONEY, minimum=0.0,
                                 help="Each contract traded, in the book's currency."),
    "trading_cost_bps": Display("Cost of trading between rolls", unit=BPS, minimum=0.0,
                                help="On the notional each trade between rolls deals."),
    "underlying": Display("Index the futures are on",
                          help="A market series' identifier. Leave blank for the fund's "
                               "own index."),
    "prices": Display("Futures prices",
                      help="A series in the data with the held contract's price each day. "
                           "Leave blank to price at fair value."),
    "mispricing_bps": Display("Typical mispricing", unit=BPS, minimum=0.0,
                              help="Its standard deviation, when the data has no price."),
    "mispricing_half_life": Display("Mispricing half-life", unit=COUNT, minimum=0.0,
                                    help="Sessions it takes on average to halve."),
    "mispricing_seed": Display("Mispricing seed", unit=COUNT),
})
class FuturesReplication(Replication):
    """Cash, with the index held through rolled futures.

    Args:
        roll_days: Sessions before expiry the position rolls.
        roll_cost_bps: What each roll costs, in basis points of the notional.
        dividend_yield: The index's annual dividend yield the futures are
            priced with. None estimates it.
        multiplier: The value of one index point per contract.
        whole_contracts: Hold a whole number of contracts, or fractional ones.
        band: How far the exposure may drift before it is resized between
            rebalances, as a share of the net assets.
        cost_per_contract: What each contract traded costs.
        trading_cost_bps: What a trade between rolls costs, in basis points
            of the notional it trades.
        underlying: The index the futures are on, when not the fund's: a
            market series' identifier, or in Python an `IndexResult` or a
            level series.
        prices: A series in the data with the held contract's price.
        mispricing_bps: The typical size of the assumed mispricing.
        mispricing_half_life: The sessions it takes on average to halve.
        mispricing_seed: The seed its draws are made from.

    Raises:
        ValueError: If a setting is out of range.
    """

    def __init__(self,
                 roll_days: int = DEFAULT_ROLL_DAYS,
                 roll_cost_bps: float = 0.0,
                 dividend_yield: float | None = None,
                 multiplier: float = DEFAULT_MULTIPLIER,
                 whole_contracts: bool = True,
                 band: float = DEFAULT_BAND,
                 cost_per_contract: float = 0.0,
                 trading_cost_bps: float = 0.0,
                 underlying: str | None = None,
                 prices: str | None = None,
                 mispricing_bps: float = 0.0,
                 mispricing_half_life: float = 10.0,
                 mispricing_seed: int = 0):
        self.futures = IndexFutures(roll_days=roll_days, roll_cost_bps=roll_cost_bps,
                                    dividend_yield=dividend_yield, multiplier=multiplier,
                                    whole_contracts=whole_contracts, band=band,
                                    cost_per_contract=cost_per_contract,
                                    trading_cost_bps=trading_cost_bps,
                                    underlying=underlying, prices=prices,
                                    mispricing_bps=mispricing_bps,
                                    mispricing_half_life=mispricing_half_life,
                                    mispricing_seed=mispricing_seed)

    @property
    def roll_days(self) -> int:
        """Sessions before expiry the position rolls."""
        return self.futures.roll_days

    @property
    def roll_cost_bps(self) -> float:
        """What each roll costs, in basis points of the notional."""
        return self.futures.roll_cost_bps

    @property
    def dividend_yield(self) -> float | None:
        """The dividend yield given, or None to estimate it."""
        return self.futures.dividend_yield

    @property
    def multiplier(self) -> float:
        """The value of one index point per contract."""
        return self.futures.multiplier

    @property
    def whole_contracts(self) -> bool:
        """Whether the position holds whole contracts."""
        return self.futures.whole_contracts

    @property
    def band(self) -> float:
        """How far the exposure may drift before it is resized."""
        return self.futures.band

    @property
    def cost_per_contract(self) -> float:
        """What each contract traded costs."""
        return self.futures.cost_per_contract

    @property
    def trading_cost_bps(self) -> float:
        """What a trade between rolls costs, in basis points of its notional."""
        return self.futures.trading_cost_bps

    @property
    def underlying(self) -> Any:
        """The index the futures are on, when not the fund's."""
        return self.futures.underlying

    @property
    def prices(self) -> str | None:
        """The series in the data with the held contract's price."""
        return self.futures.prices

    @property
    def mispricing_bps(self) -> float:
        """The typical size of the assumed mispricing."""
        return self.futures.mispricing_bps

    @property
    def mispricing_half_life(self) -> float:
        """The sessions the mispricing takes on average to halve."""
        return self.futures.mispricing_half_life

    @property
    def mispricing_seed(self) -> int:
        """The seed its draws are made from."""
        return self.futures.mispricing_seed

    def replicate(self,
                  target: dict[str, float],
                  date: pd.Timestamp,
                  context: StrategyContext) -> ReplicationStep:
        # No securities: the whole fund is cash, and the futures carry the
        # index.
        return ReplicationStep(date=date, weights={}, holdings=0)

    def __repr__(self) -> str:
        return (f"FuturesReplication(roll_days={self.roll_days!r}, "
                f"roll_cost_bps={self.roll_cost_bps!r}, "
                f"dividend_yield={self.dividend_yield!r}, "
                f"multiplier={self.multiplier!r}, "
                f"whole_contracts={self.whole_contracts!r}, band={self.band!r}, "
                f"cost_per_contract={self.cost_per_contract!r}, "
                f"trading_cost_bps={self.trading_cost_bps!r}, "
                f"underlying={self.underlying!r}, prices={self.prices!r}, "
                f"mispricing_bps={self.mispricing_bps!r}, "
                f"mispricing_half_life={self.mispricing_half_life!r}, "
                f"mispricing_seed={self.mispricing_seed!r})")


def expiries(start: pd.Timestamp,
             end: pd.Timestamp) -> list[pd.Timestamp]:
    """The quarterly expiries from *start* to a year after *end*: the third
    Friday of March, June, September and December."""
    found = []

    for year in range(start.year, end.year + 2):
        for month in EXPIRY_MONTHS:
            first = pd.Timestamp(year=year, month=month, day=1)
            # Friday is weekday 4; the first Friday, then two weeks on.
            friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)

            if friday >= start:
                found.append(friday)

    return found
