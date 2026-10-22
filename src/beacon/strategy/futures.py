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
from dataclasses import dataclass

import pandas as pd

from ..catalogue import COUNT, FRACTION, MONEY, REPLICATION, Display, register
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
    """
    roll_days: int = DEFAULT_ROLL_DAYS
    roll_cost_bps: float = 0.0
    dividend_yield: float | None = None
    # BN-358.
    multiplier: float = DEFAULT_MULTIPLIER
    whole_contracts: bool = True
    band: float = DEFAULT_BAND

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

    Raises:
        ValueError: If a setting is out of range.
    """

    def __init__(self,
                 roll_days: int = DEFAULT_ROLL_DAYS,
                 roll_cost_bps: float = 0.0,
                 dividend_yield: float | None = None,
                 multiplier: float = DEFAULT_MULTIPLIER,
                 whole_contracts: bool = True,
                 band: float = DEFAULT_BAND):
        self.futures = IndexFutures(roll_days=roll_days, roll_cost_bps=roll_cost_bps,
                                    dividend_yield=dividend_yield, multiplier=multiplier,
                                    whole_contracts=whole_contracts, band=band)

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
                f"whole_contracts={self.whole_contracts!r}, band={self.band!r})")


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
