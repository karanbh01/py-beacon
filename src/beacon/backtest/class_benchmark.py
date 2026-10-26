# src/beacon/backtest/class_benchmark.py
"""
What a share class is measured against: the index in the class's currency.

A class in the index's own currency is measured against the index as it is. An
unhedged class in another currency is measured against the index converted
into its currency at each day's rate, so the currency move is in both. A
hedged class is measured against the index **hedged the same way**: each month
the index's value in its own currency is sold forward for the class's, at the
class's hedge ratio, on the class hedge's terms. So a hedged class's tracking
difference is its fees, costs and hedge slippage, not the interest rate
differential its hedge earns or pays.

**The hedged index.** At the first session of each month the index's value
*I0* is converted at the day's spot *S0* (class currency per unit of the
index's) and a forward to the first business day of the next month is struck
at covered interest parity:

    F0 = S0 x (1 + r_class x t_class) / (1 + r_index x t_index)

On each day *t* of the month the hedged level moves by the index's return in
the class's currency, plus the forward's gain per unit of that value:

    H_t = H0 x ((I_t x S_t) / (I0 x S0) + h x (F0 - F_t) / (S0 x D_t))

where *F_t* is the day's forward to the same date, *D_t* discounts its gain
at the class currency's rate and *h* is the hedge ratio. The month's level is
the base of the next. The rates are each currency's short rate in the data on
its own day count, else the cash rate on ACT/360, as the class's hedge prices
them.
"""
# BN-314. Karan's call on 2026-10-16: a hedged class is measured against the
# index hedged the same way, so its tracking difference leaves out the carry.
from dataclasses import dataclass

import pandas as pd

from ..analysis.performance import formulas
from ..assumptions import DEFAULT_PERIODS_PER_YEAR
from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from .share_classes import ShareClassBook

# Money-market days in a year for a currency whose day count the data does not
# give: the class hedge's own default.
FORWARD_DAY_COUNT = 360.0


def index_in_class_currency(levels: pd.Series,
                            index_currency: str,
                            currency: str,
                            data_provider: DataFetcher,
                            hedge_ratio: float | None = None,
                            cash_rate: float | None = None) -> pd.Series:
    """The index's level in a share class's currency, converted or hedged.

    Args:
        levels: The index's levels, in *index_currency*, by date.
        index_currency: The currency the levels are in.
        currency: The class's currency.
        data_provider: The data the exchange and short rates are read from.
        hedge_ratio: The class's hedge ratio for a hedged class; None for an
            unhedged one.
        cash_rate: The rate for a currency with no short rate in the data.

    Returns:
        pd.Series: The levels in *currency*, on the same dates.

    Raises:
        CalculationError: If there is no rate between the two currencies on
            the first date.
    """
    if levels.empty or index_currency.upper() == currency.upper():
        return levels.astype(float)

    spot = _spot(levels.index, index_currency, currency, data_provider)

    if hedge_ratio is None:
        return levels * spot

    return _hedged(levels, spot, index_currency, currency, data_provider, hedge_ratio,
                   cash_rate)


def class_tracking(book: ShareClassBook,
                   benchmark: pd.Series,
                   periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> dict[str, float | None]:
    """A class's total return, and its tracking against its benchmark.

    Defined as a backtest's tracking is: the tracking difference is the
    class's total return per unit, with what it paid out added back, less the
    benchmark's over the run; the tracking error is the standard deviation of
    the two's daily return difference on their common dates, annualised.

    Args:
        book: The class's record of the run.
        benchmark: Its benchmark's levels, in its currency (see
            :func:`index_in_class_currency`).
        periods_per_year: How many daily returns make a year.

    Returns:
        dict: total_return, tracking_difference and tracking_error; the last
        two None without a benchmark.
    """
    total_return = book.total_return()

    if benchmark.empty:
        return {"total_return": total_return, "tracking_difference": None,
                "tracking_error": None}

    # The benchmark's first day returns nothing from its starting level, as
    # the class's first return runs from its launch price.
    benchmark_returns = formulas.returns_from_start(benchmark)
    active = (book.get_returns() - benchmark_returns).dropna()
    benchmark_total = formulas.total_return(float(benchmark.iloc[0]), float(benchmark.iloc[-1]))
    tracking_error = formulas.tracking_error(active, periods_per_year)

    return {"total_return": total_return,
            "tracking_difference": total_return - benchmark_total,
            "tracking_error": tracking_error}


def _spot(days: pd.Index,
          source: str,
          target: str,
          data_provider: DataFetcher) -> pd.Series:
    """The rate from *source* into *target* on each day, carried over gaps."""
    rates = data_provider.fx_rates_on(source, target, days)

    if rates is None or rates.empty or pd.isna(rates.iloc[0]):
        raise CalculationError("ShareClasses",
                               f"no {source} to {target} rate on "
                               f"{pd.Timestamp(days[0]).date()} for the index in the "
                               f"class's currency.")

    return rates.ffill().astype(float)


@dataclass(frozen=True)
class _Rates:
    """How a pair's forwards are priced: the two currencies, where their
    short rates come from, and the rate for one with none."""
    index_currency: str
    currency: str
    data_provider: DataFetcher
    cash_rate: float | None

    def forward(self,
                spot: float,
                day: pd.Timestamp,
                expiry: pd.Timestamp) -> float:
        """The forward rate to *expiry*, class currency per unit of the
        index's, at covered interest parity."""
        days = max((expiry - day).days, 0)

        return spot * ((1.0 + self.interest(self.currency, day, days))
                       / (1.0 + self.interest(self.index_currency, day, days)))

    def interest(self,
                 currency: str,
                 day: pd.Timestamp,
                 days: int) -> float:
        """Interest on one unit of *currency* over *days* from *day*."""
        return _interest(currency, day, days, self.data_provider, self.cash_rate)


@dataclass(frozen=True)
class _Month:
    """One month's hedge: struck at its first session on the level there."""
    base: float
    level: float
    spot: float
    forward: float
    expiry: pd.Timestamp
    month: tuple[int, int]

    @classmethod
    def struck(cls,
               base: float,
               level: float,
               spot: float,
               day: pd.Timestamp,
               rates: _Rates) -> "_Month":
        """A month's hedge struck on *day*, to the first business day of the
        next month."""
        expiry = pd.Timestamp(day + pd.offsets.BMonthBegin(1))

        return cls(base, level, spot, rates.forward(spot, day, expiry), expiry,
                   (day.year, day.month))

    def level_on(self,
                 day: pd.Timestamp,
                 level: float,
                 spot: float,
                 hedge_ratio: float,
                 rates: _Rates) -> float:
        """The hedged level on *day*: the month's base moved by the index's
        return in the class's currency and the forward's gain."""
        days = max((self.expiry - day).days, 0)
        current = rates.forward(spot, day, self.expiry)
        discount = 1.0 + rates.interest(rates.currency, day, days)
        converted = (level * spot) / (self.level * self.spot) - 1.0
        hedge = hedge_ratio * (self.forward - current) / (self.spot * discount)

        return self.base * (1.0 + converted + hedge)


def _hedged(levels: pd.Series,
            spot: pd.Series,
            index_currency: str,
            currency: str,
            data_provider: DataFetcher,
            hedge_ratio: float,
            cash_rate: float | None) -> pd.Series:
    """The index hedged into *currency*, struck again at the first session of
    each month."""
    rates = _Rates(index_currency, currency, data_provider, cash_rate)
    values: list[float] = []
    month: _Month | None = None

    for date, level in levels.items():
        day = pd.Timestamp(date)
        rate = float(spot.loc[date])

        if month is None:
            month = _Month.struck(float(level), float(level), rate, day, rates)
            values.append(month.base)
            continue

        value = month.level_on(day, float(level), rate, hedge_ratio, rates)
        values.append(value)

        # The month's hedge settles on the next month's first session, which
        # is where the next one is struck, on the level it settled at.
        if (day.year, day.month) != month.month:
            month = _Month.struck(value, float(level), rate, day, rates)

    return pd.Series(values, index=levels.index, dtype=float)


def _interest(currency: str,
              day: pd.Timestamp,
              days: int,
              data_provider: DataFetcher,
              cash_rate: float | None) -> float:
    """Interest on one unit of *currency* over *days* from *day*, as the
    class hedge reckons it."""
    described = data_provider.short_rate(currency)

    if described is not None:
        rate = data_provider.short_rate_on(currency, day)

        if rate is not None:
            return float(rate) * days / float(described.day_count)

    return (cash_rate or 0.0) * days / FORWARD_DAY_COUNT
