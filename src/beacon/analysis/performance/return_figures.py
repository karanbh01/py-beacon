# src/beacon/analysis/performance/return_figures.py
"""
A run's return figures: trailing returns and returns to date, its best and
worst periods, how often it gained, its average gain and loss, its longest
runs of winning and losing months, and its mean returns.

Every figure is worked out from daily returns, the first from the run's
starting point. A period's return compounds its days. A trailing return runs
from the close on or before the same date that many months or years before
the last day; a window the run is too short for has none. Returns to date
run from the close before the month, quarter or year began, or from the
run's start when it began within it.
"""
# BN-373.
from dataclasses import dataclass

import pandas as pd

from . import formulas

# The trailing windows, by name: months back from the last day.
TRAILING = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}

# Returns to date, by name, and the calendar period each runs over.
TO_DATE = {"MTD": "M", "QTD": "Q", "YTD": "Y"}

# The periods best and worst are found over.
DAILY, WEEKLY = "daily", "weekly"
EXTREMES = (DAILY, WEEKLY, formulas.MONTHLY, formulas.QUARTERLY, formulas.ANNUAL)

# A first or last period is partial when the run starts this many calendar
# days or more after it begins, or ends as many before it ends: enough to
# pass over a weekend and a holiday at either edge.
_PARTIAL_DAYS = 4


@dataclass(frozen=True)
class WindowReturn:
    """A return over a window ending on the run's last day.

    Attributes:
        start: The window's first day with a return.
        end: Its last day.
        days: Its returns.
        total: Its compounded return.
        annualised: The total as a rate a year, for a window longer than a
            year (3 years and more, or since inception over more than a
            calendar year); None otherwise, as a shorter return annualised
            overstates what was earned.
    """
    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    total: float
    annualised: float | None


@dataclass(frozen=True)
class PeriodReturn:
    """One calendar period's compounded return.

    Attributes:
        period: Its name: a date, ``2024-W05``, ``2024-01``, ``2024 Q1`` or
            ``2024``.
        start: Its first day with a return.
        end: Its last.
        value: Its compounded return.
        partial: Whether the run covered only part of it, at its start or end.
    """
    period: str
    start: pd.Timestamp
    end: pd.Timestamp
    value: float
    partial: bool


@dataclass(frozen=True)
class GainsAndLosses:
    """The average gain and loss over periods of one length.

    Attributes:
        average_gain: The mean of the positive returns; None without any.
        average_loss: The mean of the negative returns; None without any.
        ratio: The average gain over the size of the average loss; None
            without both.
        gains: How many returns were positive.
        losses: How many were negative.
    """
    average_gain: float | None
    average_loss: float | None
    ratio: float | None
    gains: int
    losses: int


@dataclass(frozen=True)
class Streak:
    """A run of consecutive months that all gained, or all lost.

    Attributes:
        months: Its length.
        start: Its first month, ``2024-01``.
        end: Its last month.
        total: Its compounded return.
    """
    months: int
    start: str
    end: str
    total: float


def window_return(returns: pd.Series,
                  start: pd.Timestamp,
                  periods_per_year: int,
                  annualise: bool = False) -> WindowReturn | None:
    """The return over *returns* from *start* (the first day counted) to
    their end, annualised as well when *annualise*; None when there are no
    such days."""
    window = returns.loc[start:]

    if window.empty:
        return None

    total = formulas.compounded_return(window)

    return WindowReturn(start=window.index[0], end=window.index[-1], days=len(window),
                        total=total,
                        annualised=(formulas.annualised_return(total, len(window),
                                                               periods_per_year)
                                    if annualise else None))


def longer_than_a_year(start: pd.Timestamp,
                       end: pd.Timestamp) -> bool:
    """Whether a window from *start* to *end* spans more than a calendar
    year."""
    return bool(pd.Timestamp(end) > pd.Timestamp(start) + pd.DateOffset(years=1))


def trailing_returns(returns: pd.Series,
                     periods_per_year: int) -> dict[str, WindowReturn | None]:
    """The returns over the last 1, 3 and 6 months and 1, 3, 5 and 10 years,
    by name; None for a window longer than the run."""
    if returns.empty:
        return dict.fromkeys(TRAILING)

    last = pd.Timestamp(returns.index[-1])
    trailing: dict[str, WindowReturn | None] = {}

    for name, months in TRAILING.items():
        anchor = last - pd.DateOffset(months=months)
        closed = returns.index[returns.index <= anchor]
        # From the day after the last close on or before the anchor date; a
        # run with no close by then is too short for the window.
        trailing[name] = (window_return(returns.loc[returns.index > closed[-1]],
                                        returns.index[0], periods_per_year,
                                        annualise=months > 12)
                          if len(closed) else None)

    return trailing


def returns_to_date(returns: pd.Series,
                    periods_per_year: int) -> dict[str, WindowReturn | None]:
    """The returns for the month, quarter and year to date, by name."""
    if returns.empty:
        return dict.fromkeys(TO_DATE)

    last = pd.Timestamp(returns.index[-1])
    to_date: dict[str, WindowReturn | None] = {}

    for name, period in TO_DATE.items():
        to_date[name] = window_return(returns, last.to_period(period).start_time,
                                      periods_per_year)

    return to_date


def period_returns(returns: pd.Series,
                   frequency: str) -> list[PeriodReturn]:
    """Each calendar day, week, month, quarter or year's compounded return,
    in order, with whether the run covered all of it."""
    if returns.empty:
        return []

    if frequency == DAILY:
        # Each day is its own period: no grouping, which took most of a
        # second over seven years (BN-373).
        return [PeriodReturn(period=str(day.date()), start=day, end=day, value=float(value),
                             partial=False)
                for day, value in zip(pd.DatetimeIndex(returns.index), returns.tolist(),
                                      strict=True)]

    out = []
    groups = _grouped(returns, frequency)

    for position, (label, period) in enumerate(groups):
        edge = position in (0, len(groups) - 1)
        out.append(PeriodReturn(period=label, start=period.index[0], end=period.index[-1],
                                value=formulas.compounded_return(period),
                                partial=edge and _is_partial(period, frequency)))

    return out


def best_and_worst(returns: pd.Series,
                   frequency: str) -> tuple[PeriodReturn, PeriodReturn] | None:
    """The best and the worst of *frequency*'s periods; None with none. The
    first of equals is taken."""
    periods = period_returns(returns, frequency)

    if not periods:
        return None

    return (max(periods, key=lambda period: period.value),
            min(periods, key=lambda period: period.value))


def positive_share(returns: pd.Series,
                   frequency: str) -> float | None:
    """The share of *frequency*'s periods with a positive return; None with
    none."""
    periods = period_returns(returns, frequency)

    return (sum(period.value > 0 for period in periods) / len(periods)) if periods else None


def gains_and_losses(returns: pd.Series,
                     frequency: str = DAILY) -> GainsAndLosses:
    """The average gain and average loss over *frequency*'s periods, and
    their ratio."""
    values = pd.Series([period.value for period in period_returns(returns, frequency)],
                       dtype=float)
    gains, losses = values[values > 0], values[values < 0]
    average_gain = float(gains.mean()) if len(gains) else None
    average_loss = float(losses.mean()) if len(losses) else None

    return GainsAndLosses(average_gain=average_gain,
                          average_loss=average_loss,
                          ratio=(average_gain / abs(average_loss)
                                 if average_gain is not None and average_loss else None),
                          gains=len(gains),
                          losses=len(losses))


def longest_streaks(returns: pd.Series) -> tuple[Streak | None, Streak | None]:
    """The longest run of months that all gained, and of months that all
    lost; the earlier of equals, and None without such a month. A month
    with no return breaks both."""
    months = period_returns(returns, formulas.MONTHLY)
    best: dict[int, Streak | None] = {1: None, -1: None}
    current: list[PeriodReturn] = []

    for month in [*months, None]:
        sign = 0 if month is None else (month.value > 0) - (month.value < 0)

        if current and (month is None or sign != _sign(current[0])):
            streak = Streak(months=len(current), start=current[0].period,
                            end=current[-1].period,
                            total=formulas.compounded_return(pd.Series(
                                [entry.value for entry in current])))
            held = best[_sign(current[0])]

            if held is None or streak.months > held.months:
                best[_sign(current[0])] = streak

            current = []

        if month is not None and sign != 0:
            current.append(month)

    return best[1], best[-1]


def means(returns: pd.Series,
          periods_per_year: int) -> dict[str, float | None]:
    """The arithmetic and geometric mean daily return, and each a year: the
    arithmetic one times the days a year, the geometric one compounded over
    them, which is the annualised return."""
    if returns.empty:
        return dict.fromkeys(("arithmetic", "geometric", "arithmetic_annual",
                              "geometric_annual"))

    total = formulas.compounded_return(returns)
    arithmetic = float(returns.mean())

    return {"arithmetic": arithmetic,
            "geometric": float((1.0 + total) ** (1.0 / len(returns)) - 1.0),
            "arithmetic_annual": arithmetic * periods_per_year,
            "geometric_annual": formulas.annualised_return(total, len(returns),
                                                           periods_per_year)}


def _sign(period: PeriodReturn) -> int:
    """+1 for a gain, -1 for a loss."""
    return 1 if period.value > 0 else -1


def _grouped(returns: pd.Series,
             frequency: str) -> list[tuple[str, pd.Series]]:
    """*returns* split into calendar weeks (ISO, Monday to Sunday), months,
    quarters or years, each named, in order.

    Raises:
        ValueError: If the frequency is not one of `EXTREMES`.
    """
    if frequency == WEEKLY:
        weeks = pd.DatetimeIndex(returns.index).isocalendar()
        keys = [f"{year}-W{week:02d}" for year, week in zip(weeks["year"], weeks["week"],
                                                            strict=True)]

        return [(str(label), period) for label, period
                in returns.groupby(pd.Index(keys, name="period"), sort=False)]

    if frequency not in EXTREMES:
        raise ValueError(f"Unknown frequency {frequency!r}. Known: {', '.join(EXTREMES)}.")

    return formulas.grouped(returns, frequency)


def _is_partial(period: pd.Series,
                frequency: str) -> bool:
    """Whether the run covered only part of a calendar period, at its start
    or end."""
    if frequency == DAILY:
        return False

    first = pd.Timestamp(period.index[0])
    code = {WEEKLY: "W-SUN", formulas.MONTHLY: "M", formulas.QUARTERLY: "Q",
            formulas.ANNUAL: "Y"}[frequency]
    span = first.to_period(code)
    gap = pd.Timedelta(days=_PARTIAL_DAYS)

    return bool(first - span.start_time >= gap
                or span.end_time.normalize() - pd.Timestamp(period.index[-1]) >= gap)
