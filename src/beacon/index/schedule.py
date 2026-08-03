# src/beacon/index/schedule.py
"""
When an index rebalances, and which days it has a level on.

`get_rebalance_dates()` used to answer one question one way: the first business
day of every Nth month, where a business day is Monday to Friday. That makes
Christmas Day a trading day and makes "third Friday of March, June, September
and December" — the S&P and FTSE convention — inexpressible. There was a TODO
saying so.

## Two things, kept apart

A **calendar** says which days exist. A **day rule** says which of those days
within a month is the one. Every schedule here is the product of the two, so
"third Friday, on the New York calendar" needs no special case: the third
Friday is found, and if it is not a session it rolls back to the one before.

## Rolling back, not forward

A rebalance landing on a holiday moves to the *previous* session. Forward would
push it into the next month at a month end, which is the one case where the
choice is visible — and the convention every index provider follows is back.
Good Friday is the case that makes this concrete: the third Friday of April
2025 is the 18th, which is not a session on any US exchange.

## The default is exactly what it was

An index that names no calendar and no day rule gets business days and the
first of the month, which is the behaviour every stored index was defined
against. That is not a nicety: changing it would silently redate every existing
backtest, so a test pins the old algorithm against the new one.

## Why the calendar is optional but not silently so

`exchange_calendars` is an extra, not a core dependency. But an index that
*declares* a calendar and runs without the package must not quietly fall back
to Monday-to-Friday: two installations would then compute different indices
from the same definition, and nothing would say which was which. So declaring
a calendar makes the package required, and its absence is an error naming the
extra to install.
"""
import logging

import pandas as pd

from .._optional import require

logger = logging.getLogger(__name__)

# Months between rebalances, by the cadence a definition names.
FREQUENCY_MONTHS = {
    "MONTHLY": 1,
    "QUARTERLY": 3,
    "SEMI-ANNUAL": 6,
    "ANNUAL": 12,
}
FREQUENCIES = tuple(FREQUENCY_MONTHS)

# Which day within a scheduled month the rebalance falls on.
FIRST_BUSINESS_DAY = "FIRST_BUSINESS_DAY"
LAST_BUSINESS_DAY = "LAST_BUSINESS_DAY"
THIRD_FRIDAY = "THIRD_FRIDAY"
DAY_RULES = (FIRST_BUSINESS_DAY, LAST_BUSINESS_DAY, THIRD_FRIDAY)

# The behaviour every index defined before this module was written.
DEFAULT_DAY_RULE = FIRST_BUSINESS_DAY

FRIDAY = 4
THIRD_OCCURRENCE = 3

# How far past the as-of date to look when hunting the next rebalance. Two full
# periods plus a month: enough that an annual schedule is always found, without
# generating a decade of dates to return one of them.
_LOOKAHEAD_PERIODS = 2


def sessions(start: pd.Timestamp,
             end: pd.Timestamp,
             calendar: str | None = None) -> pd.DatetimeIndex:
    """The days an index has a level on, over a range.

    Args:
        start: First date, inclusive.
        end: Last date, inclusive.
        calendar: Exchange MIC, e.g. ``"XNYS"``. None means Monday to Friday,
            which is what every index defined before calendars existed used.

    Returns:
        pd.DatetimeIndex: Sessions in ascending order.

    Raises:
        MissingDependencyError: If a calendar is named and
            ``exchange_calendars`` is not installed. Deliberately an error
            rather than a fallback — see the module docstring.
    """
    if calendar is None:
        return pd.bdate_range(start, end)

    require("exchange_calendars", f"The {calendar} trading calendar")

    import exchange_calendars  # noqa: PLC0415

    schedule = exchange_calendars.get_calendar(calendar)

    # Clamped to what the calendar knows: asking outside its bounds raises,
    # and a date range running past the published holidays is a normal thing
    # for a caller to ask for rather than a mistake.
    first = max(pd.Timestamp(start), schedule.first_session)
    last = min(pd.Timestamp(end), schedule.last_session)

    if first > last:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(schedule.sessions_in_range(first, last))


def is_known_calendar(calendar: str) -> bool:
    """Whether a MIC names a calendar this installation can use."""
    try:
        require("exchange_calendars", f"The {calendar} trading calendar")

        import exchange_calendars  # noqa: PLC0415
    except Exception:
        return False

    return bool(exchange_calendars.get_calendar_names()
                and calendar in set(exchange_calendars.get_calendar_names()))


def _roll_back(target: pd.Timestamp,
               available: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The latest session on or before a target date."""
    earlier = available[available <= target]

    return pd.Timestamp(earlier[-1]) if len(earlier) else None


def _calendar_third_friday(year: int,
                           month: int) -> pd.Timestamp | None:
    """The third Friday of a month by the calendar, holidays ignored."""
    days = pd.date_range(f"{year}-{month:02d}-01",
                         periods=31, freq="D")
    fridays = days[(days.month == month) & (days.dayofweek == FRIDAY)]

    if len(fridays) < THIRD_OCCURRENCE:
        return None

    return pd.Timestamp(fridays[THIRD_OCCURRENCE - 1])


def day_in_month(year: int,
                 month: int,
                 day_rule: str,
                 available: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The scheduled day within one month, or None if it has no sessions.

    Args:
        year: Calendar year.
        month: Calendar month, 1-12.
        day_rule: One of DAY_RULES.
        available: Sessions covering at least this month.

    Returns:
        The date, or None when the month holds no session at all.
    """
    within = available[(available.year == year) & (available.month == month)]
    if not len(within):
        return None

    if day_rule == FIRST_BUSINESS_DAY:
        return pd.Timestamp(within[0])

    if day_rule == LAST_BUSINESS_DAY:
        return pd.Timestamp(within[-1])

    if day_rule == THIRD_FRIDAY:
        # The third Friday of the *calendar*, then rolled back to a session —
        # never the third Friday that happens to be open. Those differ whenever
        # a mid-month Friday is a holiday: April 2025 has Fridays on the 4th,
        # 11th, 18th and 25th, and the 18th is Good Friday. Counting open
        # Fridays lands on the 25th, a week late and the fourth Friday of the
        # month; rolling back lands on Thursday the 17th, which is what an
        # index provider does.
        target = _calendar_third_friday(year, month)
        if target is None:
            return None

        return _roll_back(target, within)

    raise ValueError(
        f"Unsupported day rule: '{day_rule}'. Supported: {', '.join(DAY_RULES)}.")


def rebalance_dates(frequency: str,
                    start: str | pd.Timestamp,
                    end: str | pd.Timestamp,
                    day_rule: str = DEFAULT_DAY_RULE,
                    calendar: str | None = None) -> list[pd.Timestamp]:
    """Every rebalance date in a range.

    Args:
        frequency: One of FREQUENCIES.
        start: First date of the range, inclusive.
        end: Last date, inclusive.
        day_rule: Which day within a scheduled month.
        calendar: Exchange MIC, or None for business days.

    Returns:
        list: Dates in ascending order, empty when the range holds none.

    Raises:
        ValueError: If the frequency or day rule is unknown.
    """
    if frequency not in FREQUENCY_MONTHS:
        raise ValueError(
            f"Unsupported rebalancing frequency: '{frequency}'. "
            f"Supported values: {list(FREQUENCIES)}")

    if day_rule not in DAY_RULES:
        raise ValueError(
            f"Unsupported day rule: '{day_rule}'. Supported: {', '.join(DAY_RULES)}.")

    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    if first > last:
        return []

    # Widened by a month at each end so a month-end rule can see the whole of
    # the boundary months rather than the part inside the range.
    available = sessions(first - pd.offsets.MonthBegin(1),
                         last + pd.offsets.MonthEnd(1),
                         calendar)
    if not len(available):
        return []

    candidates = _monthly_candidates(first, last, day_rule, available)

    return _at_interval(candidates, FREQUENCY_MONTHS[frequency])


def _monthly_candidates(first: pd.Timestamp,
                        last: pd.Timestamp,
                        day_rule: str,
                        available: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """The scheduled day of every month touching the range, inside it."""
    months = pd.date_range(first - pd.offsets.MonthBegin(1),
                           last + pd.offsets.MonthEnd(1), freq="MS")

    found = []
    for month in months:
        date = day_in_month(month.year, month.month, day_rule, available)
        if date is not None and first <= date <= last:
            found.append(date)

    return found


def _at_interval(candidates: list[pd.Timestamp],
                 interval_months: int) -> list[pd.Timestamp]:
    """Thin monthly candidates down to the cadence.

    Anchored on the first candidate rather than on the calendar year, which is
    the behaviour the previous implementation had: a quarterly index starting
    in February rebalances in February, May, August and November, not in the
    March/June/September/December of a calendar quarter.
    """
    if not candidates:
        return []

    kept = [candidates[0]]
    for date in candidates[1:]:
        elapsed = ((date.year - kept[-1].year) * 12
                   + (date.month - kept[-1].month))
        if elapsed >= interval_months:
            kept.append(date)

    return kept


def effective_date(announced: pd.Timestamp,
                   lag_sessions: int,
                   available: pd.DatetimeIndex) -> pd.Timestamp:
    """The date an announced rebalance takes effect.

    Args:
        announced: When the composition was published.
        lag_sessions: Sessions to wait. Zero means same-day, which is what
            every index did before BN-126.
        available: Sessions covering the announcement and the lag.

    Returns:
        The effective date. The announcement itself when the lag is zero, or
        when the panel holds too few sessions after it — an index whose data
        ends mid-lag should apply its last rebalance rather than drop it.
    """
    if lag_sessions <= 0:
        return announced

    later = available[available >= announced]
    if len(later) <= lag_sessions:
        logger.warning(
            "Only %d session(s) after %s, fewer than the %d-session lag; "
            "applying the rebalance on its announcement date.",
            max(len(later) - 1, 0), announced.date(), lag_sessions)

        return announced

    return pd.Timestamp(later[lag_sessions])


def next_rebalance(frequency: str,
                   base_date: str | pd.Timestamp,
                   as_of: str | pd.Timestamp,
                   day_rule: str = DEFAULT_DAY_RULE,
                   calendar: str | None = None) -> pd.Timestamp | None:
    """The first rebalance strictly after a date.

    Anchored on the base date, because that is what the calculator anchors on:
    a "next rebalance" computed from any other origin could name a date the
    index would never actually rebalance on.

    Args:
        frequency: One of FREQUENCIES.
        base_date: The index's base date, which anchors the cadence.
        as_of: The date being asked from.
        day_rule: Which day within a scheduled month.
        calendar: Exchange MIC, or None for business days.

    Returns:
        The next date, or None if none falls within the lookahead.
    """
    anchor = pd.Timestamp(base_date)
    today = pd.Timestamp(as_of)

    if today < anchor:
        today = anchor - pd.Timedelta(days=1)

    horizon = today + pd.DateOffset(
        months=FREQUENCY_MONTHS[frequency] * _LOOKAHEAD_PERIODS + 1)

    upcoming = [date for date in rebalance_dates(frequency, anchor, horizon,
                                                 day_rule, calendar)
                if date > today]

    if not upcoming:
        logger.warning("No rebalance found between %s and %s for a %s schedule.",
                       today.date(), horizon.date(), frequency)

        return None

    return upcoming[0]
