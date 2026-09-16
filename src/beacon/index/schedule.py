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

## The calendar is required (BN-180)

It used to be optional, and the default was Monday to Friday. That default was
chosen to keep every stored index producing exactly the dates it always had —
which it did, and which was the wrong thing to preserve. A calendar-less index
rebalancing on FIRST_BUSINESS_DAY schedules 1 January, 4 July and 25 December
whenever they fall midweek, and over data that genuinely observes holidays
there is no session on any of them. The schedule and the data held two
different definitions of "a day the index trades", neither wrong alone.

So an `IndexDefinition` now always carries a calendar, `IndexDocument` requires
one on the wire, and stored documents without one are migrated to
`DEFAULT_CALENDAR` by schema version 2. `sessions()` still takes `None` — see
below — but nothing a user stores can reach it.

## Why `exchange_calendars`, and why it is now a core dependency

A required calendar cannot sit behind an extra: the core would import and then
refuse to schedule. So `exchange_calendars` moved into the core dependency set.
The alternatives were checked rather than assumed, and all failed:

* `pandas.tseries.holiday.USFederalHolidayCalendar` is wrong in **both**
  directions. Measured over 2025: it calls Columbus Day (10-13) and Veterans
  Day (11-11) closed when NYSE traded, and calls Good Friday (04-18) and
  01-09 open when NYSE was shut. Good Friday is a recurring exchange closure
  no federal calendar will ever hold, and 01-09 was a one-off day of mourning
  no rule-based calendar can predict.
* `pandas_market_calendars` depends on `exchange-calendars` (verified from its
  wheel metadata). It is a wrapper, not an alternative.
* `holidays` and `workalendar` are national-holiday packages: the same
  structural mismatch as pandas, for the same reason.

The cost is small and was measured: 1.3 MB on disk, and non-core dependencies
of `pyluach`, `toolz` and `korean_lunar_calendar`, all small and pure-Python.

## The `None` calendar survives as a primitive, not as a default

`sessions(start, end, None)` is still `pd.bdate_range`, because plain
business-day arithmetic is a real thing for a library caller to want. What it
no longer is, is reachable by *omission*: `calendar` is a required argument of
`sessions`, `rebalance_dates` and `next_rebalance`, so Monday-to-Friday is
something a caller asks for in writing, never something they get by forgetting.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

import exchange_calendars
import pandas as pd

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

# The calendar an index gets when it names none: the one every index stored
# before BN-180 was implicitly assumed to run on, and what the schema-version-2
# migration writes into a document without one.
DEFAULT_CALENDAR = "XNYS"

FRIDAY = 4
THIRD_OCCURRENCE = 3

# The datetime resolution every session index is returned in: the one
# `pd.bdate_range` uses, so a calendar's sessions and a business-day range are
# interchangeable as the index of a series rather than only as a set of dates.
SESSION_UNIT = pd.bdate_range("2024-01-01", periods=1).unit

# How far past the as-of date to look when hunting the next rebalance. Two full
# periods plus a month: enough that an annual schedule is always found, without
# generating a decade of dates to return one of them.
_LOOKAHEAD_PERIODS = 2


@lru_cache(maxsize=64)
def _built(calendar: str,
           start: str | None) -> exchange_calendars.ExchangeCalendar:
    """One calendar object per (MIC, start), built once and kept.

    `exchange_calendars` caches the *default* calendar and nothing else, so
    every request for an earlier start rebuilds from scratch. Measured before
    this existed: a `sessions()` call that widens costs 407 ms, and one that
    has to bisect for the floor costs 835 ms — against 8.5 ms for a call
    inside the default bounds. `sessions()` is called several times per run and
    `rebalance_dates` calls it again, so a pre-2006 index paid that repeatedly
    for an object that cannot change between calls.

    Bounded rather than unbounded: the keyspace is a handful of MICs times a
    handful of distinct starts, and a calendar object is large enough that an
    unbounded cache over a long-lived server is a leak rather than a saving.
    """
    if start is None:
        return exchange_calendars.get_calendar(calendar)

    return exchange_calendars.get_calendar(calendar, start=start)


def _calendar_covering(calendar: str,
                       start: pd.Timestamp) -> exchange_calendars.ExchangeCalendar:
    """A calendar whose history reaches back to *start*, where it can.

    `get_calendar(code)` builds twenty years of history and one year ahead, and
    the twenty years is the half that bites: `exchange_calendars` knows NYSE
    back to 1885, but the default object starts two decades ago and
    `sessions()` clamps to what the object holds. Before BN-186 nothing noticed
    — only the rebalance schedule read sessions, and no index in this
    repository has a base date that old. The generator does: `--long-history`
    reaches back to 1999, and a silently truncated calendar would have it
    produce seven fewer years than it was asked for.

    Only the **near** end is widened. The far end stays where the package puts
    it, because history is published and the future is extrapolated: a calendar
    asked for 2030 would answer from rules rather than from an exchange's
    notice, and clamping forward is the deliberate behaviour documented on
    :func:`sessions`.

    When the package refuses the earlier start, this falls back to the widest
    calendar it *will* build rather than to the default one (BN-198). The
    difference decides whether a request is partly answerable or not at all:
    XTKS will not build from 1996 but will from 1997, so a 1996-2000 index
    falling back to the default bounds looks like a window the calendar covers
    nothing of — and refuses — when four fifths of it were available. Falling
    back to the floor makes that what it is, a partial cover, which is
    reported rather than refused.
    """
    schedule = _built(calendar, None)

    if start >= schedule.first_session:
        return schedule

    try:
        return _built(calendar, start.strftime("%Y-%m-%d"))
    except Exception:
        floor = earliest_available(calendar)

        logger.warning(
            "%s cannot reach back to %s; its earliest session is %s.",
            calendar, start.date(), floor.date())

        if floor >= schedule.first_session:
            return schedule

        return _built(calendar, floor.strftime("%Y-%m-%d"))


@dataclass(frozen=True)
class CalendarCoverage:
    """What a calendar could offer of the window it was asked for (BN-198).

    The twin-field shape, for the same reason as `as_of`/`resolved_date`:
    a resolution step moved a field, so the unmoved one is published beside
    it. A calendar narrowing a window is a resolution step like any other, and
    a covered range alone cannot say what was asked for.

    `is_partial` exists so a client never has to compare two dates to find
    out — an edge a reader gets right for a while and then does not.
    `trimmed_start` and `trimmed_end` say *which* end moved, because the two
    have different causes and different remedies: the near end is a calendar
    whose history does not reach (change the calendar or the base date), the
    far end is one whose published sessions stop (wait, or ask for less).

    Attributes:
        calendar: The MIC asked for, or None for plain business days.
        requested_start: The window's first date, as asked for.
        requested_end: The window's last date, as asked for.
        covered_start: First date the calendar can speak for, or None when it
            can speak for none of the window.
        covered_end: Last such date, or None in the same case.
    """

    calendar: str | None
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    covered_start: pd.Timestamp | None
    covered_end: pd.Timestamp | None
    #: The calendar's own bounds as resolved for *this* request, which is not
    #: a property of the calendar alone: `_calendar_covering` widens the near
    #: end on demand, so the same MIC answers differently depending on how far
    #: back it was asked to reach. None when no calendar was named.
    calendar_start: pd.Timestamp | None = None
    calendar_end: pd.Timestamp | None = None

    @property
    def is_empty(self) -> bool:
        """Whether the calendar covers none of the requested window."""
        return self.covered_start is None or self.covered_end is None

    @property
    def starts_before_calendar(self) -> bool:
        """Whether the window opens before anything the calendar knows.

        Which end failed decides the remedy, so a refusal has to know: too
        early is a calendar whose history does not reach, too late is one
        whose published sessions stop, and the two are acted on differently.
        """
        return (self.calendar_start is not None
                and self.requested_start < self.calendar_start)

    def describe_window(self) -> str:
        """The requested window beside the covered one, for a log line."""
        if self.is_empty:
            return (f"{self.requested_start:%Y-%m-%d} to "
                    f"{self.requested_end:%Y-%m-%d}, none of which is covered")

        return (f"{self.requested_start:%Y-%m-%d} to "
                f"{self.requested_end:%Y-%m-%d}, covered from "
                f"{self.covered_start:%Y-%m-%d} to {self.covered_end:%Y-%m-%d}")

    @property
    def trimmed_start(self) -> bool:
        """Whether the calendar's history does not reach the requested start."""
        return (self.covered_start is not None
                and self.covered_start > self.requested_start)

    @property
    def trimmed_end(self) -> bool:
        """Whether the calendar's sessions stop before the requested end."""
        return (self.covered_end is not None
                and self.covered_end < self.requested_end)

    @property
    def is_partial(self) -> bool:
        """Whether either end was narrowed. False when the cover is complete."""
        return self.trimmed_start or self.trimmed_end


def calendar_coverage(start: pd.Timestamp,
                      end: pd.Timestamp,
                      calendar: str | None) -> CalendarCoverage:
    """How much of a window a calendar can actually speak for.

    A pure query: it refuses nothing and logs nothing, so the caller decides
    what a narrowing means. :func:`sessions` stays the primitive it was and
    keeps answering with what it has — `is_session` depends on that, since a
    date outside a calendar's bounds is a date it cannot call open — while an
    index calculation asks this first and refuses on an empty answer.

    Args:
        start: First date of the window, inclusive.
        end: Last date, inclusive.
        calendar: Exchange MIC, or None for plain business days, which have no
            bounds and therefore always cover the window in full.

    Returns:
        CalendarCoverage: The requested window beside the covered one.
    """
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)

    if calendar is None:
        return CalendarCoverage(calendar=None,
                                requested_start=requested_start,
                                requested_end=requested_end,
                                covered_start=requested_start,
                                covered_end=requested_end)

    schedule = _calendar_covering(calendar, requested_start)
    calendar_start = pd.Timestamp(schedule.first_session)
    calendar_end = pd.Timestamp(schedule.last_session)

    first = max(requested_start, calendar_start)
    last = min(requested_end, calendar_end)

    # An inverted window is the caller's arithmetic, not the calendar's reach,
    # and must not be reported as a coverage failure -- `rebalance_dates`
    # already answers "nothing" for one and has always been allowed to.
    empty = requested_start > requested_end or first > last

    return CalendarCoverage(calendar=calendar,
                            requested_start=requested_start,
                            requested_end=requested_end,
                            covered_start=None if empty else first,
                            covered_end=None if empty else last,
                            calendar_start=calendar_start,
                            calendar_end=calendar_end)


# Years the earliest-session search brackets between. The floor sits below
# every floor `exchange_calendars` ships — NYSE is the deepest at 1885, so a
# 1900 bracket would report *itself* as the answer for it — and the ceiling is
# the default twenty-year window every calendar builds unasked, which every
# floor is therefore at or below. Widening the bracket costs one bisection
# step, so it is bought cheaply.
_SEARCH_FLOOR = 1800
_SEARCH_CEILING = 2006


@lru_cache(maxsize=64)
def earliest_available(calendar: str) -> pd.Timestamp:
    """The earliest session the package will build for a calendar.

    Found by bisection rather than read off an attribute, because there is no
    attribute: `get_calendar(code)` returns twenty years of history whatever
    the underlying data supports, and the true floor only shows up as the
    earliest `start` it will accept. Each calendar's floor differs and none is
    the default bound — XTKS and XBOM reach back to 1997, XHKG to 1960, while
    all three report a first session of 2006 until asked for more.

    Worth the handful of calendar builds because it is only ever called to
    explain a refusal, and the alternative is a message that understates the
    reach: quoting the default bound would tell someone asking for a 1950 Hong
    Kong index that the calendar starts in 2006, which is both false and the
    wrong thing to act on.

    Args:
        calendar: Exchange MIC.

    Returns:
        pd.Timestamp: First session of the widest calendar obtainable.
    """
    low, high = _SEARCH_FLOOR, _SEARCH_CEILING

    while low < high:
        middle = (low + high) // 2

        try:
            _built(calendar, f"{middle}-01-02")
            high = middle
        except Exception:
            low = middle + 1

    return pd.Timestamp(_built(calendar, f"{low}-01-02").first_session)


def describe_bounds(coverage: CalendarCoverage) -> str:
    """Why a calendar could not cover a window, in terms of the end that failed.

    Only the failing end is searched for. A window that opens before the
    calendar's history needs the true floor, and finding it costs a handful of
    calendar builds; a window that opens after the calendar's last published
    session needs no search at all, because that bound is already known. The
    old version quoted both ends always, which paid for the search on every
    refusal and — for a calendar the package will build arbitrarily far back,
    NYSE being one — reported the search's own lower bracket as though it were
    a fact about the exchange.
    """
    calendar = coverage.calendar
    assert calendar is not None, "a coverage with no calendar cannot refuse"

    name = DISPLAY_NAMES.get(calendar, calendar)

    if coverage.starts_before_calendar:
        return (f"{calendar} ({name}) has no sessions before "
                f"{earliest_available(calendar):%Y-%m-%d}")

    last = coverage.calendar_end

    return (f"{calendar} ({name}) has no published sessions after "
            f"{last:%Y-%m-%d}" if last is not None else
            f"{calendar} ({name}) cannot cover that window")


def sessions(start: pd.Timestamp,
             end: pd.Timestamp,
             calendar: str | None) -> pd.DatetimeIndex:
    """The days an index has a level on, over a range.

    Args:
        start: First date, inclusive.
        end: Last date, inclusive.
        calendar: Exchange MIC, e.g. ``"XNYS"``. Required — passing None asks
            for Monday to Friday, holidays included, which no stored index can
            do since BN-180 and which a library caller must therefore state
            rather than fall into.

    Returns:
        pd.DatetimeIndex: Sessions in ascending order.
    """
    if calendar is None:
        return pd.bdate_range(start, end)

    schedule = _calendar_covering(calendar, pd.Timestamp(start))

    # Clamped to what the calendar knows: asking outside its bounds raises,
    # and a date range running past the published holidays is a normal thing
    # for a caller to ask for rather than a mistake. Only the forward end
    # actually clamps now -- `_calendar_covering` has already reached the
    # start date back, where the package would let it.
    first = max(pd.Timestamp(start), schedule.first_session)
    last = min(pd.Timestamp(end), schedule.last_session)

    if first > last:
        return pd.DatetimeIndex([]).as_unit(SESSION_UNIT)

    # Normalised to the unit `pd.bdate_range` produces. `exchange_calendars`
    # hands back nanoseconds, and once these sessions became the index of a
    # level series (BN-186) the difference stopped being cosmetic: a cached
    # result reloaded from disk compares unequal to a freshly calculated one
    # on dtype alone, every number in it identical.
    return pd.DatetimeIndex(
        schedule.sessions_in_range(first, last)).as_unit(SESSION_UNIT)


def is_session(date: pd.Timestamp,
               calendar: str | None) -> bool:
    """Whether the market was open on *date*.

    The single-day face of :func:`sessions`, and the question that tells a
    holiday apart from a hole in the data (BN-183): a day with no bar that the
    calendar says was **closed** is a market that was shut, while a day with no
    bar that the calendar says was **open** is data that is missing something.
    Nothing could ask it before BN-180 made the calendar a required property of
    an index.

    Args:
        date: The date asked about.
        calendar: Exchange MIC. None asks about Monday to Friday, holidays
            included — the same explicit request :func:`sessions` accepts.

    Returns:
        bool: True when *date* is a trading session on *calendar*. A date past
        the calendar's published bounds answers False, since a day the
        calendar cannot speak for is not a day it says was open.
    """
    return len(sessions(date, date, calendar)) > 0


# Display names for the venues a client is most likely to offer first. Every
# other calendar falls back to its MIC, which is why this is allowed to be
# partial: a name that is missing labels a calendar less well, and a name that
# is wrong labels it falsely. `exchange_calendars` has no friendly names of its
# own -- `.name` returns the MIC -- so the choice is a short curated list or
# none at all.
DISPLAY_NAMES = {
    "XNYS": "New York Stock Exchange",
    "XNAS": "Nasdaq",
    "XLON": "London Stock Exchange",
    "XETR": "Xetra (Frankfurt)",
    "XPAR": "Euronext Paris",
    "XAMS": "Euronext Amsterdam",
    "XBRU": "Euronext Brussels",
    "XMIL": "Borsa Italiana",
    "XSWX": "SIX Swiss Exchange",
    "XMAD": "Bolsa de Madrid",
    "XSTO": "Nasdaq Stockholm",
    "XTKS": "Tokyo Stock Exchange",
    "XHKG": "Hong Kong Stock Exchange",
    "XSHG": "Shanghai Stock Exchange",
    "XSES": "Singapore Exchange",
    "XASX": "Australian Securities Exchange",
    "XTSE": "Toronto Stock Exchange",
    "XBOM": "BSE (Bombay)",
    "XKRX": "Korea Exchange",
    "XJSE": "Johannesburg Stock Exchange",
    "BVMF": "B3 (São Paulo)",
}


def known_calendars() -> list[str]:
    """Every MIC this installation can schedule against, sorted.

    Read from the package rather than listed here, so the set a client is told
    about is the set the calculation actually accepts. A hand-kept copy of a
    hundred-odd MICs would be wrong the first time the package gained one.
    """
    return sorted(exchange_calendars.get_calendar_names())


def calendar_region(calendar: str) -> tuple[str, str]:
    """A calendar's region and IANA timezone, e.g. ``("Europe", "Europe/Oslo")``.

    Derived from the calendar's own timezone rather than from a table, for the
    same reason the code list is: a mapping kept here would be one release
    behind the package the schedule actually runs on. Checked across all 102
    calendars this installation carries -- every one has a timezone, and every
    one splits into a region: Europe 42, America 30, Asia 22, then Australia,
    Atlantic, Africa, Pacific, and two on bare UTC, which report "UTC" as their
    own region rather than being forced into a continent they do not have.
    """
    zone = str(exchange_calendars.get_calendar(calendar).tz)

    return zone.split("/", maxsplit=1)[0], zone


def is_known_calendar(calendar: str) -> bool:
    """Whether a MIC names a calendar this installation can use."""
    return calendar in set(exchange_calendars.get_calendar_names())


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
                    calendar: str | None,
                    day_rule: str = DEFAULT_DAY_RULE) -> list[pd.Timestamp]:
    """Every rebalance date in a range.

    Args:
        frequency: One of FREQUENCIES.
        start: First date of the range, inclusive.
        end: Last date, inclusive.
        calendar: Exchange MIC. Required; None asks explicitly for business
            days, which no stored index does.
        day_rule: Which day within a scheduled month.

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
                   calendar: str | None,
                   day_rule: str = DEFAULT_DAY_RULE) -> pd.Timestamp | None:
    """The first rebalance strictly after a date.

    Anchored on the base date, because that is what the calculator anchors on:
    a "next rebalance" computed from any other origin could name a date the
    index would never actually rebalance on.

    Args:
        frequency: One of FREQUENCIES.
        base_date: The index's base date, which anchors the cadence.
        as_of: The date being asked from.
        calendar: Exchange MIC. Required; None asks explicitly for business
            days.
        day_rule: Which day within a scheduled month.

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
                                                 calendar, day_rule)
                if date > today]

    if not upcoming:
        logger.warning("No rebalance found between %s and %s for a %s schedule.",
                       today.date(), horizon.date(), frequency)

        return None

    return upcoming[0]
