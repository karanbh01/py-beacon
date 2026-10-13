# src/beacon/plot/dateticks.py
"""
Where a date axis puts its ticks, and what they say.

Dates are numbers here, as every renderer can take them: days since
1970-01-01, the epoch matplotlib also counts from. The ticks fall on whole
years, months or days, at an interval that gives the axis between five and a
dozen of them, and each says only what changed since the last: the year at
each January and the month between, or the day within a month.
"""
# BN-298: ports of matplotlib 3.10's AutoDateLocator (with interval multiples)
# and ConciseDateFormatter, checked against them in tests/test_plot_ticks.py.
import datetime
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from dateutil.relativedelta import relativedelta

EPOCH = datetime.datetime(1970, 1, 1)
DAY = datetime.timedelta(days=1)

YEARLY, MONTHLY, DAILY, HOURLY = 0, 1, 2, 3

# The fewest ticks a frequency is chosen for, and the most at each.
LEAST = 5
MOST = {YEARLY: 11, MONTHLY: 12, DAILY: 11, HOURLY: 12}

# The fewest ticks an axis short of room is given.
FEWEST = 3

# The intervals each frequency may take. Days skip 3 for 4, so a 31-day month
# does not end with ticks on the 31st and the 1st side by side.
INTERVALS = {
    YEARLY: (1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500, 1000, 2000, 4000,
             5000, 10000),
    MONTHLY: (1, 2, 3, 4, 6),
    DAILY: (1, 2, 4, 7, 14),
    HOURLY: (1, 2, 3, 4, 6, 12),
}

# How many times an axis is re-ended before giving up: each pass can move the
# ticks to a different interval, and two passes settle every case seen.
PASSES = 3

# The label formats at each level (years, months, days, hours), for a tick
# that starts the next level up, and for the offset written once at the end.
FORMATS = ("%Y", "%b", "%d", "%H:%M", "%H:%M", "%S.%f")
ZERO_FORMATS = ("", "%Y", "%b", "%b-%d", "%H:%M", "%H:%M")
OFFSET_FORMATS = ("", "%Y", "%Y-%b", "%Y-%b-%d", "%Y-%b-%d", "%Y-%b-%d %H:%M")
STARTS = (0, 1, 1, 0, 0, 0)


def to_number(when: datetime.datetime) -> float:
    """A date as days since 1970-01-01."""
    return (when.replace(tzinfo=None) - EPOCH) / DAY


def to_date(number: float) -> datetime.datetime:
    """Days since 1970-01-01 as a date, to the microsecond."""
    return EPOCH + datetime.timedelta(microseconds=round(number * 86_400_000_000))


def _choose(low: datetime.datetime,
            high: datetime.datetime,
            room: int | None = None) -> tuple[int, int, list[int] | None, list[int] | None]:
    """The frequency and interval for a range, and the months and days of
    the month its ticks may fall on: no more ticks than *room*, where given."""
    delta = relativedelta(high, low)
    years = float(delta.years)
    months = years * 12 + delta.months
    days = (high - low).days
    hours = days * 24 + delta.hours
    counts = (years, months, days, hours)
    allowed: list[list[int] | None] = [None, list(range(1, 13)), list(range(1, 32))]
    chosen: list[list[int] | None] = [None, [1], [1]]
    frequency, interval = HOURLY, 1

    for level, count in enumerate(counts):
        if count < LEAST:
            if level < 3:
                chosen[level] = None
            continue

        most = MOST[level] if room is None else max(min(MOST[level], room), FEWEST)
        interval = next((step for step in INTERVALS[level] if count <= step * (most - 1)),
                        INTERVALS[level][-1])
        frequency = level
        if level in (MONTHLY, DAILY):
            options = allowed[level]
            assert options is not None
            chosen[level] = _every(options, interval, level)
            interval = 1
        break

    return frequency, interval, chosen[1], chosen[2]


def _every(options: list[int],
           interval: int,
           level: int) -> list[int]:
    if level == DAILY and interval == 14:
        return [1, 15]
    if level == DAILY and interval == 7:
        return [1, 8, 15, 22]

    return options[::interval]


def tick_values(low: float,
                high: float,
                room: int | None = None) -> list[float]:
    """Every tick for a range of dates, including the years just beyond its
    ends on a yearly axis.

    Args:
        low: Days since 1970-01-01.
        high: Days since 1970-01-01.
        room: The most ticks the axis has room for; a dozen or so when
            None, as matplotlib allows whatever the axis's length.
    """
    low, high = _nonsingular(low, high)
    start, stop = to_date(low), to_date(high)
    frequency, interval, months, days = _choose(start, stop, room)

    if frequency == YEARLY:
        first = max(int(np.floor(start.year / interval)) * interval, 1)
        last = min(-(-stop.year // interval) * interval, 9999)
        found = [datetime.datetime(year, 1, 1) for year in range(first, last + 1, interval)]
        # matplotlib starts the years from the low end's time of day, so a
        # range starting within a second drops its first year.
        found = found[1:] if start.microsecond else found
    elif frequency == HOURLY:
        found = _hours(start, stop, interval)
    else:
        found = _days(start, stop, months or list(range(1, 13)), days)

    if not found:
        return [low, high]

    return [to_number(when) for when in found]


def _days(start: datetime.datetime,
          stop: datetime.datetime,
          months: list[int],
          days: list[int] | None) -> list[datetime.datetime]:
    """Midnights from start to stop, in the given months and on the given
    days of the month (every day when None)."""
    found = []
    day = datetime.datetime(start.year, start.month, start.day)

    if day < start:
        day += DAY

    while day <= stop:
        if day.month in months and (days is None or day.day in days):
            found.append(day)
        day += DAY

    return found


def _hours(start: datetime.datetime,
           stop: datetime.datetime,
           interval: int) -> list[datetime.datetime]:
    """Whole hours from start to stop, every *interval* hours."""
    hour = start.replace(minute=0, second=0, microsecond=0)
    found = []

    if hour < start:
        hour += datetime.timedelta(hours=1)

    while hour <= stop:
        if hour.hour % interval == 0:
            found.append(hour)
        hour += datetime.timedelta(hours=1)

    return found


def _nonsingular(low: float,
                 high: float) -> tuple[float, float]:
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0.0, 1.0

    low, high = sorted((low, high))
    if low == high:
        return low - 365.25 * 2, high + 365.25 * 2

    return low, high


def ends(low: float,
         high: float,
         room: int | None = None) -> tuple[float, float, bool]:
    """The ticks at or beyond each end of a range of dates, checked so the
    ticks of the ended range still include both ends.

    Returns:
        tuple: The range's new ends, and whether it was ended. A range with no
        tick beyond one of its ends is not, though an earlier pass may already
        have moved it.
    """
    span = high - low

    for _ in range(PASSES):
        ticks = tick_values(low - span * 0.1, high + span * 0.1, room)
        below = [tick for tick in ticks if tick <= low + span * 1e-9]
        above = [tick for tick in ticks if tick >= high - span * 1e-9]

        if not below or not above:
            return low, high, False

        low, high = max(below), min(above)
        shown = tick_values(low, high, room)

        if _has(shown, low) and _has(shown, high):
            break

    return low, high, True


def _has(ticks: list[float],
         value: float) -> bool:
    return any(abs(tick - value) <= 1e-6 for tick in ticks)


@dataclass(frozen=True)
class DateLabels:
    """What a date axis's ticks say, and the offset written once at its end
    (such as "2024-Feb" under ticks that are days)."""

    labels: tuple[str, ...]
    offset: str = ""


def labels(ticks: Sequence[float],
           inverted: bool = False) -> DateLabels:
    """Each tick as only what changed since the last."""
    if not len(ticks):
        return DateLabels(())

    dates = [to_date(tick) for tick in ticks]
    parts = np.array([when.timetuple()[:6] for when in dates])
    show_offset = True
    level = 5

    for candidate in range(5, -1, -1):
        unique = np.unique(parts[:, candidate])
        if len(unique) > 1:
            level = candidate
            if level < 2 and np.any(unique == 1):
                show_offset = False
            break

    written = []
    for when, part in zip(dates, parts, strict=True):
        if level < 5:
            form = ZERO_FORMATS[level] if part[level] == STARTS[level] else FORMATS[level]
        else:
            form = ZERO_FORMATS[level] if when.second == when.microsecond == 0 else FORMATS[level]
        written.append(when.strftime(form))

    if level >= 5:
        written = _trim_zeros(written)

    offset = ""
    if show_offset:
        offset = (dates[0] if inverted else dates[-1]).strftime(OFFSET_FORMATS[level])

    return DateLabels(tuple(written), offset)


def _trim_zeros(written: list[str]) -> list[str]:
    trailing = min((len(text) - len(text.rstrip("0")) for text in written if "." in text),
                   default=None)

    if not trailing:
        return written

    return [text[:-trailing].rstrip(".") if "." in text else text for text in written]
