# src/beacon/plot/ticks.py
"""
Where a measuring axis puts its ticks, and what they say.

Worked out without a renderer so every renderer shows the same ticks, by the
rules matplotlib's own defaults follow, so a chart's ticks are the ones it
has always had:

* an axis has as many ticks as its length has room for, about three label
  widths apart across and two label heights apart up, at most ten
* the step is 1, 2, 2.5 or 5 times a power of ten, the smallest that covers
  the range in that many steps
* a number is written with as few decimals as tell its ticks apart, with a
  true minus sign, and a percentage with as many as the range needs

Dates are in beacon.plot.dateticks.
"""
# BN-298: ports of matplotlib 3.10's MaxNLocator, ScalarFormatter and
# PercentFormatter, checked against them in tests/test_plot_ticks.py.
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# The steps an automatic axis may take, as multiples of a power of ten.
STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)

# The most intervals an automatic axis divides into.
MOST_BINS = 9

MINUS = "\N{MINUS SIGN}"

# Numbers at least this many digits long are written from an offset, such
# as "+1e6", rather than in full.
OFFSET_THRESHOLD = 4

# Beyond these powers of ten, numbers are written in scientific notation.
POWER_LIMITS = (-5, 6)


@dataclass(frozen=True)
class Spacing:
    """How a measuring axis spaces its ticks.

    Attributes:
        bins: The most intervals, or None for as many as the axis has room
            for.
        steps: The steps allowed, as multiples of a power of ten.
        least: The fewest ticks shown.
    """

    bins: int | None = None
    steps: tuple[float, ...] = STEPS
    least: int = 2


def room_for(length: float,
             label_size: float,
             across: bool) -> int:
    """How many intervals an axis has room for.

    Args:
        length: The axis's length, in inches.
        label_size: Its tick labels' size, in points.
        across: True for a horizontal axis, whose labels need about three
            times their size; a vertical axis's need two.
    """
    size = label_size * (3 if across else 2)

    if size <= 0:
        return 2**31 - 1

    return int(np.floor(length * 72.0 / size))


def nonsingular(low: float,
                high: float,
                expander: float = 0.001,
                tiny: float = 1e-15) -> tuple[float, float]:
    """A range widened where it is too narrow to divide, in order."""
    if not (np.isfinite(low) and np.isfinite(high)):
        return -expander, expander

    low, high = sorted((float(low), float(high)))
    largest = max(abs(low), abs(high))

    if largest < (1e6 / tiny) * np.finfo(float).tiny:
        return -expander, expander

    if high - low <= largest * tiny:
        if high == 0 and low == 0:
            return -expander, expander
        low -= expander * abs(low)
        high += expander * abs(high)

    return low, high


class _Edge:
    """Integer multiples of a step, allowing for floating-point error."""

    def __init__(self,
                 step: float,
                 offset: float):
        self.step = step
        self.offset = abs(offset)

    def _close(self,
               ms: float,
               edge: float) -> bool:
        if self.offset > 0:
            digits = np.log10(self.offset / self.step)
            tolerance = min(0.4999, max(1e-10, 10 ** (digits - 12)))
        else:
            tolerance = 1e-10

        return bool(abs(ms - edge) < tolerance)

    def le(self, x: float) -> float:
        """The largest n with n * step <= x."""
        whole, part = divmod(x, self.step)

        return whole + 1 if self._close(part / self.step, 1) else whole

    def ge(self, x: float) -> float:
        """The smallest n with n * step >= x."""
        whole, part = divmod(x, self.step)

        return whole if self._close(part / self.step, 0) else whole + 1


def _scale_range(low: float,
                 high: float,
                 bins: int,
                 threshold: float = 100) -> tuple[float, float]:
    span = abs(high - low)
    middle = (high + low) / 2

    if abs(middle) / span < threshold:
        offset = 0.0
    else:
        offset = math.copysign(10 ** (math.log10(abs(middle)) // 1), middle)

    return 10 ** (math.log10(span / bins) // 1), offset


def _steps(spacing: Spacing) -> np.ndarray:
    """The steps allowed, from a tenth of the smallest to ten times the
    second."""
    steps = np.asarray(spacing.steps, dtype=float)

    if steps[0] != 1:
        steps = np.concatenate([[1.0], steps])
    if steps[-1] != 10:
        steps = np.concatenate([steps, [10.0]])

    return np.concatenate([0.1 * steps[:-1], steps, [10 * steps[1]]])


def _raw(low: float,
         high: float,
         bins: int,
         spacing: Spacing,
         round_numbers: bool) -> np.ndarray:
    """Ticks spanning the range, with one beyond each end where the step
    leaves the end between ticks."""
    scale, offset = _scale_range(low, high, bins)
    low_, high_ = low - offset, high - offset
    steps = _steps(spacing) * scale
    raw_step = (high_ - low_) / bins
    large = steps >= raw_step

    if round_numbers:
        floors = (low_ // steps) * steps
        large = large & (floors + steps * bins >= high_)

    first = int(np.nonzero(large)[0][0]) if large.any() else len(steps) - 1
    ticks = np.array([])

    for step in steps[:first + 1][::-1]:
        best = (low_ // step) * step
        edge = _Edge(step, offset)
        ticks = np.arange(edge.le(low_ - best), edge.ge(high_ - best) + 1) * step + best

        if ((ticks <= high_) & (ticks >= low_)).sum() >= spacing.least:
            break

    return ticks + offset


def _bins(spacing: Spacing,
          room: int) -> int:
    if spacing.bins is not None:
        return spacing.bins

    return int(np.clip(room, max(1, spacing.least - 1), MOST_BINS))


def tick_values(low: float,
                high: float,
                spacing: Spacing,
                room: int) -> list[float]:
    """Every tick the axis's locator gives for a range, including any just
    beyond its ends, which a formatter still sees.

    Args:
        low: The range's low end.
        high: Its high end.
        spacing: How the axis spaces its ticks.
        room: From room_for(), used when the spacing sets no bins.
    """
    low, high = nonsingular(low, high, expander=1e-13, tiny=1e-14)

    return [float(tick) for tick in _raw(low, high, _bins(spacing, room), spacing, False)]


def ends(low: float,
         high: float,
         spacing: Spacing,
         room: int) -> tuple[float, float]:
    """The ticks at or beyond each end of a range: where an axis that ends on
    its ticks ends.

    The same step the axis shows, so its ends are ticks the reader sees.
    """
    low, high = nonsingular(low, high, expander=1e-12, tiny=1e-13)
    ticks = _raw(low, high, _bins(spacing, room), spacing, True)

    return float(ticks[0]), float(ticks[-1])


def within(ticks: Sequence[float],
           low: float,
           high: float) -> list[float]:
    """The ticks inside a range, allowing for rounding at its ends."""
    low, high = sorted((low, high))
    slack = (high - low) * 1e-10

    return [tick for tick in ticks if low - slack <= tick <= high + slack]


def fix_minus(text: str) -> str:
    """A true minus sign in place of a hyphen."""
    return text.replace("-", MINUS)


@dataclass(frozen=True)
class Labels:
    """What an axis's ticks say: one label per tick, and any offset written
    once at the axis's end (such as "+1e6")."""

    labels: tuple[str, ...]
    offset: str = ""


def number_labels(ticks: Sequence[float],
                  low: float,
                  high: float) -> Labels:
    """Plain numbers: as few decimals as tell the ticks apart.

    Args:
        ticks: Every tick the locator gave, as tick_values() returns them.
        low: The axis's low end.
        high: Its high end.
    """
    if not len(ticks):
        return Labels(())

    offset = _offset(ticks, low, high)
    magnitude = _magnitude(ticks, low, high, offset)
    form = _format(ticks, low, high, offset, magnitude)
    labels = []

    for tick in ticks:
        value = (tick - offset) / 10.0 ** magnitude
        labels.append(fix_minus(form % (0 if abs(value) < 1e-8 else value)))

    return Labels(tuple(labels), _offset_text(offset, magnitude))


def _shown(ticks: Sequence[float],
           low: float,
           high: float) -> np.ndarray:
    low, high = sorted((low, high))
    values = np.asarray(ticks, dtype=float)

    return values[(low <= values) & (values <= high)]


def _offset(ticks: Sequence[float],
            low: float,
            high: float) -> float:
    shown = _shown(ticks, low, high)

    if not len(shown):
        return 0.0

    least, most = shown.min(), shown.max()
    if least == most or least <= 0 <= most:
        return 0.0

    small, large = sorted([abs(float(least)), abs(float(most))])
    sign = math.copysign(1, least)
    top = np.ceil(math.log10(large))
    power = 1 + next(power for power in itertools.count(top, -1)
                     if small // 10 ** power != large // 10 ** power)

    if (large - small) / 10 ** power <= 1e-2:
        power = 1 + next(power for power in itertools.count(top, -1)
                         if large // 10 ** power - small // 10 ** power > 1)

    if large // 10 ** power >= 10 ** (OFFSET_THRESHOLD - 1):
        return float(sign * (large // 10 ** power) * 10 ** power)

    return 0.0


def _magnitude(ticks: Sequence[float],
               low: float,
               high: float,
               offset: float) -> int:
    low, high = sorted((low, high))
    shown = np.abs(_shown(ticks, low, high))

    if not len(shown):
        return 0

    if offset:
        power = math.floor(math.log10(high - low))
    else:
        largest = shown.max()
        power = 0 if largest == 0 else math.floor(math.log10(largest))

    return power if power <= POWER_LIMITS[0] or power >= POWER_LIMITS[1] else 0


def _format(ticks: Sequence[float],
            low: float,
            high: float,
            offset: float,
            magnitude: int) -> str:
    few = len(ticks) < 2
    values = [*ticks, low, high] if few else list(ticks)
    scaled = (np.asarray(values, dtype=float) - offset) / 10.0 ** magnitude
    spread = float(np.ptp(scaled)) or float(np.max(np.abs(scaled))) or 1.0

    if few:
        scaled = scaled[:-2]

    power = math.floor(math.log10(spread))
    decimals = max(0, 3 - power)
    threshold = 1e-3 * 10 ** power

    while decimals >= 0:
        if np.abs(scaled - np.round(scaled, decimals=decimals)).max() < threshold:
            decimals -= 1
        else:
            break

    return f"%1.{decimals + 1}f"


def _offset_text(offset: float,
                 magnitude: int) -> str:
    if not (offset or magnitude):
        return ""

    written = ""
    if offset:
        written = _short(offset)
        if offset > 0:
            written = "+" + written

    scientific = f"1e{magnitude}" if magnitude else ""

    return fix_minus(scientific + written)


def _short(value: float) -> str:
    """A number as briefly as it can be written exactly, such as 1e6."""
    power = math.floor(math.log10(abs(value)))
    significand = round(value / 10 ** power, 10)
    written = ("%d" if significand % 1 == 0 else "%1.10g") % significand

    return written if power == 0 else f"{written}e{power}"


def percent_labels(ticks: Sequence[float],
                   low: float,
                   high: float,
                   whole: float = 1.0,
                   decimals: int | None = None,
                   symbol: str = "%") -> Labels:
    """Percentages of *whole*, with as many decimals as the axis's range
    needs, or *decimals*."""
    shown = 100.0 * abs(high - low) / whole

    if decimals is None:
        decimals = 0 if shown <= 0 else min(5, max(0, math.ceil(2.0 - math.log10(2.0 * shown))))

    return Labels(tuple(fix_minus(f"{100.0 * tick / whole:0.{decimals}f}{symbol}")
                        for tick in ticks))


def share_labels(ticks: Sequence[float],
                 low: float,
                 high: float) -> Labels:
    """Shares written as percentages, with as few decimals as tell the ticks
    apart: 5%, 10%, 15% rather than 5.0%, 10.0%, 15.0%."""
    written = number_labels([100.0 * tick for tick in ticks], 100.0 * low, 100.0 * high)

    return Labels(tuple(f"{label}%" for label in written.labels), written.offset)
