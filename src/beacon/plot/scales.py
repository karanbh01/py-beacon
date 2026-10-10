# src/beacon/plot/scales.py
"""
Where a chart's measuring axes end, and what they label.

Every axis that measures something (a level, a return, a date) runs from the
last tick at or below its data to the first at or above it, so its line stops
where its scale does rather than trailing on past the last number. An axis of
names (the constituents of a bar chart), a heatmap's fixed ticks, a colour
bar and the mark's own axes are left as they are.

Where both axes of a chart measure, their first ticks meet at the corner, and
the lowest label on the y axis would sit on top of the first on the x axis,
so that one label is left blank.
"""
import math
from typing import Any

from .._optional import require

require("matplotlib", "Charting")

import matplotlib  # noqa: E402
from matplotlib import dates  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.category import StrCategoryConverter  # noqa: E402
from matplotlib.ticker import FixedLocator, Formatter  # noqa: E402

# The label on the axes that holds the beta mark (beacon.plot.frame), whose
# axes are left alone.
MARK_LABEL = "beacon-mark"

# How many times a date axis is re-ended before giving up: each pass can move
# the locator to a different interval, and two passes settle every case seen.
DATE_PASSES = 3


def end_on_ticks(ax: Axes) -> None:
    """End each of an axes' measuring axes on its outermost ticks, and blank
    the lowest y label where both axes measure."""
    if ax.get_label() == MARK_LABEL or hasattr(ax, "_colorbar"):
        return

    x_ended = _end(ax.xaxis, ax.get_xlim, ax.set_xlim)
    y_ended = _end(ax.yaxis, ax.get_ylim, ax.set_ylim)

    if x_ended and y_ended:
        low = min(ax.get_ylim())
        ax.yaxis.set_major_formatter(SkipFirst(ax.yaxis.get_major_formatter(), low))


class SkipFirst(Formatter):
    """A formatter that leaves the label at one value blank and formats every
    other label as the formatter it wraps would."""

    def __init__(self,
                 inner: Formatter,
                 blank: float):
        self.inner = inner
        self.blank = blank

    def set_axis(self, axis: Any) -> None:
        super().set_axis(axis)
        self.inner.set_axis(axis)

    def set_locs(self, locs: Any) -> None:
        super().set_locs(locs)
        self.inner.set_locs(locs)

    def __call__(self,
                 x: float,
                 pos: int | None = None) -> str:
        if math.isclose(x, self.blank, rel_tol=1e-9, abs_tol=1e-12):
            return ""

        return str(self.inner(x, pos))

    def get_offset(self) -> str:
        return str(self.inner.get_offset())


def _end(axis: Any,
         limits: Any,
         set_limits: Any) -> bool:
    """End one axis on its ticks. Returns whether it was a measuring axis."""
    locator = axis.get_major_locator()
    low, high = sorted(limits())

    if (isinstance(axis.get_converter(), StrCategoryConverter)
            or isinstance(locator, FixedLocator) or not high > low):
        return False

    if isinstance(locator, dates.DateLocator):
        return _end_dates(axis, locator, low, high, set_limits)

    # The limits the locator itself rounds to: the same tick step the axis
    # shows, so the ends are ticks the reader sees. Widening the range first
    # and picking ticks from it can switch the locator to a coarser step and
    # end the axis between two of the ticks drawn.
    with matplotlib.rc_context({"axes.autolimit_mode": "round_numbers"}):
        ends = locator.view_limits(low, high)

    set_limits(float(ends[0]), float(ends[1]))

    return True


def _end_dates(axis: Any,
               locator: Any,
               low: float,
               high: float,
               set_limits: Any) -> bool:
    """End a date axis on its ticks, checking that the ticks drawn on the new
    range still include both ends."""
    # num2date is untyped in matplotlib's stubs.
    to_date: Any = dates.num2date
    span = high - low

    for _ in range(DATE_PASSES):
        ticks = list(locator.tick_values(to_date(low - span * 0.1),
                                         to_date(high + span * 0.1)))
        below = [tick for tick in ticks if tick <= low + span * 1e-9]
        above = [tick for tick in ticks if tick >= high - span * 1e-9]

        if not below or not above:
            return False

        low, high = float(max(below)), float(min(above))
        set_limits(low, high)

        shown = list(locator.tick_values(to_date(low), to_date(high)))
        if _has(shown, low) and _has(shown, high):
            break

    return True


def _has(ticks: list[float],
         value: float) -> bool:
    """Whether *value* is one of *ticks*."""
    return any(math.isclose(tick, value, rel_tol=0, abs_tol=1e-6) for tick in ticks)
