# src/beacon/plot/analysis_charts.py
"""
The chart methods for an `OptimisationResult` and a `RiskModel`, reached
through `result.plot` like the others in `accessors`.
"""
import dataclasses
from typing import Any, Literal

import pandas as pd

from . import style as sizes
from .base import ChartMethods
from .charts import Bar, Heatmap, Scatter
from .charts.base import Chart, Plan
from .design import StyleArgument
from .layout import LegendItem
from .marks import LineMark, PointMark, TextMark

# Correlation is shaded from 0.2 upward: below that the distinction between
# 0.05 and 0.1 is noise on any real estimate, and giving it a fifth of the
# colour range implies a precision the number does not have.
CORRELATION_DOMAIN = (0.2, 1.0)

# How a correlation heatmap is shaded: each pair its own square, or the
# colours blended from cell to cell.
SHADINGS = ("squares", "blended")

# The named portfolios on a frontier, and the Sharpe ratio's label beside the
# tangency, in points before scaling.
NAMED_POINT = 5.0
SHARPE_OFFSET = (8.0, -10.0)


def _given(value: str | None,
           default: str) -> str:
    return default if value is None else value


class OptimisationPlots(ChartMethods):
    """Charts for an `OptimisationResult`."""

    def exposures(self,
                  title: str | None = None,
                  subtitle: str | None = None,
                  notes: str | None = None,
                  style: StyleArgument = None) -> Chart:
        """Active weights as sign-coloured tilts against the index, largest
        first.

        Args:
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.
            style: Settings for this chart alone: a Style, or settings by
                name such as {"mark": False, "scale": 1.2}. None draws in the
                current style (beacon.plot.configure).

        Returns:
            Chart: The chart, drawn when it is used.
        """
        from .accessors import _largest_first

        active = self._result.active_weights
        bars = _largest_first([str(name) for name in active.index],
                              [float(value) for value in active])
        dropped = len(active) - len(bars)

        note = (f"Optimal minus index. Tracking error "
                f"{self._result.tracking_error():.2%}, turnover "
                f"{self._result.turnover():.1%}.")
        if dropped:
            note += f" {dropped} smaller tilt(s) not shown."

        return Bar(bars, percent=True, title=_given(title, "Active weights"),
                   subtitle=_given(subtitle, "The optimal portfolio's tilts from the index"),
                   notes=_given(notes, note), style=style)

    def frontier(self,
                 frontier: Any,
                 risk_free_rate: float | None = None,
                 title: str | None = None,
                 subtitle: str | None = None,
                 notes: str | None = None,
                 style: StyleArgument = None) -> Chart:
        """The efficient frontier, with the named portfolios and the capital
        market line.

        Args:
            frontier: An `EfficientFrontier` over the same universe.
            risk_free_rate: Where the capital market line starts. None uses
                the rate the frontier was traced at, which is the one its
                tangency portfolio is tangent from.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.
            style: Settings for this chart alone: a Style, or settings by
                name such as {"mark": False, "scale": 1.2}. None draws in the
                current style (beacon.plot.configure).

        Returns:
            Chart: The chart, drawn when it is used.
        """
        # BN-259: defaulted to 0.0, so a frontier traced at another rate got a
        # line that was not tangent to it.
        if risk_free_rate is None:
            risk_free_rate = float(getattr(frontier, "risk_free_rate", 0.0))

        return _Frontier(frontier, risk_free_rate,
                         title=_given(title, "Efficient frontier"),
                         subtitle=_given(subtitle, "Expected return against volatility"),
                         notes=_given(notes, "Expected return and volatility, both annualised."),
                         style=style)


class _Frontier(Scatter):
    """The frontier as a curve, its two named portfolios as points, and the
    capital market line from the risk-free rate through the tangency."""

    kind = "frontier"

    def __init__(self,
                 frontier: Any,
                 risk_free_rate: float,
                 title: str,
                 subtitle: str,
                 notes: str,
                 style: StyleArgument):
        self.frontier = frontier
        self.risk_free_rate = risk_free_rate
        named = [frontier.minimum_variance, frontier.tangency]
        points = pd.DataFrame({"Volatility": [point.volatility for point in named],
                               "Expected return": [point.expected_return or 0.0
                                                   for point in named]})
        curve = pd.Series([point.expected_return or 0.0 for point in frontier.points],
                          index=[point.volatility for point in frontier.points],
                          name="Frontier")
        super().__init__(points, x="Volatility", y="Expected return", curve=curve,
                         percent=True, title=title, subtitle=subtitle, notes=notes,
                         style=style)

    def _build(self) -> Plan:
        plan = super()._build()
        frontier = self.frontier
        tangency, minimum = frontier.tangency, frontier.minimum_variance
        # The named portfolios replace the plain points, each in its own colour.
        marks: list[Any] = [mark for mark in plan.marks if not isinstance(mark, PointMark)]
        legend = [LegendItem("Frontier", "palette-1")]
        lows: list[float] = []

        # The capital market line: from the risk-free rate through the
        # tangency portfolio and a little beyond, which is what makes the
        # tangency point look like a tangency rather than an arbitrary dot.
        # The excess return is computed on its own line on purpose. Written
        # inline as `tangency.expected_return or 0.0 - risk_free_rate`, `or`
        # binds looser than `-` and the expression silently becomes the raw
        # return: a line that still looks plausible and is not tangent.
        if tangency.volatility > 0:
            reach = max(point.volatility for point in frontier.points) * 1.05
            excess = (tangency.expected_return or 0.0) - self.risk_free_rate
            slope = excess / tangency.volatility
            marks.insert(0, LineMark((0.0, reach), (self.risk_free_rate,
                                                    self.risk_free_rate + slope * reach),
                                     "text-muted", sizes.scaled(1.0), dashed=True, layer=2.0))
            legend.append(LegendItem("Capital market line", "text-muted", "dashed"))
            lows = [0.0, self.risk_free_rate, self.risk_free_rate + slope * reach]

        for point, name, colour in ((minimum, "Minimum variance", "palette-3"),
                                    (tangency, "Tangency", "palette-2")):
            marks.append(PointMark((point.volatility,), (point.expected_return or 0.0,),
                                   colour, sizes.scaled(NAMED_POINT)))
            legend.append(LegendItem(name, colour, "marker"))

        if tangency.sharpe_ratio is not None:
            marks.append(TextMark(f"Sharpe {tangency.sharpe_ratio:.2f}", tangency.volatility,
                                  tangency.expected_return or 0.0, "palette-2",
                                  sizes.label_size(), weight=400,
                                  dx=sizes.scaled(SHARPE_OFFSET[0]),
                                  dy=sizes.scaled(SHARPE_OFFSET[1]), va="center"))

        description = plan.description
        if lows:
            panel = description.panels[0]
            x, y = panel.x, panel.y
            panel = dataclasses.replace(
                panel, x=dataclasses.replace(x, low=min(x.low, 0.0)),
                y=dataclasses.replace(y, low=min(y.low, *lows), high=max(y.high, *lows)))
            description = dataclasses.replace(description, panels=(panel,))

        return dataclasses.replace(plan, marks=tuple(marks),
                                   description=dataclasses.replace(description,
                                                                   legend=tuple(legend)))


class RiskPlots(ChartMethods):
    """Charts for a `RiskModel`."""

    def correlation(self,
                    shading: str = "squares",
                    title: str | None = None,
                    subtitle: str | None = None,
                    notes: str | None = None,
                    style: StyleArgument = None) -> Chart:
        """The correlation matrix as a heatmap.

        Args:
            shading: "squares" draws each pair as its own cell; "blended"
                runs the colours smoothly from cell to cell, which shows the
                shape of the matrix rather than each value.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.
            style: Settings for this chart alone: a Style, or settings by
                name such as {"mark": False, "scale": 1.2}. None draws in the
                current style (beacon.plot.configure).

        Returns:
            Chart: The chart, drawn when it is used.

        Raises:
            ValueError: If *shading* is not "squares" or "blended".
        """
        if shading not in SHADINGS:
            raise ValueError(f"shading must be one of {', '.join(SHADINGS)}, "
                             f"not {shading!r}")

        low, high = CORRELATION_DOMAIN
        kind: Literal["squares", "blended"] = "blended" if shading == "blended" else "squares"

        return Heatmap(self._result.correlation, colormap="beacon_corr", vmin=low, vmax=high,
                       values=kind == "squares", shading=kind,
                       title=_given(title, "Correlation"),
                       subtitle=_given(subtitle, "Between the assets' returns"),
                       notes=_given(notes, f"Shaded from {low:.0%}; below that the "
                                           f"differences are noise on any real estimate."),
                       style=style)
