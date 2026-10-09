# src/beacon/plot/analysis_charts.py
"""
The chart methods for an `OptimisationResult` and a `RiskModel`, reached
through `result.plot` like the others in `accessors`.
"""
from typing import Any

import numpy as np

from .._optional import require
from . import style as beacon_style
from .base import ChartMethods
from .drawing import _axes, _finish, _given, _ink, _signed_colours, _truncate

require("matplotlib", "Charting")

from matplotlib.axes import Axes  # noqa: E402

# Inches kept under a heatmap for its slanted asset names.
SLANTED_LABEL_ROOM = 0.5


class OptimisationPlots(ChartMethods):
    """Charts for an `OptimisationResult`."""

    def exposures(self,
                  ax: Axes | None = None,
                  title: str | None = None,
                  subtitle: str | None = None,
                  notes: str | None = None) -> Axes:
        """Active weights as sign-coloured tilts against the index.

        Args:
            ax: Axes to draw on.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "exposures")

        active = self._result.active_weights.sort_values()
        labels, values, dropped = _truncate([str(name) for name in active.index],
                                            [float(value) for value in active])

        ax.barh(labels, values, color=_signed_colours(ax, values), height=0.68)
        ax.axvline(0.0, color=_ink(ax, "border"),
                   linewidth=beacon_style.REFERENCE_WIDTH)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:+.1%}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        note = (f"Optimal minus index. Tracking error "
                f"{self._result.tracking_error():.2%}, turnover "
                f"{self._result.turnover():.1%}.")
        if dropped:
            note += f" {dropped} smaller tilt(s) not shown."

        return _finish(ax,
                       _given(title, "Active weights"),
                       _given(subtitle, "The optimal portfolio's tilts from the index"),
                       "",
                       _given(notes, note))

    def frontier(self,
                 frontier: Any,
                 risk_free_rate: float | None = None,
                 ax: Axes | None = None,
                 title: str | None = None,
                 subtitle: str | None = None,
                 notes: str | None = None) -> Axes:
        """The efficient frontier, with the named points and the capital line.

        Args:
            frontier: An `EfficientFrontier` over the same universe.
            risk_free_rate: Where the capital market line starts. None uses
                the rate the frontier was traced at, which is the one its
                tangency portfolio is tangent from.
            ax: Axes to draw on.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "frontier")

        # BN-259: defaulted to 0.0, so a frontier traced at another rate got a
        # line that was not tangent to it.
        if risk_free_rate is None:
            risk_free_rate = float(getattr(frontier, "risk_free_rate", 0.0))

        volatilities = [point.volatility for point in frontier.points]
        returns = [point.expected_return or 0.0 for point in frontier.points]

        ax.plot(volatilities, returns, color=_ink(ax, "accent"),
                linewidth=beacon_style.SERIES_WIDTH, zorder=3, label="Frontier")

        tangency = frontier.tangency
        minimum = frontier.minimum_variance

        # The capital market line: from the risk-free rate through the tangency
        # portfolio and a little beyond, which is what makes the tangency point
        # look like a tangency rather than an arbitrary dot.
        #
        # The slope is the tangency Sharpe ratio, and the excess return is
        # computed on its own line on purpose. Written inline as
        # `tangency.expected_return or 0.0 - risk_free_rate`, `or` binds looser
        # than `-` and the expression silently becomes the raw return — a line
        # that still looks plausible and is not tangent to anything.
        if tangency.volatility > 0:
            reach = max(volatilities) * 1.05
            excess = (tangency.expected_return or 0.0) - risk_free_rate
            slope = excess / tangency.volatility

            ax.plot([0.0, reach], [risk_free_rate, risk_free_rate + slope * reach],
                    color=_ink(ax, "text-muted"), linewidth=1.0, linestyle="--",
                    zorder=2, label="Capital market line")

        for point, name, ink in ((minimum, "Minimum variance", "series-2"),
                                 (tangency, "Tangency", "series-3")):
            ax.plot([point.volatility], [point.expected_return or 0.0],
                    marker="o", markersize=7, color=_ink(ax, ink), zorder=5,
                    label=name)

        if tangency.sharpe_ratio is not None:
            ax.annotate(f"Sharpe {tangency.sharpe_ratio:.2f}",
                        xy=(tangency.volatility, tangency.expected_return or 0.0),
                        xytext=(8, -10), textcoords="offset points",
                        fontsize=8, color=_ink(ax, "series-3"), fontweight="bold")

        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.legend(loc="lower right")

        return _finish(ax,
                       _given(title, "Efficient frontier"),
                       _given(subtitle, "Expected return against volatility"),
                       "Expected return",
                       _given(notes, "Volatility on the horizontal axis, both "
                                     "annualised."))


class RiskPlots(ChartMethods):
    """Charts for a `RiskModel`."""

    def correlation(self,
                    ax: Axes | None = None,
                    title: str | None = None,
                    subtitle: str | None = None,
                    notes: str | None = None) -> Axes:
        """The correlation matrix as a heatmap.

        Args:
            ax: Axes to draw on.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "correlation")

        matrix = self._result.correlation
        values = matrix.to_numpy(dtype=float)
        names = [str(label) for label in matrix.index]

        low, high = beacon_style.CORRELATION_DOMAIN
        image = ax.imshow(values, cmap=beacon_style.CORRELATION_COLORMAP,
                          vmin=low, vmax=high, aspect="equal")

        ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(names)), names)
        ax.grid(visible=False)

        figure = ax.get_figure()
        assert figure is not None  # _axes() always attaches one

        bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label("less correlated        more correlated", fontsize=7)
        # matplotlib's stubs type `outline` loosely enough that strict mypy
        # cannot see the method; the attribute is a Spine at runtime.
        outline: Any = bar.outline
        if outline is not None:
            outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, color=_ink(ax, "text-muted"))

        return _finish(ax,
                       _given(title, "Correlation"),
                       _given(subtitle, "Between the assets' returns"),
                       "",
                       _given(notes, f"Shaded from {low:.0%}; below that the "
                                     f"differences are noise on any real "
                                     f"estimate."),
                       # The asset names are slanted, so they hang lower than
                       # a row of dates does; and the colour bar's label is the
                       # chart's right edge, where the mark ends.
                       bottom_room=SLANTED_LABEL_ROOM,
                       reach=bar.ax)
