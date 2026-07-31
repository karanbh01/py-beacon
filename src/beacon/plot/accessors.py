# src/beacon/plot/accessors.py
"""
The chart methods reached through `result.plot`.

Every method takes an optional `ax=` and returns the `Axes` it drew on, so a
chart composes into a figure the caller laid out rather than insisting on its
own. That is also what keeps the signatures backend-agnostic: nothing here
returns a matplotlib-specific wrapper, so an interactive backend can offer the
same names later without the call sites changing.

Sizes come from `style.FIGSIZE` per kind — a level chart is wide because time
is the long axis, a weights chart tall because names stack — and only apply
when this module creates the figure. A caller passing `ax=` has already decided.

## Where the numbers come from

Nowhere here. Every method reads a result object and draws it; the arithmetic
belongs to the analysis layer and is tested there. The one exception is the
reconciliation total annotated on the contributions chart, which the renderer
recomputes from the bars it actually drew — because an annotation claiming a
total that does not match the bars beside it is worse than no annotation, and
the only way to be sure is to add up what is on the page.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from ..tokens import DARK, LIGHT, colour
from . import style as beacon_style
from .base import ChartMethods

require("matplotlib", "Charting")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

# Bars thinner than this are invisible; a near-zero contribution still needs to
# show that it exists and which way it points.
MINIMUM_BAR = 1e-4

# How many names a weights or contributions chart shows before it stops being
# readable. Beyond this the labels collide and the chart says less than a table.
MAX_BARS = 25


def _axes(ax: Axes | None,
          kind: str) -> Axes:
    """The axes to draw on, creating a figure only when one is not supplied."""
    beacon_style.register()

    if ax is not None:
        return ax

    _, created = plt.subplots(figsize=beacon_style.FIGSIZE.get(kind, (8.0, 5.0)))

    return created


def _mode_of(ax: Axes) -> str:
    """Which style an axes is drawn in, read off its own background.

    Charts need a colour the style does not carry — a marker, a sign — and
    hard-coding one would break in the other mode. Asking the axes is more
    reliable than tracking global state, because a caller may have applied a
    style to this figure alone.
    """
    from matplotlib.colors import to_hex

    figure = ax.get_figure()
    background = to_hex(figure.get_facecolor()) if figure is not None else ""

    return LIGHT if background.lower() == colour("canvas", LIGHT) else DARK


def _ink(ax: Axes,
         name: str) -> str:
    """One token colour, in whichever mode this axes is drawn in."""
    return colour(name, _mode_of(ax))


def _finish(ax: Axes,
            title: str,
            ylabel: str = "",
            note: str = "",
            note_offset: float = -0.14) -> Axes:
    """Apply the shared furniture: title, labels and an optional footnote.

    The note sits in axes coordinates, so how far below the axis it needs to be
    depends on how tall that axis is. A short panel — the drawdown strip is a
    quarter of its figure — needs a larger offset or the note lands on top of
    the tick labels.
    """
    ax.set_title(title, loc="left", pad=12)

    if ylabel:
        ax.set_ylabel(ylabel)

    if note:
        ax.text(0.0, note_offset, note, transform=ax.transAxes,
                fontsize=7, color=_ink(ax, "text-muted"), va="top")

    return ax


def _series(payload: pd.Series) -> pd.Series:
    """A result's series as floats."""
    return payload.astype(float)


def _rebase(series: pd.Series,
            base: float = 100.0) -> pd.Series:
    """A level series rescaled to start at *base*."""
    first = float(series.iloc[0])

    return series / first * base if first else series


def _mark_last(ax: Axes,
               series: pd.Series,
               colour_name: str = "accent") -> None:
    """Dot and label the final value.

    The number a reader actually wants from a level chart is where it ended,
    and making them read it off an axis is a small tax on every glance.
    """
    if series.empty:
        return

    ink = _ink(ax, colour_name)
    ax.plot([series.index[-1]], [series.iloc[-1]], marker="o", markersize=4.5,
            color=ink, zorder=5)
    ax.annotate(f"{series.iloc[-1]:,.1f}",
                xy=(series.index[-1], series.iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=ink, fontweight="bold")


def _signed_colours(ax: Axes,
                    values: Any) -> list[str]:
    """Green for positive, red for negative — the application's own signs."""
    up, down = _ink(ax, "success"), _ink(ax, "danger")

    return [up if float(value) >= 0 else down for value in values]


def _truncate(labels: list[str],
              values: list[float],
              limit: int = MAX_BARS) -> tuple[list[str], list[float], int]:
    """Keep the largest *limit* entries by magnitude, reporting what was cut.

    Silently dropping the tail would make a chart of thirty names look like a
    chart of twenty-five. The count comes back so the caller can say so on the
    page.
    """
    if len(labels) <= limit:
        return labels, values, 0

    order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    kept = sorted(order[:limit], key=lambda i: values[i], reverse=True)

    return ([labels[i] for i in kept], [values[i] for i in kept],
            len(labels) - limit)


class IndexPlots(ChartMethods):
    """Charts for an `IndexResult`."""

    def level(self,
              benchmark: pd.Series | None = None,
              ax: Axes | None = None,
              label: str = "Index") -> Axes:
        """The index level over time, rebased to 100.

        Args:
            benchmark: Optional comparison series, drawn subordinate.
            ax: Axes to draw on. A new figure is created when absent.
            label: Legend label for the index.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "level")
        levels = _rebase(_series(self._result.index_levels))

        ax.plot(levels.index, levels.to_numpy(), color=_ink(ax, "accent"),
                linewidth=beacon_style.SERIES_WIDTH, label=label)
        _mark_last(ax, levels)

        if benchmark is not None and len(benchmark):
            rebased = _rebase(_series(benchmark))
            ax.plot(rebased.index, rebased.to_numpy(),
                    color=_ink(ax, "text-secondary"),
                    linewidth=beacon_style.BENCHMARK_WIDTH, label="Benchmark")
            ax.legend(loc="upper left")

        return _finish(ax, "Index level", "Level",
                       "Rebased to 100 at the first observation.")

    def weights(self,
                date: pd.Timestamp | None = None,
                ax: Axes | None = None) -> Axes:
        """Constituent weights at a rebalance, with cap markers.

        Args:
            date: Which rebalance. Defaults to the latest.
            ax: Axes to draw on.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "weights")

        snapshots = self._result.weight_snapshots
        when = date if date is not None else max(snapshots)
        weights = snapshots[when]

        ordered = sorted(weights.items(), key=lambda item: item[1])
        labels, values, dropped = _truncate([name for name, _ in ordered],
                                            [value for _, value in ordered])

        ax.barh(labels, values, color=_ink(ax, "accent"), height=0.68)

        report = self._result.cap_reports.get(when)
        cap = report.cap if report else None
        if cap is not None:
            ax.axvline(cap, color=_ink(ax, "danger"), linewidth=1.0,
                       linestyle="--", zorder=4)
            ax.annotate(f"cap {cap:.1%}", xy=(cap, len(labels) - 0.5),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=_ink(ax, "danger"), va="top")

        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        note = f"At {pd.Timestamp(when).date()}."
        if dropped:
            note += f" {dropped} smaller holding(s) not shown."

        return _finish(ax, "Constituent weights", "", note)


class BacktestPlots(ChartMethods):
    """Charts for a `BacktestResult`."""

    def performance(self,
                    ax: Axes | None = None) -> Axes:
        """Growth of 100 with a drawdown panel beneath it.

        The two share an x axis and sit in one gridspec, because a drawdown is
        only meaningful against the path that produced it — reading them side
        by side means matching dates by eye.

        Args:
            ax: Ignored for this chart, which owns a two-panel figure. Accepted
                so every method has the same signature.

        Returns:
            Axes: The upper (level) panel.
        """
        beacon_style.register()

        if ax is not None:
            logger.warning(
                "performance() draws two linked panels and creates its own "
                "figure; the supplied ax is ignored.")

        figure = plt.figure(figsize=beacon_style.FIGSIZE["performance"])
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.12)

        upper = figure.add_subplot(grid[0])
        lower = figure.add_subplot(grid[1], sharex=upper)

        levels = _rebase(_series(self._result.portfolio_nav))
        upper.plot(levels.index, levels.to_numpy(), color=_ink(upper, "accent"),
                   linewidth=beacon_style.SERIES_WIDTH)
        _mark_last(upper, levels)
        upper.tick_params(labelbottom=False)

        drawdown = levels / levels.cummax() - 1.0
        lower.fill_between(drawdown.index, drawdown.to_numpy(), 0.0,
                           color=_ink(lower, "danger"), alpha=0.28, linewidth=0)
        lower.plot(drawdown.index, drawdown.to_numpy(),
                   color=_ink(lower, "danger"), linewidth=1.0)
        lower.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        lower.set_ylabel("Drawdown")

        _finish(upper, "Growth of 100", "Level")
        _finish(lower, "", "Drawdown",
                "Drawdown is the level against its running peak.",
                note_offset=-0.42)

        return upper

    def annual_returns(self,
                       ax: Axes | None = None) -> Axes:
        """Calendar-year returns as signed bars.

        Args:
            ax: Axes to draw on.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "annual_returns")

        levels = _series(self._result.portfolio_nav)
        yearly = levels.resample("YE").last()
        opening = levels.resample("YE").first()
        returns = (yearly / opening - 1.0).dropna()

        labels = [str(stamp.year) for stamp in returns.index]
        values = returns.to_numpy(dtype=float)

        ax.bar(labels, values, color=_signed_colours(ax, values), width=0.62)
        ax.axhline(0.0, color=_ink(ax, "border"), linewidth=beacon_style.SPINE_WIDTH)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

        return _finish(ax, "Annual returns", "Return",
                       "Calendar years; a partial first or last year is "
                       "measured over the days present.")


class AttributionPlots(ChartMethods):
    """Charts for an `AttributionResult`."""

    def contributions(self,
                      ax: Axes | None = None) -> Axes:
        """Per-constituent contributions as diverging bars.

        The drags are drawn as their own rows rather than folded into the
        constituents, because they are comparisons against a counterfactual
        rather than terms in the decomposition — adding them to the same total
        would mix two different questions.

        The annotated total is recomputed from the bars actually drawn. An
        annotation claiming a total that does not match what is beside it is
        worse than none, and adding up the page is the only way to be sure.

        Args:
            ax: Axes to draw on.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "contributions")
        result = self._result

        items = sorted(result.contributions, key=lambda item: item.contribution)
        labels, values, dropped = _truncate([item.asset_id for item in items],
                                            [item.contribution for item in items])

        ax.barh(labels, values, color=_signed_colours(ax, values), height=0.68)
        ax.axvline(0.0, color=_ink(ax, "border"),
                   linewidth=beacon_style.SPINE_WIDTH)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        drawn = sum(values)
        note = (f"Contributions shown sum to {drawn:.2%}"
                f"{f'; {dropped} smaller not shown' if dropped else ''}"
                f". Total return {result.total_return:.2%}, "
                f"residual {result.residual:.1e}.")

        drags = [(name, value) for name, value in
                 (("Cap drag", result.cap_drag), ("Cost drag", result.cost_drag))
                 if value is not None]
        if drags:
            note += ("  " + "  ".join(f"{name} {value:.2%}"
                                      for name, value in drags))

        return _finish(ax, "Contribution to return", "", note)


class OptimisationPlots(ChartMethods):
    """Charts for an `OptimisationResult`."""

    def exposures(self,
                  ax: Axes | None = None) -> Axes:
        """Active weights as sign-coloured tilts against the index.

        Args:
            ax: Axes to draw on.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "exposures")

        active = self._result.active_weights.sort_values()
        labels, values, dropped = _truncate([str(name) for name in active.index],
                                            [float(value) for value in active])

        ax.barh(labels, values, color=_signed_colours(ax, values), height=0.68)
        ax.axvline(0.0, color=_ink(ax, "border"),
                   linewidth=beacon_style.SPINE_WIDTH)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:+.1%}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        note = (f"Optimal minus index. Tracking error "
                f"{self._result.tracking_error():.2%}, turnover "
                f"{self._result.turnover():.1%}.")
        if dropped:
            note += f" {dropped} smaller tilt(s) not shown."

        return _finish(ax, "Active weights", "", note)

    def frontier(self,
                 frontier: Any,
                 risk_free_rate: float = 0.0,
                 ax: Axes | None = None) -> Axes:
        """The efficient frontier, with the named points and the capital line.

        Args:
            frontier: An `EfficientFrontier` over the same universe.
            risk_free_rate: Where the capital market line starts.
            ax: Axes to draw on.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "frontier")

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

        return _finish(ax, "Efficient frontier", "Expected return",
                       "Volatility on the horizontal axis, both annualised.")


class RiskPlots(ChartMethods):
    """Charts for a `RiskModel`."""

    def correlation(self,
                    ax: Axes | None = None) -> Axes:
        """The correlation matrix as a heatmap.

        Args:
            ax: Axes to draw on.

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

        return _finish(ax, "Correlation", "",
                       f"Shaded from {low:.0%}; below that the differences are "
                       f"noise on any real estimate.")
