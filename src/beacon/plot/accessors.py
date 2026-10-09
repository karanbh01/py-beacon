# src/beacon/plot/accessors.py
"""
The chart methods reached through `result.plot`.

Every method takes an optional `ax=` and returns the `Axes` it drew on, so a
chart composes into a figure the caller laid out rather than insisting on its
own. That is also what keeps the signatures backend-agnostic: nothing here
returns a matplotlib-specific wrapper, so an interactive backend can offer the
same names later without the call sites changing.

Sizes come from `style.FIGSIZE` per kind (a level chart is wide because time
is the long axis, a weights chart tall because names stack) and only apply
when this module creates the figure. A caller passing `ax=` has already decided.

## Where the numbers come from

Nowhere here. Every method reads a result object and draws it; the arithmetic
belongs to the analysis layer and is tested there. The one exception is the
reconciliation total annotated on the contributions chart, which the renderer
recomputes from the bars it actually drew, because an annotation claiming a
total that does not match the bars beside it is worse than no annotation, and
the only way to be sure is to add up what is on the page.
"""
import logging

import pandas as pd

from .._optional import require
from . import style as beacon_style
from .base import ChartMethods
from .drawing import (
    MAX_BARS,
    MINIMUM_BAR,
    _axes,
    _finish,
    _given,
    _ink,
    _mark_last,
    _rebase,
    _reference_at,
    _series,
    _signed_colours,
    _truncate,
)
from .frame import new_figure, provenance, sentences

require("matplotlib", "Charting")

from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

# Imported by name elsewhere (the tests read MAX_BARS and the helpers here).
__all__ = ["MAX_BARS", "MINIMUM_BAR", "AttributionPlots", "BacktestPlots",
           "IndexPlots", "OptimisationPlots", "RiskPlots", "_ink", "_mark_last",
           "_truncate"]


class IndexPlots(ChartMethods):
    """Charts for an `IndexResult`."""

    def level(self,
              benchmark: pd.Series | None = None,
              ax: Axes | None = None,
              label: str = "Index",
              title: str | None = None,
              subtitle: str | None = None,
              notes: str | None = None) -> Axes:
        """The index level over time, rebased to 100.

        Args:
            benchmark: Optional comparison series, drawn subordinate.
            ax: Axes to draw on. A new figure is created when absent.
            label: Legend label for the index.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "level")
        result = self._result
        levels = _rebase(_series(result.index_levels))

        ax.plot(levels.index, levels.to_numpy(), color=_ink(ax, "accent"),
                linewidth=beacon_style.SERIES_WIDTH, label=label)
        _mark_last(ax, levels)
        _reference_at(ax, 100.0)

        if benchmark is not None and len(benchmark):
            rebased = _rebase(_series(benchmark))
            ax.plot(rebased.index, rebased.to_numpy(),
                    color=_ink(ax, "text-secondary"),
                    linewidth=beacon_style.BENCHMARK_WIDTH, label="Benchmark")
            ax.legend(loc="upper left")

        return _finish(ax,
                       _given(title, "Index level"),
                       _given(subtitle, f"{result.index_id}, rebased to 100"),
                       "Level",
                       _given(notes, sentences(
                           provenance(result, "index", levels.index),
                           "Rebased to 100 at the first observation.")))

    def weights(self,
                date: pd.Timestamp | None = None,
                ax: Axes | None = None,
                title: str | None = None,
                subtitle: str | None = None,
                notes: str | None = None) -> Axes:
        """Constituent weights at a rebalance, with cap markers.

        Args:
            date: Which rebalance. Defaults to the latest.
            ax: Axes to draw on.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

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

        note = sentences(provenance(self._result, "weights", pd.Index([when])),
                         f"{dropped} smaller holding(s) not shown." if dropped else "")

        return _finish(ax,
                       _given(title, "Constituent weights"),
                       _given(subtitle, f"{self._result.index_id} at its "
                                        f"{pd.Timestamp(when):%d %B %Y} rebalance"),
                       "",
                       _given(notes, note))


class BacktestPlots(ChartMethods):
    """Charts for a `BacktestResult`."""

    def performance(self,
                    ax: Axes | None = None,
                    title: str | None = None,
                    subtitle: str | None = None,
                    notes: str | None = None) -> Axes:
        """Growth of 100 with a drawdown panel beneath it.

        The two share an x axis and sit in one gridspec, because a drawdown is
        only meaningful against the path that produced it, and reading them
        side by side would mean matching dates by eye.

        Args:
            ax: Ignored for this chart, which owns a two-panel figure. Accepted
                so every method has the same signature.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

        Returns:
            Axes: The upper (level) panel.
        """
        beacon_style.register()

        if ax is not None:
            logger.warning(
                "performance() draws two linked panels and creates its own "
                "figure; the supplied ax is ignored.")

        figure = new_figure(beacon_style.FIGSIZE["performance"])
        grid = figure.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.12)

        upper = figure.add_subplot(grid[0])
        lower = figure.add_subplot(grid[1], sharex=upper)

        # A unit's NAV when money flowed, so a flow is not drawn as a gain
        # (BN-267); the NAV itself otherwise.
        levels = _rebase(_series(self._result.performance_levels()))
        upper.plot(levels.index, levels.to_numpy(), color=_ink(upper, "accent"),
                   linewidth=beacon_style.SERIES_WIDTH)
        _mark_last(upper, levels)
        _reference_at(upper, 100.0)
        upper.tick_params(labelbottom=False)

        drawdown = levels / levels.cummax() - 1.0
        lower.fill_between(drawdown.index, drawdown.to_numpy(), 0.0,
                           color=_ink(lower, "danger"), alpha=0.28, linewidth=0)
        lower.plot(drawdown.index, drawdown.to_numpy(),
                   color=_ink(lower, "danger"), linewidth=1.0)
        lower.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        lower.set_ylabel("Drawdown")
        lower.grid(visible=False)

        return _finish(upper,
                       _given(title, "Growth of 100"),
                       _given(subtitle, "The portfolio, with its drawdown"),
                       "Level",
                       _given(notes, sentences(
                           provenance(self._result, "backtest", levels.index),
                           "Drawdown is the level against its running peak.")))

    def annual_returns(self,
                       ax: Axes | None = None,
                       title: str | None = None,
                       subtitle: str | None = None,
                       notes: str | None = None) -> Axes:
        """Calendar-year returns as signed bars.

        Args:
            ax: Axes to draw on.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

        Returns:
            Axes: What was drawn on.
        """
        ax = _axes(ax, "annual_returns")

        # BN-259: each year ran from its own first close, so every year after
        # the first lost its first day's return.
        returns = self._result.get_annual_returns()

        labels = [str(year) for year in returns.index]
        values = returns.to_numpy(dtype=float)

        ax.bar(labels, values, color=_signed_colours(ax, values), width=0.62)
        ax.axhline(0.0, color=_ink(ax, "border"), linewidth=beacon_style.REFERENCE_WIDTH)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

        return _finish(ax,
                       _given(title, "Annual returns"),
                       _given(subtitle, "The portfolio's return in each calendar year"),
                       "Return",
                       _given(notes, sentences(
                           provenance(self._result, "backtest",
                                      _series(self._result.performance_levels()).index),
                           "Each year runs from the previous year's close, and "
                           "the first from the initial capital.")))


class AttributionPlots(ChartMethods):
    """Charts for an `AttributionResult`."""

    def contributions(self,
                      ax: Axes | None = None,
                      title: str | None = None,
                      subtitle: str | None = None,
                      notes: str | None = None) -> Axes:
        """Per-constituent contributions as diverging bars.

        The drags are stated in the chart's note rather than drawn as bars,
        because they are comparisons against a counterfactual rather than
        terms in the decomposition: adding them to the same total would mix
        two different questions.

        The annotated total is recomputed from the bars actually drawn. An
        annotation claiming a total that does not match what is beside it is
        worse than none, and adding up the page is the only way to be sure.

        Args:
            ax: Axes to draw on.
            title: The heading. Defaults to the chart's own.
            subtitle: Beneath the heading. Defaults to what the chart shows.
            notes: The notes line, after "Notes:". Defaults to the data's
                source and dates and what the chart needs said.

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
                   linewidth=beacon_style.REFERENCE_WIDTH)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        # A residual inside the result's own reconciliation tolerance is
        # rounding, and printed as a number it was noise that changed with the
        # numpy build: "5.4e-15" on one machine, "5.6e-15" on another, which
        # failed the chart comparison on nothing but the last digit (BN-222).
        # So it reads 0 when the result says it reconciles, and the number
        # only when there is something left over worth seeing.
        residual = "0" if result.reconciles() else f"{result.residual:.1e}"

        drawn = sum(values)
        note = (f"Contributions shown sum to {drawn:.2%}"
                f"{f'; {dropped} smaller not shown' if dropped else ''}"
                f". Total return {result.total_return:.2%}, "
                f"residual {residual}.")

        drags = [(name, value) for name, value in
                 (("Cap drag", result.cap_drag), ("Cost drag", result.cost_drag))
                 if value is not None]
        if drags:
            note += ("  " + "  ".join(f"{name} {value:.2%}"
                                      for name, value in drags))

        return _finish(ax,
                       _given(title, "Contribution to return"),
                       _given(subtitle, "Each constituent's share of the return"),
                       "",
                       _given(notes, note))


# The optimiser and risk charts live in their own module; the lazy accessor
# looks every chart class up here.
from .analysis_charts import OptimisationPlots, RiskPlots  # noqa: E402
