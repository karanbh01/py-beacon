# src/beacon/plot/accessors.py
"""
The chart methods reached through `result.plot`.

Each method builds one of the chart classes in `beacon.plot` from a result
and returns it, so a result's chart has the same options as any other:
`.show()`, `.save(path)`, its `.figure` and `.ax`, and `style=`. A chart is
drawn only when it is used, so these methods need no matplotlib.

## Where the numbers come from

Nowhere here. Every method reads a result object and draws it; the arithmetic
belongs to the analysis layer and is tested there. The one exception is the
reconciliation total in the contributions chart's notes, recomputed from the
bars actually drawn, because a total that does not match the bars beside it
is worse than none.
"""
import dataclasses
import logging

import pandas as pd

from . import design
from .base import ChartMethods
from .charts import Area, Bar, Column, Line
from .charts.base import Chart, Plan
from .charts.grid import Grid
from .design import StyleArgument
from .notes import provenance, sentences
from .style import figure_size
from .ticks import Spacing

logger = logging.getLogger(__name__)

# Bars thinner than this are invisible; a near-zero contribution still needs to
# show that it exists and which way it points.
MINIMUM_BAR = 1e-4

# How many names a weights or contributions chart shows before it stops being
# readable. Beyond this the labels collide and the chart says less than a table.
MAX_BARS = 25

# How many constituents a constituents chart draws. More lines than this are a
# tangle in which no one line can be followed.
MAX_LINES = 8

# A drawdown panel is short, so it has room for few ticks; steps of 2.5% and
# 5% as well as 10% keep it from rounding a 10% drawdown out to 20%.
DRAWDOWN_SPACING = Spacing(bins=4, steps=(1.0, 2.0, 2.5, 5.0, 10.0))

__all__ = ["MAX_BARS", "MINIMUM_BAR", "AttributionPlots", "BacktestPlots", "IndexPlots",
           "OptimisationPlots", "RiskPlots", "_truncate"]


def _given(value: str | None,
           default: str) -> str:
    """A caller's title, subtitle or notes, or the chart's own when None.
    An empty string is kept: it is how a caller turns one off."""
    return default if value is None else value


def _series(payload: pd.Series) -> pd.Series:
    """A result's series as floats."""
    return payload.astype(float)


def _rebase(series: pd.Series,
            base: float = 100.0) -> pd.Series:
    """A level series rescaled to start at *base*."""
    first = float(series.iloc[0])

    return series / first * base if first else series


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


def _largest_first(names: list[str],
                   values: list[float]) -> pd.Series:
    """Bars from the largest at the top, kept to the most a chart can show."""
    names, values, _ = _truncate(names, values)
    ordered = sorted(zip(names, values, strict=True), key=lambda item: item[1], reverse=True)

    return pd.Series([value for _, value in ordered], index=[name for name, _ in ordered])


class IndexPlots(ChartMethods):
    """Charts for an `IndexResult`."""

    def level(self,
              benchmark: pd.Series | None = None,
              label: str = "Index",
              title: str | None = None,
              subtitle: str | None = None,
              notes: str | None = None,
              style: StyleArgument = None) -> Chart:
        """The index level over time, rebased to 100.

        Args:
            benchmark: Optional comparison series, drawn subordinate.
            label: The index's name in the legend.
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
        result = self._result
        levels = _rebase(_series(result.index_levels)).rename(label)
        lines = levels.to_frame()
        colours = ["palette-1"]

        if benchmark is not None and len(benchmark):
            lines["Benchmark"] = _rebase(_series(benchmark)).reindex(levels.index)
            colours.append("text-secondary")

        return Line(lines, reference=100.0, label_last=label, ylabel="Level", colors=colours,
                    title=_given(title, "Index level"),
                    subtitle=_given(subtitle, f"{result.index_id}, rebased to 100"),
                    notes=_given(notes, sentences(provenance(result, "index", levels.index),
                                                  "Rebased to 100 at the first observation.")),
                    style=style)

    def constituents(self,
                     limit: int = MAX_LINES,
                     title: str | None = None,
                     subtitle: str | None = None,
                     notes: str | None = None,
                     style: StyleArgument = None) -> Chart:
        """The index among its constituents, each rebased to 100.

        The constituents are the names held at the latest rebalance, each
        drawn from its close in the data the index was calculated from and
        named at the end of its line. The index is drawn over them.

        Args:
            limit: How many constituents to draw, the largest by weight at
                the latest rebalance. The notes say how many were left out.
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
            ValueError: If the result is bound to no data to read the
                constituents' prices from.
        """
        result = self._result
        fetcher = result._data_fetcher

        if fetcher is None:
            raise ValueError(
                "the constituents' prices come from the data the index was "
                "calculated from, and this result has none bound; call "
                "result.with_data(fetcher) first.")

        levels = _rebase(_series(result.index_levels))
        start, end = levels.index[0], levels.index[-1]
        latest = result.weight_snapshots[max(result.weight_snapshots)]
        names = sorted(latest, key=lambda name: latest[name], reverse=True)
        drawn = names[:limit]
        lines = pd.DataFrame({"Index": levels})

        for name in drawn:
            closes = fetcher.fetch_market_data(name, start, end)["CLOSE"]
            closes = _series(closes).reindex(levels.index).dropna()

            if closes.empty:
                logger.warning("No closes for %s between %s and %s; it is "
                               "left off the constituents chart.",
                               name, start.date(), end.date())
                continue

            lines[name] = _rebase(closes)

        left_out = len(names) - len(drawn)
        note = sentences(
            provenance(result, "index", levels.index),
            "Each constituent's close, rebased to 100 on the index's first day.",
            f"{left_out} smaller constituent(s) not shown." if left_out else "")
        shown = (f"{result.index_id} among its {len(drawn)} largest constituents, rebased to 100"
                 if left_out else f"{result.index_id} among its constituents, rebased to 100")

        return _ConstituentLines(lines, reference=100.0, label_last="Index", end_names=True,
                                 legend=False,
                                 ylabel="Level",
                                 colors=["palette-1", *["text-muted"] * (len(lines.columns) - 1)],
                                 title=_given(title, "The index and its constituents"),
                                 subtitle=_given(subtitle, shown), notes=_given(notes, note),
                                 style=style)

    def weights(self,
                date: pd.Timestamp | None = None,
                title: str | None = None,
                subtitle: str | None = None,
                notes: str | None = None,
                style: StyleArgument = None) -> Chart:
        """Constituent weights at a rebalance, largest first, with the cap.

        Args:
            date: Which rebalance. Defaults to the latest.
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
        snapshots = self._result.weight_snapshots
        when = date if date is not None else max(snapshots)
        weights = snapshots[when]
        names = list(weights)
        bars = _largest_first(names, [float(weights[name]) for name in names])
        dropped = len(names) - len(bars)
        report = self._result.cap_reports.get(when)

        note = sentences(provenance(self._result, "weights", pd.Index([when])),
                         f"{dropped} smaller holding(s) not shown." if dropped else "")

        return Bar(bars, percent=True, cap=report.cap if report else None, signed=False,
                   color="accent", title=_given(title, "Constituent weights"),
                   subtitle=_given(subtitle, f"{self._result.index_id} at its "
                                             f"{pd.Timestamp(when):%d %B %Y} rebalance"),
                   notes=_given(notes, note), style=style)


class _ConstituentLines(Line):
    """A level chart with its constituents' lines lighter, under the
    index's."""

    kind = "constituents"


class BacktestPlots(ChartMethods):
    """Charts for a `BacktestResult`."""

    def performance(self,
                    title: str | None = None,
                    subtitle: str | None = None,
                    notes: str | None = None,
                    style: StyleArgument = None) -> Chart:
        """Growth of 100 with a drawdown panel beneath it.

        The two share a horizontal axis, because a drawdown is only
        meaningful against the path that produced it, and reading them side
        by side would mean matching dates by eye.

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
        # A unit's NAV when money flowed, so a flow is not drawn as a gain
        # (BN-267); the NAV itself otherwise.
        levels = _rebase(_series(self._result.performance_levels())).rename("Level")
        drawdown = (levels / levels.cummax() - 1.0).rename("Drawdown")

        return _Performance([[Line(levels, reference=100.0, label_last=True, ylabel="Level",
                                   style=style)],
                             [Area(drawdown, percent=True, ylabel="Drawdown",
                                   colors=["danger"], style=style)]],
                            heights=(3.0, 1.0), sharex=True,
                            title=_given(title, "Growth of 100"),
                            subtitle=_given(subtitle, "The portfolio, with its drawdown"),
                            notes=_given(notes, sentences(
                                provenance(self._result, "backtest", levels.index),
                                "Drawdown is the level against its running peak.")),
                            style=style)

    def constituents(self,
                     limit: int = MAX_LINES,
                     title: str | None = None,
                     subtitle: str | None = None,
                     notes: str | None = None,
                     style: StyleArgument = None) -> Chart:
        """The index the backtest tracked, among its constituents.

        The same chart as `IndexResult.plot.constituents()`, for the index
        the run aimed at.

        Args:
            limit: How many constituents to draw, the largest by weight.
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
            ValueError: If the run tracked weights rather than an index.
        """
        source = self._result.index.target.source

        if source is None:
            raise ValueError(
                "this backtest tracked weights rather than a calculated "
                "index, so it has no constituents to draw.")

        return IndexPlots(source).constituents(limit=limit, title=title, subtitle=subtitle,
                                               notes=notes, style=style)

    def annual_returns(self,
                       title: str | None = None,
                       subtitle: str | None = None,
                       notes: str | None = None,
                       style: StyleArgument = None) -> Chart:
        """Calendar-year returns as signed columns.

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
        # BN-259: each year ran from its own first close, so every year after
        # the first lost its first day's return.
        returns = self._result.get_annual_returns().astype(float)
        returns.index = [str(year) for year in returns.index]

        return Column(returns.rename("Return"), percent=True, ylabel="Return",
                      title=_given(title, "Annual returns"),
                      subtitle=_given(subtitle, "The portfolio's return in each calendar year"),
                      notes=_given(notes, sentences(
                          provenance(self._result, "backtest",
                                     _series(self._result.performance_levels()).index),
                          "Each year runs from the previous year's close, and "
                          "the first from the initial capital.")),
                      style=style)


class _Performance(Grid):
    """A level over its drawdown, three to one, on one calendar: drawn at the
    performance chart's size, the drawdown's panel spaced for its few
    ticks."""

    def _size(self) -> tuple[float, float]:
        with design.scoped(self.style):
            return figure_size("performance")

    def _build(self) -> Plan:
        plan = super()._build()
        upper, lower = plan.description.panels
        lower = dataclasses.replace(lower, y=dataclasses.replace(lower.y,
                                                                 spacing=DRAWDOWN_SPACING))
        description = dataclasses.replace(plan.description, panels=(upper, lower))

        return dataclasses.replace(plan, description=description)


class AttributionPlots(ChartMethods):
    """Charts for an `AttributionResult`."""

    def contributions(self,
                      title: str | None = None,
                      subtitle: str | None = None,
                      notes: str | None = None,
                      style: StyleArgument = None) -> Chart:
        """Per-constituent contributions as diverging bars, largest first.

        The drags are stated in the chart's notes rather than drawn as bars,
        because they are comparisons against a counterfactual rather than
        terms in the decomposition: adding them to the same total would mix
        two different questions.

        The total in the notes is recomputed from the bars actually drawn. A
        total that does not match what is beside it is worse than none, and
        adding up the page is the only way to be sure.

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
        result = self._result
        items = result.contributions
        bars = _largest_first([item.asset_id for item in items],
                              [item.contribution for item in items])
        dropped = len(items) - len(bars)

        # A residual inside the result's own reconciliation tolerance is
        # rounding, and printed as a number it was noise that changed with the
        # numpy build: "5.4e-15" on one machine, "5.6e-15" on another, which
        # failed the chart comparison on nothing but the last digit (BN-222).
        # So it reads 0 when the result says it reconciles, and the number
        # only when there is something left over worth seeing.
        residual = "0" if result.reconciles() else f"{result.residual:.1e}"
        note = (f"Contributions shown sum to {float(bars.sum()):.2%}"
                f"{f'; {dropped} smaller not shown' if dropped else ''}"
                f". Total return {result.total_return:.2%}, "
                f"residual {residual}.")

        drags = [(name, value) for name, value in
                 (("Cap drag", result.cap_drag), ("Cost drag", result.cost_drag))
                 if value is not None]
        if drags:
            note += ("  " + "  ".join(f"{name} {value:.2%}" for name, value in drags))

        return Bar(bars, percent=True, title=_given(title, "Contribution to return"),
                   subtitle=_given(subtitle, "Each constituent's share of the return"),
                   notes=_given(notes, note), style=style)


# The optimiser and risk charts live in their own module; the lazy accessor
# looks every chart class up here.
from .analysis_charts import OptimisationPlots, RiskPlots  # noqa: E402
