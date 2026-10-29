# src/beacon/analysis/reports/rolling.py
"""
Statistics over each trailing window: the same calculation as each one's
figure over the whole window, run on each window in turn, so a rolling figure
is the point figure of its window.

    tables = rolling_all([CATALOGUE["sharpe_ratio"], CATALOGUE["beta"]], context)

A window ends every day, or every month end; a statistic that is heavy to
roll steps month end unless asked otherwise. A window is a number of daily
returns, measured from the close before its first.

The most read figures (return, volatility, Sharpe and Sortino ratios,
maximum drawdown, against each comparator the tracking difference and error,
information ratio, beta and correlation, and value at risk and expected
shortfall) roll in one pass over the whole series instead, which the tests
check gives the window-by-window figures. Every other statistic is worked out
window by window, all of them together, so those that read the same
calculation share it.
"""
# BN-398.
from collections.abc import Callable, Sequence
from dataclasses import replace

import pandas as pd

from ..performance.rolling import rolling_figures, rolling_relative
from .context import FUND, Context
from .options import DAILY, MONTH_END
from .statistics import HEAVY, Statistic, Unavailable, figures_of

ACTIVE = "active"


def step_of(statistic: Statistic,
            context: Context) -> str:
    """How often *statistic*'s windows end: as asked, else by its cost."""
    asked = context.options.rolling.step

    if asked is not None:
        return asked

    return MONTH_END if statistic.rolling == HEAVY else DAILY


def window_ends(days: pd.DatetimeIndex,
                window: int,
                step: str) -> list[int]:
    """The positions in *days* each full window ends at."""
    ends = list(range(window - 1, len(days)))

    if step == MONTH_END and ends:
        months = pd.Series(ends, index=days[ends]).groupby(days[ends].to_period("M")).max()
        return [int(end) for end in months]

    return ends


def rolling(statistic: Statistic,
            context: Context) -> pd.DataFrame:
    """*statistic*'s figures over each trailing window: dates by figure
    name, empty when the window is longer than the run.

    Raises:
        ValueError: If the statistic cannot roll.
    """
    return rolling_all([statistic], context)[statistic.id]


def rolling_all(statistics: Sequence[Statistic],
                context: Context) -> dict[str, pd.DataFrame]:
    """Each of *statistics* over each trailing window, by id.

    Raises:
        ValueError: If one of them cannot roll.
    """
    found: dict[str, pd.DataFrame] = {}
    by_window: dict[str, list[Statistic]] = {}

    for statistic in statistics:
        if statistic.rolling is None:
            raise ValueError(f"{statistic.label} is not a figure of a window of returns, so it "
                             f"does not roll.")

        if statistic.roll is not None:
            found[statistic.id] = statistic.roll(context)
        elif statistic.id in FAST:
            found[statistic.id] = _stepped(FAST[statistic.id](context), statistic, context)
        else:
            by_window.setdefault(step_of(statistic, context), []).append(statistic)

    for step, chosen in by_window.items():
        found.update(window_by_window(chosen, context, step))

    return found


def window_by_window(statistics: Sequence[Statistic],
                     context: Context,
                     step: str) -> dict[str, pd.DataFrame]:
    """*statistics* worked out on each window in turn, sharing each window's
    analyses."""
    window = context.options.rolling.window
    days = pd.DatetimeIndex(context.performance.returns.index)
    rows: dict[str, dict[pd.Timestamp, dict[str, object]]] = {item.id: {} for item in statistics}

    for end in window_ends(days, window, step):
        dated = replace(context.options, window=replace(context.options.window,
                                                          start=days[end - window + 1],
                                                          end=days[end]))
        windowed = Context(context.run, dated, context.source, context.factors)

        for statistic in statistics:
            try:
                rows[statistic.id][days[end]] = figures_of(statistic.compute(windowed))
            except Unavailable:
                rows[statistic.id][days[end]] = {}

    return {statistic_id: _dated(table) for statistic_id, table in rows.items()}


def _dated(rows: dict[pd.Timestamp, dict[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.DatetimeIndex(table.index, name="DATE")

    return table


def _stepped(table: pd.DataFrame,
             statistic: Statistic,
             context: Context) -> pd.DataFrame:
    """A daily rolling table cut to its full windows, at the statistic's
    step."""
    days = pd.DatetimeIndex(table.index)
    ends = window_ends(days, context.options.rolling.window, step_of(statistic, context))
    found = table.iloc[ends]
    found.index = pd.DatetimeIndex(found.index, name="DATE")

    return found


def _own(column: str,
         active: bool = False) -> Callable[[Context], pd.DataFrame]:
    """A figure of each book's returns, rolled in one pass a book."""
    def compute(context: Context) -> pd.DataFrame:
        window = context.options.rolling.window
        books = {}

        for name in context.books():
            book = context.book(name)
            books[name] = rolling_figures(book.returns, window, book.periods_per_year,
                                          book.risk_free_rate)[column]

        table = pd.DataFrame(books)

        if active and context.tracked is not None:
            table[ACTIVE] = table[FUND] - table[context.tracked]

        return table.reindex(context.performance.returns.index)

    return compute


def _against(column: str) -> Callable[[Context], pd.DataFrame]:
    """A figure of the fund against each comparator, rolled in one pass."""
    def compute(context: Context) -> pd.DataFrame:
        window = context.options.rolling.window
        found = {}

        for name in context.comparators:
            relative = context.relative(name)
            found[name] = rolling_relative(relative.ours, relative.theirs, window,
                                           relative.periods_per_year)[column]

        return pd.DataFrame(found).reindex(context.performance.returns.index)

    return compute


FAST: dict[str, Callable[[Context], pd.DataFrame]] = {
    "total_return": _own("return", active=True),
    "volatility": _own("volatility"),
    "sharpe_ratio": _own("sharpe_ratio"),
    "sortino_ratio": _own("sortino_ratio"),
    "max_drawdown": _own("max_drawdown"),
    "tracking_difference": _against("relative_return"),
    "tracking_error": _against("tracking_error"),
    "information_ratio": _against("information_ratio"),
    "beta": _against("beta"),
    "correlation": _against("correlation"),
}


def _register_loss() -> None:
    # Imported here: the loss module reads this one's context.
    from .rolling_loss import ES, VAR, rolling_loss  # noqa: PLC0415

    FAST["value_at_risk"] = rolling_loss(VAR)
    FAST["expected_shortfall"] = rolling_loss(ES)


_register_loss()
