# src/beacon/analysis/analytics/forms.py
"""
An analytic's keywords, which pick the form a statistic is given in, and the
statistic worked out in that form. Called bare, an analytic gives the fund
over the whole run; keywords combine:

    total_return()                       # the fund's, a float
    total_return(of="all")               # each portfolio's, a Series
    total_return(by="year")              # each year's, a Series
    total_return(by="year", of="all")    # a DataFrame, a year a row
    volatility(rolling=126, step="month_end")
    tracking_error(against="benchmark", start="2024-01-01")
    holdings_count(as_of="2024-06-28")
    sharpe_ratio(by="month", annualise=False)
    costs(by="reason")                   # a table's own groups (`groupings`)

Each statistic takes the keywords its catalogue entry lists (`forms`); any
other is refused, naming the ones it takes. Only one of ``by``,
``trailing``, ``rolling`` and ``as_of`` is given at a time.
"""
# BN-434 (Karan, 2026-11-24).
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import pandas as pd

from ...data.base import as_of_position
from ..annualising import ANY_WINDOW
from ..performance.formulas import ANNUAL, MONTHLY, QUARTERLY, daily_rate, grouped
from ..performance.return_figures import window_starts
from ..reports.catalogue.forms import (
    AGAINST,
    ANNUALISE,
    AS_OF,
    BY,
    BY_BOOK_TABLES,
    COMPARATOR,
    END,
    GRID,
    GROUPINGS,
    OF,
    PORTFOLIO,
    ROLLING,
    START,
    STEP,
    TRAILING,
)
from ..reports.context import FUND, Context
from ..reports.options import DAILY, MONTH_END, RollingOptions
from ..reports.rolling import rolling as rolled
from ..reports.statistics import FIGURES, StatisticSpec, Unavailable, figures_of
from . import groupings
from .shape import ALL, number, of_figures, of_row, of_table, plain

# The calendar periods `by` takes, and the frequency each is grouped by.
PERIODS = {"month": MONTHLY, "quarter": QUARTERLY, "year": ANNUAL}

# Only one of these at a time: each adds the time dimension.
OVER_TIME = (BY, TRAILING, ROLLING, AS_OF)

# A rolling window named in months or years, as `trailing` names them.
MONTHS = {"M": 1, "Y": 12}


def check(statistic: StatisticSpec,
          form: Mapping[str, Any]) -> None:
    """Refuse a keyword *statistic* does not take, or keywords that do not
    combine.

    Raises:
        TypeError: If a keyword is not one the statistic takes.
        ValueError: If more than one of ``by``, ``trailing``, ``rolling``
            and ``as_of`` is given, or ``step`` without ``rolling``.
    """
    unknown = [keyword for keyword in form if keyword not in statistic.forms]

    if unknown:
        takes = ", ".join(statistic.forms) or "no keywords"
        raise TypeError(f"{statistic.id} takes {takes}; not {', '.join(unknown)}.")

    over_time = [keyword for keyword in OVER_TIME if keyword in form]

    if len(over_time) > 1:
        raise ValueError(f"Give one of {', '.join(OVER_TIME)} at a time, not "
                         f"{' and '.join(over_time)}.")

    if STEP in form and ROLLING not in form:
        raise ValueError("step goes with rolling.")


def value(statistic: StatisticSpec,
          form: Mapping[str, Any],
          context_for: Callable[..., Context]) -> Any:
    """*statistic* in the form *form* asks for.

    Args:
        statistic: The statistic.
        form: Its keywords, those left as None not given.
        context_for: The context for a window (`start`, `end`), a tracked
            comparator (`tracked`), rolling options (`rolling`) and a way of
            annualising (`annualise`).

    Raises:
        Unavailable: If the run cannot give it.
    """
    form = {keyword: given for keyword, given in form.items() if given is not None}
    check(statistic, form)
    tracked = _tracked(statistic, form.get(AGAINST))
    window = {"start": form.get(START), "end": form.get(END), "tracked": tracked,
              "annualise": _annualise(form)}

    if form.get(GRID):
        return groupings.grid(form, context_for(**window))

    if BY in form and statistic.id in GROUPINGS:
        return groupings.grouped(statistic, form, context_for(**window))

    if BY in form:
        return _by_period(statistic, form, context_for, window)

    if TRAILING in form:
        return _trailing(statistic, form, context_for, window)

    # A rolling window in months is counted in the run's days, even per day.
    days = context_for(**{**window, "annualise": None})
    context = context_for(**window, rolling=_rolling_options(form, days))

    if ROLLING in form:
        return _picked_table(statistic, rolled(statistic, context), form, context)

    found = plain(statistic.compute(context))

    if AS_OF in form:
        return _as_of(statistic, found, form, context)

    return _picked(statistic, found, form, context)


def _annualise(form: Mapping[str, Any]) -> bool | None:
    """How an annualised figure is given: over any window (True), per day
    (False), or over a window long enough for it (None)."""
    annualise = form.get(ANNUALISE)

    if annualise is not None and not isinstance(annualise, bool):
        raise ValueError(f"annualise is True, False or None, got {annualise!r}.")

    return annualise


def _tracked(statistic: StatisticSpec,
             against: str | None) -> str | None:
    """The comparator to track for *statistic*: none to change when its
    figures are given one each comparator, and *against* picks among them."""
    if against is None or (statistic.figure_names[:1] == (COMPARATOR,)
                           and statistic.kind == FIGURES):
        return None

    return against


def _picked(statistic: StatisticSpec,
            found: Any,
            form: Mapping[str, Any],
            context: Context) -> Any:
    """A worked-out value in the shape its keywords give it."""
    if statistic.kind != FIGURES:
        return of_table(found, form.get(OF, FUND)) if statistic.id in BY_BOOK_TABLES else found

    first = statistic.figure_names[:1]
    choice, what = None, ""

    # A statistic given for the fund alone takes no `of`, and gives the fund.
    if first == (PORTFOLIO,):
        choice, what = form.get(OF, FUND), "portfolio"
    elif first == (COMPARATOR,):
        choice, what = form.get(AGAINST) or context.require_tracked(), "comparator"

    return of_figures(figures_of(found), statistic.figure_names, choice, what)


def _picked_table(statistic: StatisticSpec,
                  table: pd.DataFrame,
                  form: Mapping[str, Any],
                  context: Context) -> Any:
    """A rolling table, dates by figure name, in the shape its keywords
    give it."""
    first = statistic.figure_names[:1]

    # A table's rolling figures are its own columns (factor_betas' factors).
    if statistic.kind != FIGURES:
        return table

    if first == (PORTFOLIO,):
        return of_table(table, form.get(OF, FUND))

    if first == (COMPARATOR,):
        return of_table(table, form.get(AGAINST) or context.require_tracked())

    return table


def _as_of(statistic: StatisticSpec,
           found: Any,
           form: Mapping[str, Any],
           context: Context) -> Any:
    """A dated table or series as it stood on ``as_of``: its last row on or
    before that day."""
    day = pd.Timestamp(form[AS_OF])
    dated = isinstance(found.index, pd.DatetimeIndex)
    position = as_of_position(found.index, day) if dated else None

    if position is None:
        raise ValueError(f"{statistic.id} has nothing on or before {day:%Y-%m-%d}.")

    row = of_row(found, position)

    if statistic.id in BY_BOOK_TABLES and isinstance(row, pd.Series):
        return number(_picked(statistic, row.to_frame().T, form, context).iloc[0])

    return row


def _by_period(statistic: StatisticSpec,
               form: Mapping[str, Any],
               context_for: Callable[..., Context],
               window: Mapping[str, Any]) -> Any:
    """*statistic* in each calendar period of the window, a row each."""
    period = form[BY]

    if period not in PERIODS:
        raise ValueError(f"by is one of {', '.join(PERIODS)}, got {period!r}.")

    returns = context_for(**window).performance.returns
    spans = {label: (days.index[0], days.index[-1])
             for label, days in grouped(returns, PERIODS[period])}

    return _over(statistic, form, context_for, window, spans, "period")


def _trailing(statistic: StatisticSpec,
              form: Mapping[str, Any],
              context_for: Callable[..., Context],
              window: Mapping[str, Any]) -> Any:
    """*statistic* over each trailing window ending on the window's last
    day, a row each; None for a window longer than the run."""
    wanted = form[TRAILING]
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    returns = context_for(**window).performance.returns
    starts = window_starts(returns)
    unknown = [name for name in names if name not in starts]

    if unknown:
        raise ValueError(f"Unknown window {', '.join(unknown)}. Known: {', '.join(starts)}.")

    end = returns.index[-1] if len(returns) else None
    spans = {name: (starts[name], end) for name in names}

    return _over(statistic, form, context_for, window, spans, "window")


def _over(statistic: StatisticSpec,
          form: Mapping[str, Any],
          context_for: Callable[..., Context],
          window: Mapping[str, Any],
          spans: Mapping[str, tuple[Any, Any]],
          name: str) -> Any:
    """*statistic* over each span, a row each: a Series of single figures,
    else a DataFrame."""
    rows: dict[str, Any] = {}

    for label, (start, end) in spans.items():
        if start is None:
            rows[label] = None
            continue

        context = context_for(start=start, end=end, tracked=window["tracked"],
                              annualise=window["annualise"])

        try:
            rows[label] = _picked(statistic, plain(statistic.compute(context)), form, context)
        except Unavailable:
            rows[label] = None

    if all(not isinstance(row, pd.Series) for row in rows.values()):
        found = pd.Series(rows, dtype=float)
    else:
        found = pd.DataFrame({label: row for label, row in rows.items()
                              if isinstance(row, pd.Series)}).T.reindex(list(rows))

    found.index.name = name

    return found


def _rolling_options(form: Mapping[str, Any],
                     context: Context) -> RollingOptions | None:
    """The rolling window and step asked for; None when not asked."""
    if ROLLING not in form:
        return None

    window = form[ROLLING]
    step = form.get(STEP, context.options.rolling.step)

    if step not in (None, DAILY, MONTH_END):
        raise ValueError(f"step is {DAILY!r} or {MONTH_END!r}, got {step!r}.")

    if isinstance(window, str):
        window = _days(window, context.performance.periods_per_year)

    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError(f"rolling is a number of days, at least 2, or a span such as "
                         f"'1Y', got {form[ROLLING]!r}.")

    return RollingOptions(window=window, step=step)


def _days(span: str,
          periods_per_year: int) -> int | str:
    """A span such as ``6M`` or ``1Y`` in days; *span* itself when it is
    not one."""
    count, unit = span[:-1], span[-1:].upper()

    if not count.isdigit() or unit not in MONTHS:
        return span

    return round(int(count) * MONTHS[unit] * periods_per_year / 12)


def windowed(context: Context,
             start: Any = None,
             end: Any = None,
             tracked: str | None = None,
             rolling: RollingOptions | None = None,
             annualise: bool | None = None) -> Context:
    """*context* over another window, tracking another comparator, rolling
    another window, or annualising over any window (*annualise* True) or per
    day (False); *context* itself when nothing changes."""
    if (start is None and end is None and tracked is None and rolling is None
            and annualise is None):
        return context

    window = context.options.window
    window = replace(window,
                     start=pd.Timestamp(start) if start is not None else window.start,
                     end=pd.Timestamp(end) if end is not None else window.end,
                     tracked=tracked if tracked is not None else window.tracked)

    if annualise is not None:
        window = replace(window, min_annualised_rate_window=ANY_WINDOW,
                         min_annualised_ratio_window=ANY_WINDOW)

    # Per day is a year of one period, with the daily rate that compounds to
    # the risk-free rate.
    if annualise is False:
        performance = context.performance
        window = replace(window, periods_per_year=1,
                         risk_free_rate=daily_rate(performance.risk_free_rate,
                                                   performance.periods_per_year))
    options = replace(context.options, window=window,
                      rolling=rolling if rolling is not None else context.options.rolling)

    return Context(context.run, options, context.source, context.factors, context.folder)


__all__ = ["ALL", "PERIODS", "check", "value", "windowed"]
