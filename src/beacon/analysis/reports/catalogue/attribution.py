# src/beacon/analysis/reports/catalogue/attribution.py
"""
The statistics of where the fund's return came from: by name and group,
against the tracked comparator, by Brinson-Fachler effect, by currency, by
factor, and the implementation breakdown of the tracking difference.
"""
# BN-398/BN-400.
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import pandas as pd

from ...attribution.contribution import AttributionResult
from ...attribution.effects import EffectsResult
from ..context import Context
from ..statistics import ATTRIBUTION, NO_GROUPING, TABLE, Statistic, Summarised, Unavailable

CATEGORY = "attribution"

NAME, GROUP = "name", "group"
GROUPING = ("attribution.grouping",)


def _contributions(read: Callable[[Context, str | None], AttributionResult]) -> Callable[
        [Context], Summarised]:
    """Contributions by name and, where the run kept the grouping field, by
    group, in one table with a ``level`` column."""
    def compute(context: Context) -> Summarised:
        by_name = read(context, None)
        tables = [by_name.to_frame().assign(level=NAME)]

        # Groups only for a run that kept the grouping field for its names.
        with suppress(KeyError):
            tables.append(read(context, context.options.attribution.grouping)
                          .to_frame().assign(level=GROUP))

        return Summarised(pd.concat(tables, ignore_index=True), _summary(by_name))

    return compute


def _summary(result: AttributionResult) -> dict[str, Any]:
    return {"start": result.start, "end": result.end, "total_return": result.total_return,
            "residual": result.residual, "cap_drag": result.cap_drag,
            "cost_drag": result.cost_drag}


def _effects(result: EffectsResult,
             prefix: str = "") -> dict[str, Any]:
    """An effects result's figures: its totals and each effect."""
    found = {"start": result.start, "end": result.end, "total_return": result.total_return,
             "explained": result.explained, "residual": result.residual,
             **{f"effect/{name}": value for name, value in result.effects.items()}}

    return {f"{prefix}{name}": value for name, value in found.items()}


def _brinson(context: Context) -> Summarised:
    start, end = context.window()

    try:
        found = context.attribution.brinson(context.options.attribution.grouping,
                                            context.require_tracked(), start, end)
    except KeyError as error:
        raise Unavailable(NO_GROUPING) from error

    # Each group's row, then its names' (BN-415), told apart by level.
    groups = found.groups.assign(level=GROUP, group=found.groups.index)
    names = found.names.reset_index(GROUP).assign(level=NAME)
    table = pd.concat([groups, names])

    return Summarised(table, {"start": found.start, "end": found.end,
                                     "fund_return": found.fund_return,
                                     "comparator_return": found.comparator_return,
                                     "active_return": found.active_return,
                                     "explained": found.explained, "residual": found.residual})


def _currency(context: Context) -> Summarised:
    start, end = context.window()
    found = context.attribution.currency(context.tracked, True, start, end)

    return Summarised(found.table, _effects(found))


def _factors(context: Context) -> Summarised:
    start, end = context.window()
    analytics = context.factor_analytics
    holdings = context.attribution.factors(analytics, context.tracked, "holdings", start, end)
    returns = context.attribution.factors(analytics, context.tracked, "returns", start, end)

    return Summarised(holdings.effects.rename("holdings").to_frame()
                      .join(returns.effects.rename("returns"), how="outer"),
                      {**_effects(holdings, "holdings/"), **_effects(returns, "returns/")})


def _implementation(context: Context) -> Summarised:
    start, end = context.window()
    found = context.attribution.implementation(context.require_tracked(), start, end)

    return Summarised(found.effects.rename("effect").to_frame(), _effects(found))


def statistics() -> list[Statistic]:
    """The attribution statistics, in order."""
    def stat(id_: str,
             label: str,
             description: str,
             read: Callable[[Context], Any],
             settings: tuple[str, ...] = ()) -> Statistic:
        return Statistic(id_, label, CATEGORY, TABLE, ATTRIBUTION, description, read,
                         unit="fraction", settings=settings)

    return [
        stat("contribution", "Contribution by name and group", "Each name's and group's "
             "contribution to the return, Carino-linked.",
             _contributions(lambda context, field: context.attribution.contribution(
                 field=field, start=context.window()[0], end=context.window()[1])),
             settings=GROUPING),
        stat("active_attribution", "Active attribution", "Each name's and group's "
             "contribution to the return over the tracked comparator, from its active "
             "weight.", _contributions(lambda context, field: context.attribution.active(
                 context.require_tracked(), field, *context.window())),
             settings=(*GROUPING, "window.tracked")),
        stat("brinson", "Brinson-Fachler effects", "Allocation, selection and interaction "
             "by group, cash its own group, and each group's selection and interaction by "
             "name, linked over the run.", _brinson,
             settings=(*GROUPING, "window.tracked")),
        stat("currency_attribution", "Currency attribution", "Local return, currency effect "
             "and cross term, by currency.", _currency, settings=("window.tracked",)),
        stat("factor_attribution", "Factor attribution", "The return explained by factor "
             "exposures, holdings-based and returns-based.", _factors,
             settings=("window.tracked",)),
        stat("implementation_breakdown", "Implementation breakdown", "The tracking difference "
             "split into cap and screens, drift, cash drag, equitisation, each cost, fees, "
             "withholding, creation fees and dividend treatment.", _implementation,
             settings=("window.tracked",)),
    ]
