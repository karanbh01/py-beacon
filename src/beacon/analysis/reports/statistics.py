# src/beacon/analysis/reports/statistics.py
"""
What a statistic is: its id and label, what it measures, the kind of value it
gives, how it is worked out, and whether it can roll.

A statistic gives one of three kinds of value:

- ``figures``: named numbers, by book (``fund``, ``index``, ``benchmark``,
  ``class/<name>``), by comparator, or by the parts of a compound name joined
  with ``/`` (``historical/0.95/1``);
- ``series``: one series, dated or not;
- ``table``: one table.

One that cannot be worked out for a run raises `Unavailable` with a short
code and a sentence saying why.
"""
# BN-398/BN-400.
import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FIGURES, SERIES, TABLE = "figures", "series", "table"
KINDS = (FIGURES, SERIES, TABLE)

PERFORMANCE, PORTFOLIO, RISK, ATTRIBUTION = "performance", "portfolio", "risk", "attribution"
PARTS = (PERFORMANCE, PORTFOLIO, RISK, ATTRIBUTION)

# What rolling a statistic costs: running it over each trailing window is
# light or moderate for most, heavy for a few, which step month end unless
# asked otherwise.
LIGHT, MODERATE, HEAVY = "light", "moderate", "heavy"

# Why a statistic could not be worked out for a run.
NO_INDEX = "no_index"
NO_BENCHMARK = "no_benchmark"
NO_QUOTES = "no_quotes"
NO_FUTURES = "no_futures"
NO_SHARE_CLASSES = "no_share_classes"
NO_DATA_STORE = "no_data_store"
TOO_SHORT = "too_short"
RUN_PREDATES = "run_predates"
NO_CASH_BAND = "no_cash_band"
NO_FLOWS = "no_flows"
NO_GROUPING = "no_grouping"

REASONS = {
    NO_INDEX: ("No index", "Relative figures need the index the run traded toward."),
    NO_BENCHMARK: ("No benchmark", "This needs a benchmark set on the run."),
    NO_QUOTES: ("No exchange quotes", "This needs the fund's quotes on an exchange."),
    NO_FUTURES: ("No futures", "This needs a run that held index futures."),
    NO_SHARE_CLASSES: ("No share classes", "This needs a run with share classes."),
    NO_DATA_STORE: ("No data store", "This reads the data store the run was made from, "
                                     "which is not loaded."),
    TOO_SHORT: ("Too short", "The window is shorter than this statistic needs."),
    RUN_PREDATES: ("Run predates it", "The run was stored before this was kept; run the "
                                      "backtest again to have it."),
    NO_CASH_BAND: ("No cash band", "This needs a run that kept its cash in a band."),
    NO_FLOWS: ("No flows", "This needs a run with money arriving or leaving."),
    NO_GROUPING: ("No grouping", "The run kept no reference field to group its names by."),
}


class Unavailable(Exception):
    """Why a statistic cannot be worked out for a run.

    Attributes:
        code: One of `REASONS`.
        reason: A plain sentence saying why, fit to show as it is.
    """

    def __init__(self,
                 code: str,
                 reason: str | None = None):
        self.code = code
        self.reason = reason or REASONS[code][1]
        super().__init__(self.reason)


@dataclass(frozen=True)
class Column:
    """One column of a table statistic or a large table.

    Attributes:
        name: As it is keyed.
        label: As it is shown.
        unit: What it is measured in; None for text.
        labels: For a column of coded text, each value's label.
    """
    name: str
    label: str
    unit: str | None = None
    labels: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Statistic:
    """One statistic.

    Attributes:
        id: Its id, which requests and stored choices name it by.
        label: Its name, as shown.
        category: What it is a figure of, which reports are sectioned by.
        kind: ``figures``, ``series`` or ``table``.
        part: The analysis that works it out: ``performance``,
            ``portfolio``, ``risk`` or ``attribution``.
        description: One line saying what it measures.
        compute: Works it out from a `Context`.
        unit: What its figures (or series) are measured in.
        decimals: How many decimals it is shown to; by unit when None.
        rolling: What rolling it costs, when it can roll; None when it
            cannot.
        settings: The settings it reads, as ``section.setting``.
        columns: For a table, its columns in order.
        figure_names: What its figure names are made of, part by part.
        roll: Works out its rolling figures itself, for a statistic that
            cannot be worked out window by window; dates by figure name.
    """
    id: str
    label: str
    category: str
    kind: str
    part: str
    description: str
    compute: Callable[[Any], Any]
    unit: str | None = None
    decimals: int | None = None
    rolling: str | None = None
    settings: tuple[str, ...] = ()
    columns: tuple[Column, ...] = ()
    figure_names: tuple[str, ...] = ("book",)
    roll: Callable[[Any], pd.DataFrame] | None = None


@dataclass(frozen=True)
class Summarised:
    """A table statistic's table, with the figures that sum it up (its
    totals, its residual)."""
    table: pd.DataFrame
    summary: Mapping[str, Any]


# How many decimals a figure is shown to, by its unit.
DECIMALS = {"fraction": 4, "ratio": 2, "bps": 1, "money": 2, "count": 0, "days": 0, "years": 1}


def decimals_of(statistic: Statistic) -> int | None:
    """How many decimals *statistic* is shown to."""
    if statistic.decimals is not None:
        return statistic.decimals

    return DECIMALS.get(statistic.unit or "")


@dataclass(frozen=True)
class Catalogue:
    """Every statistic, by id, in order."""
    statistics: Mapping[str, Statistic] = field(default_factory=dict)

    def __getitem__(self,
                    statistic_id: str) -> Statistic:
        return self.statistics[statistic_id]

    def __contains__(self,
                     statistic_id: object) -> bool:
        return statistic_id in self.statistics

    def __iter__(self) -> Any:
        return iter(self.statistics.values())

    def __len__(self) -> int:
        return len(self.statistics)


def figures_of(value: Any,
               prefix: str = "") -> dict[str, Any]:
    """*value* as named figures: a mapping or a dataclass flattened, nested
    names joined with ``/``, each figure a number, text or None."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}

    if not isinstance(value, Mapping):
        return {prefix or "value": scalar(value)}

    found: dict[str, Any] = {}

    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(item, Mapping) or (dataclasses.is_dataclass(item)
                                         and not isinstance(item, type)):
            found.update(figures_of(item, name))
        elif not isinstance(item, (pd.Series, pd.DataFrame, list, tuple)):
            found[name] = scalar(item)

    return found


def scalar(value: Any) -> Any:
    """One figure as plain data: a number, text, true or false, or None for
    a missing one; a date as ``YYYY-MM-DD``."""
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return None if math.isnan(float(value)) else float(value)

    return str(value)
