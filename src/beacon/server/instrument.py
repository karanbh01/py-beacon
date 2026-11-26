# src/beacon/server/instrument.py
"""
A strategy's latest run's analytics a name at a time, as the server gives
them: the per-name table (`GET /strategies/{id}/instrument`), the figures
Drilldown shows for one name (`GET /strategies/{id}/assets/{identifier}`) and
a name's profile over the run (`GET /strategies/{id}/instrument/{identifier}`).

Each reads `result.analytics.instrument` on the stored run, under the options
its own analysis was made with, so the server works nothing out itself and
each agrees with the run's analysis.
"""
# BN-454 (Karan, 2026-11-25): Drilldown reads the analytics suite, as the
# risk-return cloud does.
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.analytics import Analytics
from ..analysis.analytics import profile as profiles
from ..analysis.analytics.instrument_table import columns_of
from ..analysis.data import RunData
from ..analysis.reports.statistics import Unavailable
from ..analysis.request import AnalysisRequest
from ..data.fetcher import DataFetcher
from ..exceptions import InvalidArgumentError
from .analysis import _options, read_analytics

# What Drilldown shows of a name, from its row of the table.
ASSET_COLUMNS = ("observations", "total_return", "excess_return", "tracking_error",
                 "correlation", "beta")


def stored_analytics(folder: Path,
                     source: DataFetcher | None) -> Analytics:
    """The analytics of the run kept in *folder*, under the options its own
    analysis was made with, or the defaults when it was made without one."""
    return Analytics(RunData.from_record(folder), _stored_request(folder), source, folder)


def instrument_table(analytics: Analytics,
                     columns: Sequence[str] | None,
                     names: Sequence[str] | None,
                     start: str | None,
                     end: str | None,
                     against: str | None,
                     frequency: str) -> dict[str, Any]:
    """`instrument.table()` as the API gives it: a row a name, with each
    column's label and unit and what it was measured on.

    Raises:
        InvalidArgumentError: If a column, name, comparator or frequency is
            refused.
    """
    try:
        table = analytics.instrument.table(columns=columns, names=names, start=start, end=end,
                                           against=against, frequency=frequency)
    except (ValueError, TypeError, Unavailable) as error:
        raise InvalidArgumentError(f"The table asked for cannot be given: "
                                   f"{getattr(error, 'reason', None) or error}") from error

    attrs = table.attrs

    return {"basis": attrs["basis"], "against": attrs.get("against"),
            "currency": attrs["currency"], "start": _day(attrs["start"]),
            "end": _day(attrs["end"]),
            "columns": [{"name": column.name, "label": column.label, "unit": column.unit}
                        for column in columns_of(table)],
            "names": [str(name) for name in table.index],
            "data": [[_plain(value) for value in row] for row in table.itertuples(index=False)],
            "unavailable": dict(attrs["unavailable"])}


def asset_figures(analytics: Analytics,
                  identifier: str) -> dict[str, Any]:
    """The figures Drilldown shows for *identifier*: its row of the
    per-name table and the comparator's return, each None with the reason
    when the run cannot give them."""
    try:
        table = analytics.instrument.table(columns=ASSET_COLUMNS, names=[identifier])
    except (ValueError, Unavailable) as error:
        return {"figures_unavailable": getattr(error, "reason", None) or str(error)}

    row = table.loc[identifier]
    against = table.attrs.get("against")
    theirs = None if against is None else analytics.portfolio.performance.total_return(of=against)

    return {"total_return": _plain(row.get("total_return")), "index_return": _plain(theirs),
            "excess_return": _plain(row.get("excess_return")),
            "tracking_error": _plain(row.get("tracking_error")),
            "correlation": _plain(row.get("correlation")), "beta": _plain(row.get("beta")),
            "observations": int(row["observations"]) if "observations" in row else 0,
            "against": against, "basis": table.attrs["basis"],
            "currency": table.attrs["currency"], "start": _day(table.attrs["start"]),
            "end": _day(table.attrs["end"]),
            "figures_unavailable": "; ".join(f"{column}: {reason}" for column, reason
                                             in table.attrs["unavailable"].items()) or None}


def instrument_profile(analytics: Analytics,
                       identifier: str,
                       start: str | None,
                       end: str | None,
                       against: str | None,
                       by: str = "quarter") -> dict[str, Any]:
    """`instrument.profile()` as the API gives it: each part a dated table,
    with each column's label and unit, and each part the run cannot give
    named with why.

    Raises:
        InvalidArgumentError: If no portfolio held *identifier*, or the
            comparator or *by* is refused.
    """
    # BN-468; BN-469 added by.
    try:
        profile = analytics.instrument.profile(identifier, start=start, end=end,
                                               against=against, by=by)
    except (ValueError, Unavailable) as error:
        raise InvalidArgumentError(f"The profile asked for cannot be given: "
                                   f"{getattr(error, 'reason', None) or error}") from error

    parts: dict[str, Any] = {}
    unavailable: dict[str, str] = {}

    for part in profile.parts:
        try:
            table = getattr(profile, part)
        except Unavailable as error:
            unavailable[part] = error.reason
            continue

        parts[part] = {"columns": [{"name": column.name, "label": column.label,
                                    "unit": column.unit, "labels": column.labels}
                                   for column in profiles.columns_of(table)],
                       "dates": [_day(date) for date in table.index],
                       "data": [[_plain(value) for value in row]
                                for row in table.itertuples(index=False)]}

    days = analytics._context_for(start=start, end=end).performance.returns.index

    return {"identifier": identifier, "currency": analytics.run_data.currency,
            "start": _day(days[0]) if len(days) else None,
            "end": _day(days[-1]) if len(days) else None,
            "against": (profile.daily_weights.attrs.get("against")
                        if "daily_weights" in parts else None),
            "by": by, "linking": analytics._context.options.attribution.linking,
            "parts": parts, "unavailable": unavailable}


def _stored_request(folder: Path) -> AnalysisRequest | None:
    """The request the run's own analysis was made with, as far as its
    options go; None when it was made without one."""
    block = read_analytics(folder)
    options = ((block or {}).get("request") or {}).get("options")

    if not options:
        return None

    return AnalysisRequest(reports={}, statistic_tables={}, options=_options(options))


def _day(value: Any) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "item"):
        found = value.item()

        return None if isinstance(found, float) and pd.isna(found) else found

    return value


__all__ = ["ASSET_COLUMNS", "asset_figures", "instrument_profile", "instrument_table",
           "stored_analytics"]
