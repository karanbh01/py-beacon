# src/beacon/server/analysis.py
"""
A run's analysis on the wire: a backtest request's analysis built into the
library's request, a stored run's analytics block kept and read beside its
record, the catalogue of everything an analysis can be asked for, and a run's
large tables by column or as Arrow.

The server works out no figure here: the library does, and this turns its
answers into JSON or Arrow.
"""
# BN-398/BN-400.
import io
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.data import RunData
from ..analysis.reports.catalogue import CATALOGUE, CATEGORIES
from ..analysis.reports.definitions import REPORTS
from ..analysis.reports.options import AnalysisOptions, describe
from ..analysis.reports.rerun import rerun, same_choices
from ..analysis.reports.rolling import step_for
from ..analysis.reports.statistics import REASONS, decimals_of
from ..analysis.request import AnalysisRequest
from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError, InvalidArgumentError
from ..portfolio.reasons import REASONS as TRADE_REASONS
from .analysis_schemas import AnalyticsBlock
from .job_threads import Report, run_in_thread
from .jobs import ProgressReporter

ANALYTICS_FILE = "analytics.json"

# Analyses kept by name with a run, one file each (BN-399).
SAVED_FOLDER = "analyses"

ARROW = "application/vnd.apache.arrow.stream"

TRADES = "trades"

# A large table's columns as the run keeps them, as they are sent, and the
# decimals each number is rounded to: units to six so holdings worked out
# from them stay exact, costs to four as a cent would lose most of them.
TABLE_COLUMNS: dict[str, list[tuple[str, str, str, str | None, int | None]]] = {
    TRADES: [("DATE", "date", "Date", "date", None),
             ("IDENTIFIER", "name", "Name", None, None),
             ("SIDE", "side", "Side", None, None),
             ("QUANTITY", "units", "Units", "count", 6),
             ("PRICE", "price", "Price", "money", 4),
             ("VALUE", "value", "Value", "money", 2),
             ("COST", "cost", "Cost", "money", 4),
             ("IMPACT", "impact", "Market impact", "money", 4),
             ("REASON", "reason", "Reason", None, None),
             ("TAGS", "tags", "Tags", None, None)],
}
TABLE_LABELS = {TRADES: "Trades"}

# The coded text columns, each value's label: a trade's reason as written,
# with a capital.
CODED_LABELS = {
    "side": {"BUY": "Buy", "SELL": "Sell"},
    "reason": {reason: reason[:1].upper() + reason[1:] for reason in TRADE_REASONS},
}

# Labels for figure names that are not statistic ids, by dimension.
NAMES = {
    "book": {"fund": "Fund", "index": "Index", "optimised": "Optimised index",
             "benchmark": "Benchmark", "active": "Active", "class": "Class"},
    "comparator": {"index": "Index", "optimised": "Optimised index", "benchmark": "Benchmark"},
    "method": {"historical": "Historical", "normal": "Normal", "cornish_fisher": "Cornish-Fisher"},
    "confidence": {"0.9": "90%", "0.95": "95%", "0.975": "97.5%", "0.99": "99%",
                   "0.995": "99.5%"},
    "horizon": {"1": "1 day", "5": "5 days", "10": "10 days", "20": "20 days"},
    "frequency": {"daily": "Days", "weekly": "Weeks", "monthly": "Months",
                  "quarterly": "Quarters", "annual": "Years"},
    "extreme": {"best": "Best", "worst": "Worst"},
    "streak": {"winning": "Winning", "losing": "Losing"},
    "mean": {"arithmetic": "Arithmetic", "geometric": "Geometric",
             "arithmetic_annual": "Arithmetic, a year", "geometric_annual": "Geometric, a year"},
    "factor": {"market": "Market", "size": "Size", "value": "Value", "momentum": "Momentum"},
    "category": dict(CATEGORIES),
}


def build_request(spec: Any) -> AnalysisRequest:
    """The library's request from a backtest request's ``analysis``.

    Raises:
        InvalidArgumentError: If a report, level, statistic, view, section
            or setting is unknown, or a setting's value is refused.
    """
    try:
        return AnalysisRequest(reports=spec.reports, add=spec.add, remove=spec.remove,
                               time_views=spec.time_views, statistics=spec.statistics,
                               options=_options(spec.options))
    except (TypeError, ValueError) as error:
        raise InvalidArgumentError(f"The analysis asked for cannot be given: {error}") from error


def _options(sections: dict[str, dict[str, Any]]) -> AnalysisOptions:
    """The options from their sections, each setting left out at its
    default."""
    defaults = AnalysisOptions()
    known = {item.name: item for item in fields(AnalysisOptions)}
    built = {}

    for name, values in sections.items():
        if name not in known:
            raise ValueError(f"Unknown section {name!r}. Known: {', '.join(known)}.")

        section = type(getattr(defaults, name))
        given = {key: tuple(value) if isinstance(value, list) else value
                 for key, value in values.items()}
        built[name] = section(**given)

    return AnalysisOptions(**built)


def keep_analytics(folder: Path,
                   block: dict[str, Any]) -> None:
    """Keep a run's analytics block beside its tables."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ANALYTICS_FILE).write_text(json.dumps(block, allow_nan=False), encoding="utf-8")


def read_analytics(folder: Path) -> dict[str, Any] | None:
    """A stored run's analytics block, or None when it was made without one."""
    path = folder / ANALYTICS_FILE

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_rerun_job(strategy_id: str,
                    folder: Path,
                    record: dict[str, Any] | None,
                    request: AnalysisRequest,
                    source: DataFetcher | None,
                    save: str | None = None) -> Callable[[ProgressReporter],
                                                         Awaitable[dict[str, Any]]]:
    """The job that gives *request*'s analysis of a strategy's latest run.

    With the choices the run's own analysis was made with, the run's block
    gains what was asked for and is kept; with other choices the answer is
    kept only under *save*. The run's factor work is read from its folder
    and kept there.
    """
    def work(report: Report) -> dict[str, Any]:
        report(0.05, "Reading the run")
        run = RunData.from_record(folder, record)
        previous = read_analytics(folder)
        block = rerun(run, request, source, folder, previous,
                      lambda message: report(0.5, message))
        block = with_tables(block, strategy_id, run.tables)

        if previous is None or same_choices(previous, request):
            keep_analytics(folder, block)

        if save is not None:
            keep_saved(folder, save, block)

        report(1.0, "Complete.")

        # In the shape every read of it gives.
        return AnalyticsBlock.model_validate(block).model_dump()

    # In a worker thread, so the event loop stays free (BN-333).
    async def job(report: ProgressReporter) -> dict[str, Any]:
        return await run_in_thread(report, work)

    return job


def keep_saved(folder: Path,
               name: str,
               block: dict[str, Any]) -> None:
    """Keep *block* with the run under *name*, replacing one kept so."""
    saved = folder / SAVED_FOLDER
    saved.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
             "analytics": block}
    (saved / f"{name}.json").write_text(json.dumps(entry, allow_nan=False), encoding="utf-8")


def saved_analyses(folder: Path) -> list[dict[str, Any]]:
    """Every analysis kept by name with the run, newest first, without
    their blocks."""
    saved = folder / SAVED_FOLDER

    if not saved.exists():
        return []

    found = []

    for path in saved.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        request = entry["analytics"].get("request", {})
        found.append({"name": entry["name"], "saved_at": entry["saved_at"],
                      "reports": request.get("reports", {}), "options": request.get("options", {})})

    return sorted(found, key=lambda entry: entry["saved_at"], reverse=True)


def read_saved(folder: Path,
               name: str) -> dict[str, Any]:
    """An analysis kept by name with the run.

    Raises:
        DataNotFoundError: If none is kept under *name*.
    """
    path = folder / SAVED_FOLDER / f"{name}.json"

    if not path.exists():
        raise DataNotFoundError(f"a saved analysis '{name}'",
                                source="GET /strategies/{strategy_id}/analyses lists them")

    entry: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    request = entry["analytics"].get("request", {})

    return {**entry, "reports": request.get("reports", {}), "options": request.get("options", {})}


def delete_saved(folder: Path,
                 name: str) -> None:
    """Remove the analysis kept under *name*.

    Raises:
        DataNotFoundError: If none is kept under it.
    """
    read_saved(folder, name)
    (folder / SAVED_FOLDER / f"{name}.json").unlink()


def with_tables(block: dict[str, Any],
                strategy_id: str,
                tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """*block* naming the run's large tables and where to read each."""
    found = {name: {"rows": len(tables[name]), "path": f"/strategies/{strategy_id}/tables/{name}"}
             for name in TABLE_COLUMNS if name in tables}

    return {**block, "tables": found}


def catalogue() -> dict[str, Any]:
    """Everything an analysis can be asked for."""
    options = AnalysisOptions()

    return {
        "settings": describe(options),
        "statistics": [{"id": statistic.id, "label": statistic.label,
                        "category": statistic.category, "kind": statistic.kind,
                        "part": statistic.part, "description": statistic.description,
                        "unit": statistic.unit, "decimals": decimals_of(statistic),
                        "rolling": statistic.rolling,
                        "step": step_for(statistic, options) if statistic.rolling else None,
                        "settings": list(statistic.settings),
                        "figure_names": list(statistic.figure_names),
                        "columns": [{"name": column.name, "label": column.label,
                                     "unit": column.unit} for column in statistic.columns],
                        "signed": list(statistic.signed)}
                       for statistic in CATALOGUE],
        "reports": [{"id": report.id, "label": report.label, "order": report.order,
                     "levels": {name: {"sections": [
                         {"label": section.label, "order": order,
                          "statistics": [{"id": entry.id, "state": entry.state,
                                          "view": entry.view} for entry in section.entries]}
                         for order, section in enumerate(level.sections, start=1)]}
                         for name, level in report.levels.items()}}
                    for report in REPORTS.values()],
        "names": NAMES,
        "tables": [{"name": name, "label": TABLE_LABELS[name],
                    "columns": [{"name": wire, "label": label, "unit": unit,
                                 "labels": CODED_LABELS.get(wire)}
                                for _, wire, label, unit, _ in columns]}
                   for name, columns in TABLE_COLUMNS.items()],
        "unavailable": [{"code": code, "label": label, "help": help_text}
                        for code, (label, help_text) in REASONS.items()],
        "renamed": {},
        "dropped": [],
    }


def wire_table(name: str,
               table: pd.DataFrame) -> pd.DataFrame:
    """A run's large table with its columns as they are sent, numbers
    rounded.

    Raises:
        DataNotFoundError: If the run gives no such table.
    """
    if name not in TABLE_COLUMNS:
        raise DataNotFoundError(f"the run's {name} table",
                                source=f"a run gives {', '.join(TABLE_COLUMNS)}")

    frame = table.rename_axis("DATE").reset_index()
    found = {}

    for kept, wire, _, _, decimals in TABLE_COLUMNS[name]:
        if kept not in frame.columns:
            continue

        column = frame[kept]
        found[wire] = column.round(decimals) if decimals is not None else column

    return pd.DataFrame(found)


def columnar(name: str,
             table: pd.DataFrame) -> dict[str, Any]:
    """A large table by column: text as its distinct values and a code a
    row, a column the same on every row sent once."""
    frame = wire_table(name, table)
    columns: dict[str, Any] = {}
    constant: dict[str, Any] = {}

    for wire, column in frame.items():
        if len(frame) and column.nunique(dropna=False) == 1 and wire not in ("date", "name"):
            constant[str(wire)] = _plain(column.iloc[0])
        elif pd.api.types.is_numeric_dtype(column):
            columns[str(wire)] = [_plain(value) for value in column.to_numpy()]
        else:
            text = column.astype(str) if wire != "date" else column.dt.strftime("%Y-%m-%d")
            values, codes = np.unique(text.to_numpy(dtype=str), return_inverse=True)
            columns[str(wire)] = {"values": values.tolist(), "codes": codes.tolist()}

    decimals = {wire: places for _, wire, _, _, places in TABLE_COLUMNS[name]
                if places is not None and wire in frame.columns}

    return {"length": len(frame), "columns": columns, "constant": constant,
            "decimals": decimals}


def arrow(name: str,
          table: pd.DataFrame) -> bytes:
    """A large table as an Arrow IPC stream, its text columns
    dictionary-encoded."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.ipc as ipc  # noqa: PLC0415

    found = pa.Table.from_pandas(wire_table(name, table), preserve_index=False)

    for position, item in enumerate(found.schema):
        if pa.types.is_string(item.type) or pa.types.is_large_string(item.type):
            coded = found.column(position).dictionary_encode()
            found = found.set_column(position, item.name, coded)

    sink = io.BytesIO()

    with ipc.new_stream(sink, found.schema) as writer:
        writer.write_table(found)

    return sink.getvalue()


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value
