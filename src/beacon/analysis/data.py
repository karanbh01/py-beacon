# src/beacon/analysis/data.py
"""
Everything a run's analysis reads, from a finished run or a stored one.

    data = RunData.from_result(result)
    data.write(folder)
    again = RunData.from_record(folder)

Every analysis of a run is built from a `RunData`, so it works the same on a
run just finished and on one stored months ago: the stored one is read back
exactly as it was written, whatever has happened to the data store since.

A run stored before its tables were kept loads with what it has, and
`missing` says, for each table it lacks, why.
"""
# BN-369.
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtest.result import BacktestResult
from ..exceptions import ConfigurationError
from .packing import CLOSES_FLOAT, CLOSES_PACKED, CLOSES_SCALE, pack_closes, unpack_closes
from .panels import (
    CASH,
    CHARACTERISTICS,
    CLOSES,
    COSTS,
    FUND_WEIGHTS,
    INDEX_WEIGHTS,
    NAME_RETURNS,
    REFERENCE,
    TRADED_VALUES,
    name_returns_from,
    read_panels,
    run_panels,
    write_panels,
)

# What a run records, beside its tables.
RUN_FILE = "run.json"
RUN_VERSION = 1

# The fund's own series and the comparators' levels, each kept as a table of
# one column so it reads back exactly, its gaps included.
RETURNS, NAV, UNITS = "returns", "nav", "units"
INDEX, OPTIMISED, BENCHMARK = "index", "optimised", "benchmark"
_LEVELS = "levels_"
_VALUE = "value"

# The tables a stored run works out again from the ones it keeps.
DERIVED = frozenset({NAME_RETURNS})

# The tables every run keeps, which a record without them says it lacks.
EXPECTED = (FUND_WEIGHTS, NAME_RETURNS, COSTS, CASH, INDEX_WEIGHTS, REFERENCE,
            CHARACTERISTICS, TRADED_VALUES)

_OLDER = ("kept before py-beacon 0.10.0, which did not keep it; run the backtest "
          "again to have it")


@dataclass(frozen=True)
class RunData:
    """A run, as its analysis reads it.

    Attributes:
        currency: The book's currency, which every amount is in.
        returns: The fund's daily returns, as its metrics measure them: of a
            unit when the run had flows, with cash paid out added back.
        nav: The fund's net assets at the end of each day, net of fees owed.
        units: Units outstanding at the end of each day.
        initial_capital: What the fund started with.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate excess returns are measured over.
        comparators: Each comparator's levels by name: ``index``,
            ``optimised`` and ``benchmark``, for those the run had.
        tables: The run's tables by name (`beacon.analysis.panels`).
        missing: For each table or series a stored run lacks, why.
    """
    currency: str
    returns: pd.Series
    nav: pd.Series
    units: pd.Series
    initial_capital: float
    periods_per_year: int
    risk_free_rate: float
    comparators: Mapping[str, pd.Series] = field(default_factory=dict)
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    missing: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_result(cls,
                    result: BacktestResult) -> "RunData":
        """A finished run's data, with every table it keeps."""
        books = {INDEX: result.index.target, OPTIMISED: result.index.optimised,
                 BENCHMARK: result.benchmark}
        nav = result.trading_nav
        tables = run_panels(result)

        return cls(currency=result.currency,
                   returns=result.get_returns(),
                   nav=nav,
                   units=result.units.reindex(nav.index),
                   initial_capital=float(result.portfolio.initial_capital),
                   periods_per_year=result._periods_per_year(),
                   risk_free_rate=result._risk_free_rate(),
                   comparators={name: book.levels for name, book in books.items()
                                if book is not None and not book.levels.empty},
                   tables=tables,
                   missing={name: "the run did not have it" for name in EXPECTED
                            if name not in tables})

    def write(self,
              folder: Path) -> None:
        """Keep this run in *folder*, replacing whatever it held: its tables
        and series as `beacon.analysis.panels` writes them, and the rest in
        ``run.json``."""
        series = {RETURNS: self.returns, NAV: self.nav, UNITS: self.units,
                  **{_LEVELS + name: levels for name, levels in self.comparators.items()}}
        # What is worked out again on reading is not written; the closes it
        # is worked out from are written packed (BN-403).
        derived = [name for name in self.tables if name in DERIVED]
        kept = {name: self.tables[name] for name in self.tables
                if name not in DERIVED and name != CLOSES}
        closes = self.tables.get(CLOSES)

        if closes is not None:
            kept.update(pack_closes(closes))

        write_panels(folder, {**kept,
                              **{name: values.to_frame(_VALUE) for name, values in series.items()}})
        run = {"version": RUN_VERSION, "currency": self.currency,
               "initial_capital": self.initial_capital,
               "periods_per_year": self.periods_per_year,
               "risk_free_rate": self.risk_free_rate,
               "missing": dict(self.missing),
               "derived": derived,
               "closes": [] if closes is None else [str(name) for name in closes.columns]}
        (folder / RUN_FILE).write_text(json.dumps(run, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_record(cls,
                    folder: Path,
                    record: Mapping[str, Any] | None = None) -> "RunData":
        """A stored run, as it was written.

        Args:
            folder: Where the run was written.
            record: For a run stored before its data was kept this way, its
                JSON record (a backtest record's ``portfolio.nav`` and the
                comparators' ``levels``), read for what the folder lacks.

        Raises:
            ConfigurationError: If neither the folder nor *record* holds the
                run, or the folder was written by a newer version.
        """
        path = folder / RUN_FILE

        if not path.exists():
            return _older(folder, record)

        run = json.loads(path.read_text(encoding="utf-8"))

        if int(run.get("version", 0)) > RUN_VERSION:
            raise ConfigurationError(str(path), "the run was kept by a newer version of "
                                                "Beacon than this one. Upgrade py-beacon-kit.")

        tables = read_panels(folder)
        series = {name: _column(tables.pop(name)) for name in list(tables)
                  if name in (RETURNS, NAV, UNITS) or name.startswith(_LEVELS)}

        if CLOSES_PACKED in tables:
            packed = {name: tables.pop(name) for name in (CLOSES_PACKED, CLOSES_SCALE, CLOSES_FLOAT)
                      if name in tables}
            tables[CLOSES] = unpack_closes(packed, run.get("closes") or None)

        return cls(currency=run["currency"],
                   returns=series[RETURNS],
                   nav=series[NAV],
                   units=series[UNITS],
                   initial_capital=float(run["initial_capital"]),
                   periods_per_year=int(run["periods_per_year"]),
                   risk_free_rate=float(run["risk_free_rate"]),
                   comparators={name[len(_LEVELS):]: levels for name, levels in series.items()
                                if name.startswith(_LEVELS)},
                   tables=_Tables(tables, _rebuilds(tables, run)),
                   missing=run.get("missing", {}))


class _Tables(Mapping[str, pd.DataFrame]):
    """A stored run's tables: those it kept, and those worked out from them
    the first time each is asked for."""

    def __init__(self,
                 kept: dict[str, pd.DataFrame],
                 rebuilds: dict[str, Callable[[], pd.DataFrame]]):
        self._kept = kept
        self._rebuilds = rebuilds

    def __getitem__(self,
                    name: str) -> pd.DataFrame:
        if name not in self._kept and name in self._rebuilds:
            self._kept[name] = self._rebuilds[name]()

        return self._kept[name]

    def __iter__(self) -> Iterator[str]:
        yield from self._kept
        yield from (name for name in self._rebuilds if name not in self._kept)

    def __len__(self) -> int:
        return len(set(self._kept) | set(self._rebuilds))


def _rebuilds(tables: dict[str, pd.DataFrame],
              run: Mapping[str, Any]) -> dict[str, Callable[[], pd.DataFrame]]:
    """How each table a stored run worked out again is rebuilt from the ones
    it kept."""
    rebuilds: dict[str, Callable[[], pd.DataFrame]] = {}

    if NAME_RETURNS in run.get("derived", []):
        # The run's days are the costs table's, a row for each.
        rebuilds[NAME_RETURNS] = lambda: name_returns_from(tables, run["currency"],
                                                           pd.DatetimeIndex(tables[COSTS].index))

    return rebuilds


def _column(table: pd.DataFrame) -> pd.Series:
    """A series kept as a table of one column."""
    column = table[_VALUE]
    column.name = None

    return column


def _older(folder: Path,
           record: Mapping[str, Any] | None) -> RunData:
    """A run stored before its data was kept with it: the tables its folder
    has and the series its record has, the rest said to be missing."""
    if record is None or not record.get("portfolio"):
        raise ConfigurationError(str(folder), "it holds no run, and no record was given "
                                              "to read one from")

    tables = read_panels(folder)
    portfolio = record["portfolio"]
    nav = _series(portfolio.get("nav"))
    previous = nav.shift(1)
    previous.iloc[:1] = float(portfolio.get("initial_capital") or nav.iloc[0])
    books = {INDEX: (record.get("index") or {}).get("target"),
             OPTIMISED: (record.get("index") or {}).get("optimised"),
             BENCHMARK: record.get("benchmark")}
    missing = {name: _OLDER for name in EXPECTED if name not in tables}
    missing[RETURNS] = ("worked out from the NAV, since a record kept before py-beacon "
                        "0.10.0 does not hold its returns: cash paid out and flows are "
                        "not accounted for")
    missing[UNITS] = _OLDER

    return RunData(currency=str(record.get("currency") or "USD"),
                   returns=nav / previous - 1.0,
                   nav=nav,
                   units=pd.Series(dtype=float),
                   initial_capital=float(portfolio.get("initial_capital") or 0.0),
                   periods_per_year=int(((record.get("settings") or {})
                                         .get("modelling_assumptions") or {})
                                        .get("periods_per_year") or 252),
                   risk_free_rate=float(((record.get("settings") or {})
                                         .get("modelling_assumptions") or {})
                                        .get("risk_free_rate") or 0.0),
                   comparators={name: _series(book.get("levels")) for name, book in books.items()
                                if book and book.get("levels")},
                   tables=tables,
                   missing=missing)


def _series(payload: Mapping[str, Any] | None) -> pd.Series:
    """A series stored as ``{"index": [...], "data": [...]}``."""
    if not payload:
        return pd.Series(dtype=float)

    return pd.Series(payload["data"], index=pd.DatetimeIndex(payload["index"]),
                     dtype=float).dropna()
