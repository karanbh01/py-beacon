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
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtest.result import BacktestResult
from ..exceptions import ConfigurationError
from ..portfolio.cash_flows import DISTRIBUTION, REDEMPTION, SUBSCRIPTION
from .packing import CLOSES_FLOAT, CLOSES_PACKED, CLOSES_SCALE, pack_closes, unpack_closes
from .panels import (
    CASH,
    CHARACTERISTICS,
    CLOSES,
    COSTS,
    FUND_WEIGHTS,
    HOLDINGS,
    INDEX_WEIGHTS,
    NAME_RETURNS,
    REFERENCE,
    TRADED_VALUES,
    WEIGHT_TABLES,
    name_returns_from,
    read_manifest,
    read_panels,
    run_panels,
    weights_from,
    write_panels,
)

# What a run records, beside its tables.
RUN_FILE = "run.json"
RUN_VERSION = 1

# The fund's own series and the comparators' levels, each kept as a table of
# one column so it reads back exactly, its gaps included.
RETURNS, NAV, UNITS = "returns", "nav", "units"
# The level a run's figures are measured along, and its investors' cash
# flows, for a run that had them (BN-371).
PERFORMANCE, INVESTOR_FLOWS = "performance", "investor_flows"
INDEX, OPTIMISED, BENCHMARK = "index", "optimised", "benchmark"
_LEVELS = "levels_"
_VALUE = "value"

# The tables a stored run works out again from the ones it keeps, when it
# kept what they are worked out from.
DERIVED = frozenset({NAME_RETURNS, *WEIGHT_TABLES})

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
        tables: The run's tables by name (`beacon.analysis.panels`), each
            built or read the first time it is asked for.
        missing: For each table or series a stored run lacks, why.
        performance: The level the fund's total return and drawdown are
            measured along: its NAV, or, when it had flows or paid cash out,
            what it started with grown by its returns. None for a run stored
            before it was kept, whose NAV is used.
        performance_start: What *performance* starts from: the initial
            capital, or a unit's launch price.
        investor_flows: The investors' cash flows, dated, from their side
            (money put in negative, the final NAV positive), for a run with
            flows; None for any other.
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
    performance: pd.Series | None = None
    performance_start: float | None = None
    investor_flows: pd.Series | None = None

    @classmethod
    def from_result(cls,
                    result: BacktestResult) -> "RunData":
        """A finished run's data, with every table it keeps, built the first
        time any is asked for."""
        books = {INDEX: result.index.target, OPTIMISED: result.index.optimised,
                 BENCHMARK: result.benchmark}
        nav = result.trading_nav
        # Built when first asked for (BN-371): a backtest's summary reads a
        # run's data without its tables, which take seconds at 6,000 names.
        tables = _Built(partial(run_panels, result))
        performance, start = result._performance_nav()

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
                   missing=_Lacking(tables),
                   performance=performance,
                   performance_start=float(start),
                   investor_flows=_investor_flows(result) if result.flows else None)

    def latest(self,
               name: str,
               rows: int) -> pd.DataFrame:
        """The last *rows* rows of table *name*. A stored run's daily weights
        are worked out for those days alone, unless already read whole, and
        are those rows of the whole table exactly."""
        if isinstance(self.tables, _Tables):
            return self.tables.latest(name, rows)

        return self.tables[name].tail(rows)

    def write(self,
              folder: Path) -> None:
        """Keep this run in *folder*, replacing whatever it held: its tables
        and series as `beacon.analysis.panels` writes them, and the rest in
        ``run.json``."""
        series = {RETURNS: self.returns, NAV: self.nav, UNITS: self.units,
                  **{_LEVELS + name: levels for name, levels in self.comparators.items()}}

        if self.performance is not None:
            series[PERFORMANCE] = self.performance

        if self.investor_flows is not None:
            series[INVESTOR_FLOWS] = self.investor_flows
        # What is worked out again on reading is not written; the closes it
        # is worked out from are written packed (BN-403).
        derived = [name for name in self.tables if name in DERIVED and (
            name == NAME_RETURNS or name + HOLDINGS in self.tables)]
        kept = {name: self.tables[name] for name in self.tables
                if name not in derived and name != CLOSES}
        closes = self.tables.get(CLOSES)

        if closes is not None:
            kept.update(pack_closes(closes))

        write_panels(folder, {**kept,
                              **{name: values.to_frame(_VALUE) for name, values in series.items()}})
        run = {"version": RUN_VERSION, "currency": self.currency,
               "initial_capital": self.initial_capital,
               "periods_per_year": self.periods_per_year,
               "risk_free_rate": self.risk_free_rate,
               "performance_start": self.performance_start,
               "missing": dict(self.missing),
               "derived": derived,
               "dtypes": {name: str(self.tables[name].dtypes.iloc[0]) for name in derived
                          if name != NAME_RETURNS and len(self.tables[name].columns)},
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

        # Only the series are read now; each table is read, or worked out
        # again, the first time it is asked for, so a run opened for its
        # performance figures reads only what they need (BN-403).
        held = list((read_manifest(folder) or {}).get("panels", {}))
        series = {name: _column(table) for name, table in read_panels(
            folder, [name for name in held if name in (RETURNS, NAV, UNITS, PERFORMANCE,
                                                       INVESTOR_FLOWS)
                     or name.startswith(_LEVELS)]).items()}
        packed = [name for name in (CLOSES_PACKED, CLOSES_SCALE, CLOSES_FLOAT) if name in held]
        kept = [name for name in held if name not in series and name not in packed]
        tables = _Tables({}, {name: partial(_read_one, folder, name) for name in kept})

        if packed:
            tables.add(CLOSES, lambda: unpack_closes(read_panels(folder, packed),
                                                     run.get("closes") or None))

        for name, rebuild in _rebuilds(tables, run).items():
            tables.add(name, rebuild, _window(name, tables, run))

        return cls(currency=run["currency"],
                   returns=series[RETURNS],
                   nav=series[NAV],
                   units=series[UNITS],
                   initial_capital=float(run["initial_capital"]),
                   periods_per_year=int(run["periods_per_year"]),
                   risk_free_rate=float(run["risk_free_rate"]),
                   comparators={name[len(_LEVELS):]: levels for name, levels in series.items()
                                if name.startswith(_LEVELS)},
                   tables=tables,
                   missing=run.get("missing", {}),
                   performance=series.get(PERFORMANCE),
                   performance_start=(None if run.get("performance_start") is None
                                      else float(run["performance_start"])),
                   investor_flows=series.get(INVESTOR_FLOWS))


def _investor_flows(result: BacktestResult) -> pd.Series:
    """A run's investor cash flows, as its money-weighted return counts
    them: the initial capital, every subscription, redemption and
    distribution, and the final NAV, each on its date; empty with no NAV."""
    nav = result.trading_nav

    if nav.empty:
        return pd.Series(dtype=float)

    portfolio = result.portfolio
    start = portfolio.inception if portfolio.inception is not None else nav.index[0]
    flows = [(pd.Timestamp(start), -portfolio.initial_capital)]
    flows += [(flow.date, -flow.amount) for flow in portfolio.cash_flows
              if flow.kind in (SUBSCRIPTION, REDEMPTION, DISTRIBUTION)]
    flows.append((pd.Timestamp(nav.index[-1]), float(nav.iloc[-1])))

    return pd.Series([amount for _, amount in flows],
                     index=pd.DatetimeIndex([date for date, _ in flows]), dtype=float)


class _Built(Mapping[str, pd.DataFrame]):
    """A finished run's tables, built the first time any is asked for."""

    def __init__(self,
                 build: Callable[[], dict[str, pd.DataFrame]]):
        self._build = build
        self._tables: dict[str, pd.DataFrame] | None = None

    def _all(self) -> dict[str, pd.DataFrame]:
        if self._tables is None:
            self._tables = self._build()

        return self._tables

    def __getitem__(self,
                    name: str) -> pd.DataFrame:
        return self._all()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())


class _Lacking(Mapping[str, str]):
    """The tables every run keeps that a finished run did not have, worked
    out when first asked."""

    def __init__(self,
                 tables: Mapping[str, pd.DataFrame]):
        self._tables = tables

    def _all(self) -> dict[str, str]:
        return {name: "the run did not have it" for name in EXPECTED if name not in self._tables}

    def __getitem__(self,
                    name: str) -> str:
        return self._all()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())


class _Tables(Mapping[str, pd.DataFrame]):
    """A stored run's tables, each read from its folder, or worked out again
    from the ones it kept, the first time it is asked for."""

    def __init__(self,
                 kept: dict[str, pd.DataFrame],
                 rebuilds: dict[str, Callable[[], pd.DataFrame]]):
        self._kept = kept
        self._rebuilds = rebuilds
        self._windows: dict[str, Callable[[int], pd.DataFrame]] = {}

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

    def add(self,
            name: str,
            rebuild: Callable[[], pd.DataFrame],
            window: Callable[[int], pd.DataFrame] | None = None) -> None:
        """Have *name* read or worked out the first time it is asked for;
        *window* works out only its last rows, when it can."""
        self._rebuilds[name] = rebuild

        if window is not None:
            self._windows[name] = window

    def latest(self,
               name: str,
               rows: int) -> pd.DataFrame:
        """The last *rows* rows of *name*: worked out for those alone when it
        is worked out again and has not been already."""
        if name in self._kept or name not in self._windows:
            return self[name].tail(rows)

        return self._windows[name](rows)


def _read_one(folder: Path,
              name: str) -> pd.DataFrame:
    """One stored table."""
    return read_panels(folder, [name])[name]


def _rebuilds(tables: Mapping[str, pd.DataFrame],
              run: Mapping[str, Any]) -> dict[str, Callable[[], pd.DataFrame]]:
    """How each table a stored run worked out again is rebuilt from the ones
    it kept."""
    rebuilds: dict[str, Callable[[], pd.DataFrame]] = {}

    if NAME_RETURNS in run.get("derived", []):
        # The run's days are the costs table's, a row for each.
        rebuilds[NAME_RETURNS] = lambda: name_returns_from(tables, run["currency"],
                                                           pd.DatetimeIndex(tables[COSTS].index))

    for name in WEIGHT_TABLES:
        if name in run.get("derived", []):
            rebuilds[name] = partial(weights_from, name, tables, run["currency"],
                                     run.get("dtypes", {}).get(name, "float64"))

    return rebuilds


def _window(name: str,
            tables: Mapping[str, pd.DataFrame],
            run: Mapping[str, Any]) -> Callable[[int], pd.DataFrame] | None:
    """How a stored run's daily weights are worked out for their last days
    alone; None for any other table."""
    if name not in WEIGHT_TABLES:
        return None

    return partial(_latest_weights, name, tables, run["currency"],
                   run.get("dtypes", {}).get(name, "float64"))


def _latest_weights(name: str,
                    tables: Mapping[str, pd.DataFrame],
                    currency: str,
                    dtype: str,
                    rows: int) -> pd.DataFrame:
    """*name*'s daily weights on its last *rows* days."""
    return weights_from(name, tables, currency, dtype, last=rows)


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
