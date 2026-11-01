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

Written with a data store's shared folder (`beacon.analysis.shared_inputs`),
a run keeps its market inputs there, once for every run on that store, and
only the list of pieces it used itself; it reads them back from there, and
says so when the store has since been deleted.
"""
# BN-369.
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
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
    HOLDINGS,
    INDEX_WEIGHTS,
    NAME_RETURNS,
    REFERENCE,
    TRADED_VALUES,
    TRADES,
    WEIGHT_TABLES,
    read_manifest,
    read_panels,
    run_panels,
    write_panels,
)
from .shared_inputs import SHARED, SharedInputs
from .stored_tables import (
    FUND_HOLDINGS,
    HOLDINGS_KEPT,
    SHARED_PIECES,
    VALUE,
    StoredTables,
    add_shared,
    column_of,
    holdings_from,
    packed_holdings,
    packed_trades,
    read_one,
    rebuilds_of,
    relative_folder,
    trades_from,
    window_of,
)
from .trade_packing import PACKED_TRADES, TRADES_KEPT

logger = logging.getLogger(__name__)

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

# The tables a stored run works out again from the ones it keeps, when it
# kept what they are worked out from.
DERIVED = frozenset({NAME_RETURNS, *WEIGHT_TABLES})

# The tables every run keeps, which a record without them says it lacks.
EXPECTED = (FUND_WEIGHTS, NAME_RETURNS, COSTS, CASH, INDEX_WEIGHTS, REFERENCE,
            CHARACTERISTICS, TRADED_VALUES, TRADES)

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
        warnings: What a stored run says as it opens: that the data store
            its market inputs came from was deleted, say.
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
    warnings: tuple[str, ...] = ()

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
                   investor_flows=result._investor_flows() if result.flows else None)

    @classmethod
    def of_returns(cls,
                   returns: pd.Series,
                   periods_per_year: int,
                   risk_free_rate: float = 0.0,
                   comparators: Mapping[str, pd.Series] | None = None,
                   currency: str = "") -> "RunData":
        """A run known only by its daily returns (the first from its start)
        and its comparators' levels, for the figures that need nothing more:
        its headline figures, measured from a start of 1, its comparisons and
        its figures by period. It has no tables."""
        path = (1.0 + returns).cumprod()

        return cls(currency=currency,
                   returns=returns,
                   nav=path,
                   units=pd.Series(1.0, index=returns.index),
                   initial_capital=1.0,
                   periods_per_year=periods_per_year,
                   risk_free_rate=risk_free_rate,
                   comparators=dict(comparators or {}),
                   performance=path,
                   performance_start=1.0)

    def latest(self,
               name: str,
               rows: int) -> pd.DataFrame:
        """The last *rows* rows of table *name*. A stored run's daily weights
        are worked out for those days alone, unless already read whole, and
        are those rows of the whole table exactly."""
        if isinstance(self.tables, StoredTables):
            return self.tables.latest(name, rows)

        return self.tables[name].tail(rows)

    def write(self,
              folder: Path,
              shared: SharedInputs | None = None,
              store: Mapping[str, Any] | None = None) -> None:
        """Keep this run in *folder*, replacing whatever it held: its tables
        and series as `beacon.analysis.panels` writes them, and the rest in
        ``run.json``.

        Args:
            folder: Where the run is kept.
            shared: The shared folder of the data store the run read, which
                keeps its market inputs instead of *folder*; None to keep
                them in *folder*.
            store: The data store the run read (``id`` and ``name``), kept
                with the run when *shared* is given.
        """
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
        inputs = None

        if shared is not None:
            # BN-413: the market inputs once for every run on the store.
            pieces, layouts = shared.add({name: self.tables[name] for name in SHARED
                                          if name in self.tables})
            kept = {name: table for name, table in kept.items() if name not in layouts}
            kept[SHARED_PIECES] = pieces
            inputs = {"folder": relative_folder(shared.folder, folder), "store": dict(store or {}),
                      "layouts": layouts}
        elif closes is not None:
            kept.update(pack_closes(closes))

        # The trades are kept as what was traded, the rest worked out again
        # on reading (BN-412), when that gives them back exactly.
        trades = packed_trades(self.tables, self.currency)

        if trades is not None:
            del kept[TRADES]
            kept.update(trades[0])

        # The fund's holdings are worked out again from its trades, when
        # that gives them back exactly (BN-413).
        holdings = packed_holdings(self.tables)

        if holdings is not None:
            del kept[FUND_HOLDINGS]
            kept.update({HOLDINGS_KEPT[part]: table for part, table in holdings.items()})

        write_panels(folder, {**kept,
                              **{name: values.to_frame(VALUE) for name, values in series.items()}})
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

        if inputs is not None:
            run["inputs"] = inputs

        if trades is not None:
            run["trades"] = {"rate": trades[1], "columns": list(self.tables[TRADES].columns)}

        if holdings is not None:
            run["holdings"] = {"dtypes": {str(column): str(dtype) for column, dtype
                                          in self.tables[FUND_HOLDINGS].dtypes.items()}}

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
        series = {name: column_of(table) for name, table in read_panels(
            folder, [name for name in held if name in (RETURNS, NAV, UNITS, PERFORMANCE,
                                                       INVESTOR_FLOWS)
                     or name.startswith(_LEVELS)]).items()}
        packed = [name for name in (CLOSES_PACKED, CLOSES_SCALE, CLOSES_FLOAT) if name in held]
        kept = [name for name in held if name not in series and name not in packed
                and name not in PACKED_TRADES and name != SHARED_PIECES
                and name not in HOLDINGS_KEPT.values()]
        tables = StoredTables({}, {name: partial(read_one, folder, name) for name in kept})
        warnings = add_shared(tables, folder, run.get("inputs"))

        if packed:
            tables.add(CLOSES, lambda: unpack_closes(read_panels(folder, packed),
                                                     run.get("closes") or None))

        if TRADES_KEPT in held:
            tables.add(TRADES, partial(trades_from, folder, tables, run))

        if run.get("holdings"):
            tables.add(FUND_HOLDINGS, partial(holdings_from, folder, tables, run))

        for name, rebuild in rebuilds_of(tables, run).items():
            tables.add(name, rebuild, window_of(name, tables, run))

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
                   investor_flows=series.get(INVESTOR_FLOWS),
                   warnings=warnings)


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
