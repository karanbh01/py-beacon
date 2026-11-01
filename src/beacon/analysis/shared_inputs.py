# src/beacon/analysis/shared_inputs.py
"""
A data store's market inputs, kept once for every run on it.

A run keeps the market inputs it read (closes, exchange rates, corporate
actions, currencies, traded values, characteristics and reference data), and
runs on the same data read mostly the same ones. Kept here, each is kept
once:

- **Pieces.** Each table is cut into pieces of one name's year (a currency's
  for the exchange rates; a name's whole entry for its currency), each named
  by a fingerprint of its contents. A run keeps the list of pieces it used.
  A repeat run on the same data adds nothing; a longer run adds the earlier
  years; a wider one the new names; a revised price is a new piece, so an
  older run keeps the numbers it ran on.
- **On disk.** One folder per data store: a catalogue (table, name, year,
  fingerprint, file, first row, rows) and a few files per table. Everything a
  run adds goes into one new file per table. `merge` rewrites the files as
  one per table, keeping only the pieces still in use, beside the old ones,
  and swaps the catalogue in; runs name fingerprints, not files, so a merge
  does not touch them.

A table read back from its pieces equals the one written, to the last bit:
the run keeps its days and names, or the order of its rows, beside its list.
"""
# BN-413. Agreed with Karan on 2026-10-29.
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .panels import ACTIONS, CHARACTERISTICS, CLOSES, CURRENCIES, RATES, REFERENCE, TRADED_VALUES

# The tables kept here, by how they are laid out: dates by names; dated rows
# with an IDENTIFIER column; rows with IDENTIFIER and EX_DATE columns; and a
# row a name.
WIDE = (CLOSES, RATES)
DATED = (TRADED_VALUES, CHARACTERISTICS, REFERENCE)
SHARED = (*WIDE, *DATED, ACTIONS, CURRENCIES)

CATALOGUE = "catalogue.parquet"
DELETED = "deleted.json"
# A folder with more files than this is due a merge.
MERGE_AFTER = 20
# A merge keeps what was written this recently whether a run lists it or not:
# a run saved alongside has added its pieces but may not have written its
# list yet.
GRACE_SECONDS = 3_600

# The long form every piece is kept in: the name, the day, and the values;
# while being written, its piece and its place.
NAME, DAY, VALUE = "NAME", "DAY", "VALUE"
_PIECE, _ORDER = "_piece", "_order"
CATALOGUE_COLUMNS = ("table", "name", "year", "fingerprint", "file", "start", "rows")
PIECE_COLUMNS = ("table", "fingerprint")

# How a dated table's rows are ordered when it is one of these: by day then
# name, or by name then day.
BY_DAY, BY_NAME = [DAY, NAME], [NAME, DAY]

# One lock a folder, so runs saved together in one process add their pieces
# one after another.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def folder_lock(folder: Path) -> threading.Lock:
    """The lock that runs adding to, merging or reading *folder* take."""
    key = str(folder.resolve())

    with _LOCKS_LOCK:
        return _LOCKS.setdefault(key, threading.Lock())


class SharedInputs:
    """A data store's shared folder of market inputs.

    Args:
        folder: The folder; created when the first pieces are added.
    """

    def __init__(self,
                 folder: Path):
        self.folder = Path(folder)

    def add(self,
            tables: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Keep the pieces of *tables* (those of `SHARED`) the folder does
        not hold yet.

        Returns:
            tuple: The pieces the tables are made of (``table`` and
            ``fingerprint``, a row each), and what rebuilds each table from
            them (its days and names, or the order of its rows), for the run
            to keep.
        """
        longs, pieces, layouts = {}, [], {}

        for table, frame in tables.items():
            if table in SHARED:
                long, layouts[table] = to_long(table, frame)
                longs[table], found = _pieces(table, long)
                pieces.append(found)

        used = (pd.concat(pieces, ignore_index=True) if pieces
                else _empty_catalogue()[list(CATALOGUE_COLUMNS[:4])])

        with folder_lock(self.folder):
            catalogue = self.catalogue()
            known = set(zip(catalogue["table"], catalogue["fingerprint"], strict=True))
            added = []

            for table, long in longs.items():
                found = used[used["table"] == table]
                new = found[[(table, int(fingerprint)) not in known
                             for fingerprint in found["fingerprint"]]]

                if not new.empty:
                    added.append(self._write(table, long[long[_PIECE].isin(new["fingerprint"])],
                                             new))

            if added:
                _write_atomic(pd.concat([catalogue, *added], ignore_index=True),
                              self.folder / CATALOGUE)

        return used[list(PIECE_COLUMNS)].reset_index(drop=True), layouts

    def read(self,
             table: str,
             fingerprints: Iterable[int],
             layout: Mapping[str, Any]) -> pd.DataFrame:
        """*table* put back together from its pieces *fingerprints*, as the
        run that kept *layout* wrote it.

        Raises:
            KeyError: If a piece is not in the folder.
        """
        wanted = {int(fingerprint) for fingerprint in fingerprints}

        with folder_lock(self.folder):
            catalogue = self.catalogue()
            rows = catalogue[(catalogue["table"] == table)
                             & catalogue["fingerprint"].isin(wanted)]

            if len(rows) != len(wanted):
                raise KeyError(f"{len(wanted) - len(rows)} of the run's {table} pieces are "
                               f"not in {self.folder}.")

            long = _rows(self.folder, rows)

        return from_long(table, long.drop(columns=[_PIECE]), layout)

    def catalogue(self) -> pd.DataFrame:
        """Every piece the folder holds: its table, name, year, fingerprint,
        file, first row and rows."""
        path = self.folder / CATALOGUE

        return pd.read_parquet(path) if path.exists() else _empty_catalogue()

    def files(self) -> list[str]:
        """The files of pieces the folder holds."""
        return sorted(set(self.catalogue()["file"]))

    def due_a_merge(self) -> bool:
        """Whether the folder holds more files than `MERGE_AFTER`."""
        return len(self.files()) > MERGE_AFTER

    def merge(self,
              in_use: Iterable[tuple[str, int]]) -> None:
        """Rewrite the folder as one file per table holding only the pieces
        *in_use* (table and fingerprint), and those in files written in the
        last `GRACE_SECONDS`: written beside the old files, the catalogue
        swapped in, then the old files removed."""
        keep = {(str(table), int(fingerprint)) for table, fingerprint in in_use}

        with folder_lock(self.folder):
            catalogue = self.catalogue()
            old = set(catalogue["file"])
            recent = {file for file in old if _recent(self.folder / file)}
            kept = catalogue[[(table, int(fingerprint)) in keep or file in recent
                              for table, fingerprint, file
                              in zip(catalogue["table"], catalogue["fingerprint"],
                                     catalogue["file"], strict=True)]]
            written = [self._write(str(table), _rows(self.folder, held), held)
                       for table, held in kept.groupby("table", sort=False)]
            _write_atomic(pd.concat(written, ignore_index=True) if written
                          else _empty_catalogue(), self.folder / CATALOGUE)

            for file in old:
                (self.folder / file).unlink(missing_ok=True)

    def mark_deleted(self,
                     store: Mapping[str, Any]) -> None:
        """Mark the folder as kept for a data store that was deleted."""
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / DELETED).write_text(json.dumps(dict(store), sort_keys=True),
                                           encoding="utf-8")

    def deleted(self) -> dict[str, Any] | None:
        """The deleted data store this folder was kept for, or None."""
        path = self.folder / DELETED

        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _write(self,
               table: str,
               long: pd.DataFrame,
               pieces: pd.DataFrame) -> pd.DataFrame:
        """*long*'s rows (piece after piece, as they come) in a new file of
        *table*'s; their pieces' catalogue rows."""
        self.folder.mkdir(parents=True, exist_ok=True)
        counts = long.groupby(_PIECE, sort=False).size()
        starts = np.concatenate([[0], np.cumsum(counts.to_numpy())[:-1]]).astype("int64")
        file = f"{table}.{uuid.uuid4().hex[:12]}.parquet"
        _write_atomic(long.drop(columns=[_PIECE]).reset_index(drop=True), self.folder / file)
        placed = pd.DataFrame({"fingerprint": counts.index.to_numpy(dtype="int64"),
                               "start": starts, "rows": counts.to_numpy(dtype="int64")})
        found = (pieces[["table", "name", "year", "fingerprint"]]
                 .drop_duplicates("fingerprint").merge(placed, on="fingerprint"))
        found["file"] = file

        return found[list(CATALOGUE_COLUMNS)]


def _recent(path: Path) -> bool:
    """Whether *path* was written in the last `GRACE_SECONDS`."""
    return path.exists() and time.time() - path.stat().st_mtime < GRACE_SECONDS


def deleted_warning(store: Mapping[str, Any]) -> str:
    """The warning a run on the deleted data store *store* opens with."""
    return (f"The run's market inputs came from the data store "
            f"{store.get('name') or store.get('id')!r}, which has been deleted; they "
            f"are kept for the runs on it.")


def _empty_catalogue() -> pd.DataFrame:
    return pd.DataFrame({"table": pd.Series(dtype=object), "name": pd.Series(dtype=object),
                         "year": pd.Series(dtype="int64"),
                         "fingerprint": pd.Series(dtype="int64"),
                         "file": pd.Series(dtype=object), "start": pd.Series(dtype="int64"),
                         "rows": pd.Series(dtype="int64")})


def _write_atomic(frame: pd.DataFrame,
                  path: Path) -> None:
    """Write *frame* beside *path* and swap it in, so a reader never sees
    half a file."""
    partial = path.with_name(path.name + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, path)


def _rows(folder: Path,
          pieces: pd.DataFrame) -> pd.DataFrame:
    """The rows of *pieces* (catalogue rows), piece after piece, each marked
    with its fingerprint."""
    parts = []

    for file, held in pieces.groupby("file", sort=False):
        frame = pd.read_parquet(folder / str(file))
        counts = held["rows"].to_numpy(dtype="int64")
        positions = np.concatenate([np.arange(start, start + rows) for start, rows
                                    in zip(held["start"], counts, strict=True)])
        part = frame.iloc[positions].reset_index(drop=True)
        part[_PIECE] = np.repeat(held["fingerprint"].to_numpy(dtype="int64"), counts)
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=[NAME, DAY, _PIECE])

    return pd.concat(parts, ignore_index=True)


def _pieces(table: str,
            long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """*long*'s rows piece after piece (each row's piece as it came), each
    marked with its piece's fingerprint, and a row a piece."""
    years = long[DAY].dt.year.fillna(0).astype("int64").to_numpy()
    names, labels = pd.factorize(long[NAME].astype(str))
    codes = names.astype("int64") * 10_000 + years
    order = np.lexsort((np.arange(len(long)), codes))
    long = long.iloc[order].reset_index(drop=True)
    codes = codes[order]
    hashed = pd.util.hash_pandas_object(long, index=False).to_numpy()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]) if len(codes) else codes
    ends = np.r_[starts[1:], len(codes)].astype("int64") if len(codes) else codes
    prefix = json.dumps([table, [str(column) for column in long.columns],
                         [str(dtype) for dtype in long.dtypes]]).encode()
    prints = np.array([_fingerprint(prefix, hashed[start:end])
                       for start, end in zip(starts, ends, strict=True)], dtype="int64")
    long[_PIECE] = np.repeat(prints, ends - starts)
    found = pd.DataFrame({"table": table,
                          "name": np.asarray(labels, dtype=object)[codes[starts] // 10_000],
                          "year": codes[starts] % 10_000, "fingerprint": prints})

    return long, found


def _fingerprint(prefix: bytes,
                 hashes: np.ndarray) -> int:
    """A piece's fingerprint: 64 bits of a hash of its rows' hashes."""
    digest = hashlib.blake2b(prefix + hashes.tobytes(), digest_size=8).digest()

    return int.from_bytes(digest, "little", signed=True)


def to_long(table: str,
            frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """*frame* as rows of a name, a day and its values, and what puts it
    back: its days and names, or how its rows are ordered, and its dtypes."""
    if table in WIDE:
        return _wide_to_long(frame)

    if table == CURRENCIES:
        long = frame.reset_index(names=NAME).assign(**{DAY: pd.NaT})
        long[NAME] = long[NAME].astype(str)
        layout = {"index_dtype": str(frame.index.dtype), "columns": list(frame.columns),
                  "dtypes": _dtypes(frame)}

        return long[[NAME, DAY, *frame.columns]], layout

    days = frame["EX_DATE"] if table == ACTIONS else frame.index
    long = frame.reset_index(drop=True)
    long[NAME] = frame["IDENTIFIER"].astype(str).to_numpy()
    long[DAY] = pd.DatetimeIndex(days)
    long = long[[NAME, DAY, *frame.columns]]
    layout = {"columns": list(frame.columns), "index_dtype": str(frame.index.dtype),
              "index_name": frame.index.name, "order": _order_of(long),
              "dtypes": _dtypes(frame)}

    return long, layout


def _dtypes(frame: pd.DataFrame) -> dict[str, str]:
    """Each column's dtype, as text."""
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _wide_to_long(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A table of dates by names as a row a held cell."""
    grid = frame.to_numpy()
    rows, columns = np.nonzero(~pd.isna(grid))
    names = np.asarray([str(name) for name in frame.columns], dtype=object)
    long = pd.DataFrame({NAME: names[columns], DAY: pd.DatetimeIndex(frame.index[rows]),
                         VALUE: grid[rows, columns]})
    layout = {"days": [day.isoformat() for day in frame.index],
              "index_name": frame.index.name, "index_dtype": str(frame.index.dtype),
              "names": names.tolist(), "names_dtype": str(frame.columns.dtype),
              "dtype": str(frame.dtypes.iloc[0]) if len(frame.columns) else "float64"}

    return long, layout


def _keys(long: pd.DataFrame,
          keys: list[str]) -> tuple[np.ndarray, ...]:
    """*keys* as `np.lexsort` takes them: the last sorts first."""
    found = []

    for key in reversed(keys):
        values = long[key]
        found.append(pd.DatetimeIndex(values).asi8 if key == DAY
                     else pd.factorize(values.astype(str), sort=True)[0])

    return tuple(found)


def _order_of(long: pd.DataFrame) -> Any:
    """How *long*'s rows are ordered: `BY_DAY` or `BY_NAME` when they are
    sorted so, else each row's place among them sorted by name then day."""
    places = np.arange(len(long))

    for keys in (BY_DAY, BY_NAME):
        if (np.lexsort(_keys(long, keys)) == places).all():
            return keys

    ranked = np.lexsort((places, *_keys(long, BY_NAME)))
    found = np.empty(len(long), dtype="int64")
    found[ranked] = places

    return {"rows": found.tolist()}


def from_long(table: str,
              long: pd.DataFrame,
              layout: Mapping[str, Any]) -> pd.DataFrame:
    """The table `to_long` was given, from its rows and *layout*."""
    if table in WIDE:
        return _long_to_wide(long, layout)

    if long.empty:
        # No pieces: the columns as they were, with no rows.
        long = pd.DataFrame({NAME: pd.Series(dtype=object), DAY: pd.Series(dtype="datetime64[ns]"),
                             **{column: pd.Series(dtype=dtype)
                                for column, dtype in layout["dtypes"].items()}})

    if table == CURRENCIES:
        frame = long.set_index(NAME)[list(layout["columns"])].astype(layout["dtypes"])
        frame.index = frame.index.astype(layout["index_dtype"])
        frame.index.name = None

        return frame

    order = layout["order"]
    places = np.arange(len(long))

    # Ties keep the order the rows came in, which within a piece is the
    # order they were written in.
    if isinstance(order, list):
        long = long.iloc[np.lexsort((places, *_keys(long, order)))]
    else:
        long = long.iloc[np.lexsort((places, *_keys(long, BY_NAME)))].iloc[order["rows"]]

    frame = long[list(layout["columns"])].reset_index(drop=True).astype(layout["dtypes"])

    if table == ACTIONS:
        frame.index = frame.index.astype(layout["index_dtype"])
    else:
        frame.index = pd.DatetimeIndex(long[DAY]).astype(layout["index_dtype"])
        frame.index.name = layout["index_name"]

    return frame


def _long_to_wide(long: pd.DataFrame,
                  layout: Mapping[str, Any]) -> pd.DataFrame:
    """A table of dates by names from its held cells."""
    days = pd.DatetimeIndex(layout["days"]).astype(layout["index_dtype"])
    days.name = layout["index_name"]
    names = pd.Index(layout["names"], dtype=layout["names_dtype"])
    grid = np.full((len(days), len(names)), np.nan, dtype=layout["dtype"])

    if long.empty:
        return pd.DataFrame(grid, index=days, columns=names)

    rows = days.get_indexer(pd.DatetimeIndex(long[DAY]).astype(layout["index_dtype"]))
    columns = names.get_indexer(long[NAME].astype(str))
    grid[rows, columns] = long[VALUE].to_numpy(dtype=layout["dtype"])

    return pd.DataFrame(grid, index=days, columns=names)
