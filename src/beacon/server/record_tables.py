# src/beacon/server/record_tables.py
"""
A backtest record's large parts, kept as tables beside it.

Almost all of a stored record is a handful of large parts: each book's daily
weights, the fund's positions and transactions, each index's rebalance
snapshots and the holdings document. The record keeps its own shape with
those parts left empty, and says under `TABLES_KEY` where each one is kept:

- a book's daily weights, in the run's own weights tables, which the run's
  analysis keeps anyway;
- the fund's units in the holdings document, in the fund's holdings, which
  the run keeps for the same reason;
- the positions and transactions, as a table each;
- the rebalance snapshots and the rest of the holdings document, as a table
  each of every value in them, which gives back the same JSON.

A part goes into a table only when what it reads back as is the part exactly,
checked as it is written; one that would not come back exactly stays in the
JSON. `whole_record` puts the parts back, so every response is the one the
whole record gave.
"""
# BN-404. At 6,000 names the record was 241.6 MB of JSON, of which 0.2 MB
# is not these parts.
import json
import shutil
import tempfile
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..analysis.data import RunData
from ..analysis.panels import (
    BENCHMARK_WEIGHTS,
    FUND_WEIGHTS,
    HOLDINGS,
    INDEX_WEIGHTS,
    OPTIMISED_WEIGHTS,
    TRADES,
)
from ..exceptions import ConfigurationError
from .holdings import HOLDINGS_KEY
from .schemas import MAX_WEIGHT_DATES
from .serialisation import dataframe_to_payload

# Where a slim record says where its parts are kept; a key its model does
# not read, as the holdings document is.
TABLES_KEY = "tables"

# The folder in the run's tables folder that holds the record's own tables.
RECORD_FOLDER = "record"

# A record whose parts were moved by this version of the layout.
LAYOUT_VERSION = 1

Where = tuple[str, ...]

# Where each book's daily weights are served from.
WEIGHT_PARTS: dict[Where, str] = {("portfolio", "weights"): FUND_WEIGHTS,
                                  ("index", "target", "weights"): INDEX_WEIGHTS,
                                  ("index", "optimised", "weights"): OPTIMISED_WEIGHTS,
                                  ("benchmark", "weights"): BENCHMARK_WEIGHTS}

# The fund's units on the days they changed, served from its holdings.
UNITS_PART: Where = (HOLDINGS_KEY, "units")

# The fund's transactions, served from the run's trades (BN-406).
TRANSACTIONS_PART: Where = ("portfolio", "transactions")

# Parts kept as a table with a column each.
FRAME_PARTS: tuple[Where, ...] = (("portfolio", "positions"), TRANSACTIONS_PART)

# Parts kept as a table of every value in them.
TREE_PARTS: tuple[Where, ...] = (("index", "target", "rebalances"),
                                 ("index", "optimised", "rebalances"),
                                 ("benchmark", "rebalances"),
                                 (HOLDINGS_KEY, "implementation"),
                                 (HOLDINGS_KEY, "selections"))

# The tree table's columns: where a value sits, what kind it is, and it.
PARENT, KEY, KIND, NUMBER, INTEGER, TEXT = "PARENT", "KEY", "KIND", "NUMBER", "INTEGER", "TEXT"
INDEX = "__index__"


def slim_record(document: dict[str, Any],
                run: RunData,
                folder: Path) -> dict[str, Any]:
    """*document* with its large parts left empty and kept as tables in
    *folder*, the run's tables folder, where `whole_record` finds them.

    A book's daily weights are served from *run*'s weights tables, so the
    document may carry them empty already: what is kept is how many of the
    latest days it serves, `weights_dates_total` of them up to the record's
    bound. Every other part is moved only when what it reads back as is the
    part exactly.

    Args:
        document: The record, as it is stored whole.
        run: The run's analysis data, as written to *folder*.
        folder: The run's tables folder.
    """
    slim = _copy_along(document, [*WEIGHT_PARTS, UNITS_PART, *FRAME_PARTS, *TREE_PARTS])
    parts: list[dict[str, Any]] = []
    staging = _staging(folder)

    try:
        for path, table in WEIGHT_PARTS.items():
            parts.extend(_weights_part(slim, path, table, run))

        parts.extend(_units_part(slim, run))
        parts.extend(_transactions_part(slim, run))

        for number, path in enumerate([*FRAME_PARTS, *TREE_PARTS]):
            parts.extend(_table_part(slim, path, staging / f"part_{number}.parquet"))

        _replace(folder / RECORD_FOLDER, staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    slim[TABLES_KEY] = {"version": LAYOUT_VERSION, "parts": parts}

    return slim


def whole_record(document: dict[str, Any],
                 folder: Path,
                 wanted: Collection[str] | None = None) -> dict[str, Any]:
    """*document* with the parts `slim_record` moved put back, or as it is
    when nothing was moved, as a record stored before.

    Args:
        document: The record as it is stored.
        folder: The run's tables folder.
        wanted: The record's top-level keys to put back, every one when
            None; the rest stay empty.

    Raises:
        ConfigurationError: If a part's table is missing or cannot be read.
    """
    layout = document.get(TABLES_KEY)

    if not layout:
        return document

    parts = [part for part in layout["parts"] if wanted is None or part["path"][0] in wanted]
    whole = _copy_along(document, [tuple(part["path"]) for part in parts])
    whole.pop(TABLES_KEY, None)
    run: RunData | None = None

    try:
        for part in parts:
            # A part whose home is gone is left out rather than failing.
            if not isinstance(_get(whole, tuple(part["path"][:-1])), dict):
                continue

            if run is None and ("weights" in part or "units" in part or "trades" in part):
                run = RunData.from_record(folder)

            _set(whole, part["path"], _restored(part, run, folder / RECORD_FOLDER))
    except (OSError, KeyError, ValueError, pa.ArrowException) as error:
        raise ConfigurationError(str(folder), f"a backtest record's tables cannot be read: "
                                              f"{error}") from error

    return whole


def _weights_part(slim: dict[str, Any],
                  path: Where,
                  table: str,
                  run: RunData) -> list[dict[str, Any]]:
    """Where a book's daily weights are served from, if it has any."""
    book = _get(slim, path[:-1])
    total = int((book or {}).get("weights_dates_total") or 0)

    if not total or table not in run.tables:
        return []

    _set(slim, path, _empty_like({"index": [], "columns": [], "data": []}))

    return [{"path": list(path), "weights": table, "rows": min(total, MAX_WEIGHT_DATES)}]


def _units_part(slim: dict[str, Any],
                run: RunData) -> list[dict[str, Any]]:
    """The fund's units, served from its holdings when they are the same."""
    units = _get(slim, UNITS_PART)
    table = FUND_WEIGHTS + HOLDINGS

    if not units or table not in run.tables or units_from(run.tables[table]) != units:
        return []

    _set(slim, UNITS_PART, {})

    return [{"path": list(UNITS_PART), "units": table}]


def _transactions_part(slim: dict[str, Any],
                       run: RunData) -> list[dict[str, Any]]:
    """The fund's transactions, served from the run's trades when they are
    the same."""
    value = _get(slim, TRANSACTIONS_PART)

    if not value or TRADES not in run.tables or transactions_from(run.tables[TRADES]) != value:
        return []

    _set(slim, TRANSACTIONS_PART, _empty_like(value))

    return [{"path": list(TRANSACTIONS_PART), "trades": TRADES}]


def transactions_from(trades: Any) -> dict[str, Any]:
    """The run's trades as the record's transactions give them."""
    frame = pd.DataFrame({"date": pd.DatetimeIndex(trades.index),
                          "asset_id": trades["IDENTIFIER"].astype(str).to_numpy(),
                          "type": trades["SIDE"].astype(str).to_numpy(),
                          "quantity": trades["QUANTITY"].to_numpy(dtype=float),
                          "price": trades["PRICE"].to_numpy(dtype=float),
                          "cost": trades["COST"].to_numpy(dtype=float)},
                         columns=["date", "asset_id", "type", "quantity", "price", "cost"])

    # As the stored record reads back: through JSON.
    found: dict[str, Any] = json.loads(json.dumps(dataframe_to_payload(frame)))

    return found


def units_from(changes: Any) -> dict[str, list[list[Any]]]:
    """Each name's units on the days they changed, 0 from the day it left,
    from the fund's holdings, as the holdings document gives them."""
    units: dict[str, list[list[Any]]] = {}
    days = changes.index.strftime("%Y-%m-%d").tolist()
    names = changes["NAME"].astype(str).tolist()
    amounts = changes["AMOUNT"].astype(float).tolist()
    held = changes["HELD"].astype(bool).tolist()

    for day, name, amount, kept in zip(days, names, amounts, held, strict=True):
        units.setdefault(name, []).append([day, amount if kept else 0.0])

    return units


def _table_part(slim: dict[str, Any],
                path: Where,
                file: Path) -> list[dict[str, Any]]:
    """A part kept as a table, when it reads back exactly; else none."""
    value = _get(slim, path)

    if not value or (isinstance(value, dict) and not value.get("data", True)):
        return []

    for kind, encode, decode in _ENCODINGS[_kind_of(value)]:
        try:
            table = encode(value)
        except (pa.ArrowException, OverflowError):
            # A value no table column holds, such as a whole number past
            # 64 bits; the part stays in the JSON.
            continue

        if table is not None and decode(table) == value:
            pq.write_table(table, file, compression="zstd", compression_level=3)
            _set(slim, path, _empty_like(value))

            return [{"path": list(path), kind: file.name}]

    return []


def _kind_of(value: Any) -> str:
    """Whether *value* is a frame (``{index, columns, data}``) or a tree."""
    if isinstance(value, dict) and set(value) == {"index", "columns", "data"}:
        return "frame"

    return "tree"


def _empty_like(value: Any) -> Any:
    """What a part is left as in the slim record."""
    if _kind_of(value) == "frame":
        return {"index": [], "columns": [], "data": []}

    return {} if isinstance(value, dict) else []


def _restored(part: dict[str, Any],
              run: RunData | None,
              folder: Path) -> Any:
    """A moved part, as it was."""
    if "weights" in part and run is not None:
        return dataframe_to_payload(run.latest(part["weights"], int(part["rows"])))

    if "units" in part and run is not None:
        return units_from(run.tables[part["units"]])

    if "trades" in part and run is not None:
        return transactions_from(run.tables[part["trades"]])

    if "frame" in part:
        return frame_from(pq.read_table(folder / part["frame"]))

    return tree_from(pq.read_table(folder / part["tree"]))


def frame_table(frame: dict[str, Any]) -> pa.Table | None:
    """A frame's index and columns as a table, or None when a column holds
    values of more than one kind (a number beside text), which a table
    column cannot."""
    columns = list(zip(*frame["data"], strict=True)) if frame["data"] else [
        () for _ in frame["columns"]]
    arrays: dict[str, pa.Array] = {}

    try:
        for values in [frame["index"], *columns]:
            array = pa.array(list(values))
            arrays[f"{len(arrays)}"] = (array.dictionary_encode() if pa.types.is_string(array.type)
                                        else array)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return None

    names = [INDEX, *(str(column) for column in frame["columns"])]

    return pa.table(arrays).replace_schema_metadata({"names": _STEP.join(names)})


def frame_from(table: pa.Table) -> dict[str, Any]:
    """The frame `frame_table` kept."""
    names = (table.schema.metadata or {}).get(b"names", b"").decode().split(_STEP)
    columns = [_values(table.column(position)) for position in range(table.num_columns)]

    return {"index": columns[0], "columns": names[1:],
            "data": [list(row) for row in zip(*columns[1:], strict=True)]}


def tree_table(tree: Any) -> pa.Table:
    """Every value in *tree*, a row each in the order they come, beneath a
    row for each list and dict that holds them."""
    rows = list(_leaves(tree, ""))
    parents, keys, kinds, numbers, integers, texts = (list(column) for column in zip(*rows,
                                                                                     strict=True))

    return pa.table({PARENT: pa.array(parents).dictionary_encode(),
                     KEY: pa.array(keys).dictionary_encode(),
                     KIND: pa.array(kinds).dictionary_encode(),
                     NUMBER: pa.array(numbers, pa.float64()),
                     INTEGER: pa.array(integers, pa.int64()),
                     TEXT: pa.array(texts, pa.string()).dictionary_encode()})


def _leaves(node: Any,
            parent: str) -> Iterator[tuple[str, str, str, float | None, int | None, str | None]]:
    """*node*'s contents as tree rows, *parent* naming where it sits."""
    items = node.items() if isinstance(node, dict) else enumerate(node)

    for key, value in items:
        if isinstance(value, dict | list):
            yield parent, str(key), "dict" if isinstance(value, dict) else "list", None, None, None
            yield from _leaves(value, f"{parent}{_STEP}{key}")
        elif isinstance(value, bool):
            yield parent, str(key), "bool", None, int(value), None
        elif isinstance(value, int):
            yield parent, str(key), "int", None, value, None
        elif isinstance(value, float):
            yield parent, str(key), "float", value, None, None
        elif value is None:
            yield parent, str(key), "null", None, None, None
        else:
            yield parent, str(key), "str", None, None, str(value)


def tree_from(table: pa.Table) -> Any:
    """The tree `tree_table` kept."""
    columns = [_values(table.column(name)) for name in (PARENT, KEY, KIND, NUMBER, INTEGER, TEXT)]
    root: dict[str, Any] = {}
    homes: dict[str, Any] = {"": root}

    for parent, key, kind, number, integer, text in zip(*columns, strict=True):
        value = _value(kind, number, integer, text)

        if kind == "dict" or kind == "list":
            homes[f"{parent}{_STEP}{key}"] = value

        home = homes[parent]

        if type(home) is list:
            home.append(value)
        else:
            home[key] = value

    return root[_ROOT]


def _value(kind: str,
           number: float | None,
           integer: int | None,
           text: str | None) -> Any:
    """One tree row's value."""
    if kind == "float":
        return number
    if kind == "str":
        return text
    if kind == "int":
        return integer
    if kind == "dict":
        return {}
    if kind == "list":
        return []
    if kind == "bool":
        return bool(integer)

    return None


def _values(column: pa.ChunkedArray) -> list[Any]:
    """A column's values as Python's, a whole column at a time: a text
    column through its dictionary, a number column through numpy."""
    array = column.combine_chunks()

    if pa.types.is_dictionary(array.type):
        lookup = np.array([*array.dictionary.to_pylist(), None], dtype=object)
        codes = pc.fill_null(array.indices, len(lookup) - 1).to_numpy()

        return list(lookup[codes].tolist())

    if not (pa.types.is_floating(array.type) or pa.types.is_integer(array.type)):
        return list(array.to_pylist())

    values = list(pc.fill_null(array, 0).to_numpy().tolist())

    for position in np.flatnonzero(array.is_null().to_numpy(zero_copy_only=False)).tolist():
        values[position] = None

    return values


# What joins the keys on the way to a value in a tree table's PARENT;
# no identifier, date or field name holds it.
_STEP = "\x1f"

# The single key a tree is kept under, so its own top level is a value.
_ROOT = "root"

# How each kind of part may be kept, in the order tried.
_ENCODINGS: dict[str, list[tuple[str, Any, Any]]] = {
    "frame": [("frame", frame_table, frame_from),
              ("tree", lambda value: tree_table({_ROOT: value}), tree_from)],
    "tree": [("tree", lambda value: tree_table({_ROOT: value}), tree_from)],
}


def _get(document: dict[str, Any],
         path: Where) -> Any:
    """The value at *path*, or None where any step is missing."""
    node: Any = document

    for key in path:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]

    return node


def _set(document: dict[str, Any],
         path: list[str] | Where,
         value: Any) -> None:
    """Put *value* at *path*, whose parents exist."""
    node = document

    for key in path[:-1]:
        node = node[key]

    node[path[-1]] = value


def _copy_along(document: dict[str, Any],
                paths: list[Where]) -> dict[str, Any]:
    """*document* with each dict on the way to *paths* copied, so setting a
    part leaves the document as it was."""
    copy = dict(document)

    for path in paths:
        node = copy

        for key in path[:-1]:
            if not isinstance(node.get(key), dict):
                break
            node[key] = dict(node[key])
            node = node[key]

    return copy


def _staging(folder: Path) -> Path:
    """A new folder beside where the record's tables go."""
    folder.mkdir(parents=True, exist_ok=True)

    return Path(tempfile.mkdtemp(dir=folder, prefix=f".{RECORD_FOLDER}."))


def _replace(target: Path,
             staging: Path) -> None:
    """*staging* in place of *target*."""
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
