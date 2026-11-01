# src/beacon/analysis/shared_indices.py
"""
An index's results within a run, kept once for every run that tracked it.

Strategies tracking the same index on the same data store over the same days
work out the same index: its target book's holdings, days, names and patch
(`beacon.analysis.holdings`), and its optimised book's when it has one. Kept
in the store's shared folder (`beacon.analysis.shared_inputs`) under
``indices/<fingerprint>``, they are kept once. The fingerprint is of the
tables themselves, so a changed definition, other data or other days give
another entry, and an older run keeps the index it ran against.
"""
# BN-414. Agreed with Karan on 2026-10-29.
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from .panels import (
    DAYS,
    HOLDINGS,
    INDEX_WEIGHTS,
    NAMES,
    OPTIMISED_WEIGHTS,
    PATCH,
    read_manifest,
    read_panels,
    write_panels,
)
from .shared_inputs import GRACE_SECONDS, folder_lock

INDICES = "indices"
# The tables of an index's results, each book's.
PARTS = tuple(book + part for book in (INDEX_WEIGHTS, OPTIMISED_WEIGHTS)
              for part in (HOLDINGS, DAYS, NAMES, PATCH))


def index_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The tables of *tables* that are the index's results."""
    return {name: table for name, table in tables.items() if name in PARTS}


def fingerprint(tables: Mapping[str, pd.DataFrame]) -> str:
    """A fingerprint of *tables*, their names, columns, dtypes, index and
    every value."""
    digest = hashlib.blake2b(digest_size=16)

    for name in sorted(tables):
        table = tables[name]
        digest.update(json.dumps([name, [str(column) for column in table.columns],
                                  [str(dtype) for dtype in table.dtypes],
                                  str(table.index.dtype), table.index.name]).encode())
        digest.update(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes())

    return digest.hexdigest()


def keep(shared_folder: Path,
         tables: Mapping[str, pd.DataFrame]) -> str:
    """Keep *tables* (an index's results) in the shared folder, unless an
    entry with their fingerprint is there already; the fingerprint."""
    key = fingerprint(tables)
    folder = Path(shared_folder) / INDICES / key

    with folder_lock(Path(shared_folder)):
        if read_manifest(folder) is None:
            write_panels(folder, dict(tables))
        else:
            # Kept again: as recent as if written now, so a prune spares it.
            os.utime(folder)

    return key


def read(shared_folder: Path,
         key: str,
         names: Iterable[str]) -> dict[str, pd.DataFrame]:
    """The index's results kept under *key*.

    Raises:
        KeyError: If the folder holds no such entry.
    """
    folder = Path(shared_folder) / INDICES / key

    if read_manifest(folder) is None:
        raise KeyError(f"The index results {key} are not in {shared_folder}.")

    return read_panels(folder, list(names))


def keys(shared_folder: Path) -> list[str]:
    """Every entry the shared folder holds."""
    folder = Path(shared_folder) / INDICES

    return sorted(path.name for path in folder.iterdir() if path.is_dir()) if folder.exists() \
        else []


def prune(shared_folder: Path,
          in_use: Iterable[str]) -> list[str]:
    """Remove the entries no run uses, sparing those written in the last
    `GRACE_SECONDS`; those removed."""
    used = set(in_use)
    root = Path(shared_folder) / INDICES

    with folder_lock(Path(shared_folder)):
        gone = [key for key in keys(shared_folder) if key not in used
                and time.time() - (root / key).stat().st_mtime >= GRACE_SECONDS]

        for key in gone:
            shutil.rmtree(root / key, ignore_errors=True)

    return gone
