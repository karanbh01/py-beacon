# src/beacon/server/market_inputs.py
"""
Each data store's market inputs, kept once for every run on it.

A backtest's run keeps the market inputs it read in its data store's shared
folder (`beacon.analysis.shared_inputs`), under ``market_inputs/<store id>``
in the server's storage, and only the list of pieces it used beside its
record. A run on data from no registered store keeps its inputs itself, as
before. The bookkeeping is `beacon.analysis.shared_runs`'; this gives it the
stored runs.

- **Merging.** A folder holding more than 20 files is merged into one file
  per table, dropping the pieces no run uses.
- **Deleting a store.** The runs on it keep their pieces, and the folder is
  marked as kept for a deleted store, which such a run says when it opens;
  or, with cascade, the runs are deleted, and the folder with them.
- **Runs stored before.** A run that kept its inputs itself is moved to the
  shared layout when the server starts, under ``unattributed``, and reads
  the same.
"""
# BN-413. Agreed with Karan on 2026-10-29.
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..analysis import shared_runs
from ..analysis.shared_inputs import SharedInputs
from .store import DocumentStore


def root_of(records: DocumentStore) -> Path:
    """The storage root the records are kept under, where the shared
    folders are."""
    return records.directory.parent


def stored_runs(records: DocumentStore) -> dict[str, Path]:
    """Each stored strategy's run folder."""
    return {strategy: records.panels_directory(strategy) for strategy in records.list_ids()}


def shared_for(records: DocumentStore,
               store: Mapping[str, Any] | None) -> SharedInputs | None:
    """The shared folder of the data store *store* (its ``id``), or None for
    data from no registered store."""
    if store is None or not store.get("id"):
        return None

    return shared_runs.shared_folder(root_of(records), str(store["id"]))


def merge_if_due(records: DocumentStore,
                 shared: SharedInputs,
                 store: Mapping[str, Any]) -> bool:
    """Merge *shared* when it holds too many files; whether it did."""
    return shared_runs.merge_if_due(stored_runs(records), shared, str(store["id"]))


def forget_store(records: DocumentStore,
                 store: Mapping[str, Any],
                 cascade: bool = False) -> dict[str, Any]:
    """Deleting the data store *store*: the runs on it keep their market
    inputs, or with *cascade* are deleted.

    Returns:
        dict: ``kept`` and ``deleted``, the strategies whose runs were kept
        and deleted, and the ``warning`` said when runs were kept.
    """
    found = shared_runs.forget_store(stored_runs(records), root_of(records), store, cascade)

    if cascade:
        for strategy in found["on_it"]:
            records.delete(strategy)

    return {"kept": [] if cascade else found["on_it"],
            "deleted": found["on_it"] if cascade else [], "warning": found["warning"]}


def warnings_of(folder: Path) -> list[str]:
    """What a stored run says as it opens."""
    return shared_runs.warnings_of(folder)


def migrate(records: DocumentStore) -> list[str]:
    """Move the stored runs that keep their market inputs themselves to the
    shared layout; the strategies moved."""
    return shared_runs.migrate(stored_runs(records), root_of(records))
