# src/beacon/analysis/shared_runs.py
"""
The stored runs that keep their market inputs in a data store's shared
folder (`beacon.analysis.shared_inputs`): which runs read a store, the pieces
they use, what deleting the store leaves, and moving a run that keeps its
inputs itself into the shared layout.

Each function takes the stored runs as a mapping of a name (a strategy, say)
to the folder the run was written to.
"""
# BN-413. Agreed with Karan on 2026-10-29.
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .data import RUN_FILE, RunData
from .panels import MANIFEST_FILE, read_manifest, read_panels
from .shared_inputs import SHARED, SharedInputs, deleted_warning
from .stored_tables import SHARED_PIECES

logger = logging.getLogger(__name__)

# Where the shared folders are kept, one a data store, under a root.
FOLDER = "market_inputs"
# The folder for runs kept before their store was recorded.
UNATTRIBUTED = "unattributed"


def shared_folder(root: Path,
                  store_id: str) -> SharedInputs:
    """The shared folder of the data store *store_id* under *root*."""
    return SharedInputs(Path(root).joinpath(FOLDER, store_id))


def store_of(folder: Path) -> str | None:
    """The data store whose shared folder the run in *folder* keeps its
    inputs in, or None for a run that keeps them itself."""
    inputs = _inputs(folder)

    return None if inputs is None else (inputs.get("store") or {}).get("id")


def runs_on(runs: Mapping[str, Path],
            store_id: str) -> list[str]:
    """The *runs* that keep their inputs in *store_id*'s folder."""
    return [name for name, folder in runs.items() if store_of(folder) == store_id]


def in_use(runs: Mapping[str, Path],
           store_id: str) -> set[tuple[str, int]]:
    """Every piece (table and fingerprint) the *runs* on *store_id* use."""
    used: set[tuple[str, int]] = set()

    for name in runs_on(runs, store_id):
        pieces = read_panels(runs[name], [SHARED_PIECES])[SHARED_PIECES]
        used.update(zip(pieces["table"], pieces["fingerprint"].astype("int64"), strict=True))

    return used


def merge_if_due(runs: Mapping[str, Path],
                 shared: SharedInputs,
                 store_id: str) -> bool:
    """Merge *shared*, keeping the pieces *runs* use, when it holds too many
    files; whether it did."""
    if not shared.due_a_merge():
        return False

    shared.merge(in_use(runs, store_id))
    logger.info("Merged the shared market inputs of data store %s.", store_id)

    return True


def forget_store(runs: Mapping[str, Path],
                 root: Path,
                 store: Mapping[str, Any],
                 cascade: bool = False) -> dict[str, Any]:
    """What deleting the data store *store* leaves of its shared folder:
    with runs on it, kept and marked as for a deleted store, unless
    *cascade*, which removes it, as does no run using it.

    Returns:
        dict: ``on_it``, the runs on the store (the caller deletes them
        under *cascade*), and ``warning``, said when they are kept.
    """
    store_id = str(store["id"])
    on_it = runs_on(runs, store_id)
    shared = shared_folder(root, store_id)

    if on_it and not cascade:
        shared.mark_deleted({"id": store_id, "name": store.get("name")})
        count = len(on_it)
        warning = (f"{count} stored {'run reads' if count == 1 else 'runs read'} this "
                   f"store's data; their market inputs are kept.")

        return {"on_it": on_it, "warning": warning}

    shutil.rmtree(shared.folder, ignore_errors=True)

    return {"on_it": on_it, "warning": None}


def warnings_of(folder: Path) -> list[str]:
    """What the run in *folder* says as it opens, without opening it: that
    its data store was deleted, say."""
    inputs = _inputs(folder)

    if not inputs:
        return []

    deleted = SharedInputs(folder.joinpath(inputs["folder"]).resolve()).deleted()

    return [] if deleted is None else [deleted_warning(deleted)]


def migrate(runs: Mapping[str, Path],
            root: Path) -> list[str]:
    """Move each of *runs* that keeps its market inputs itself to the shared
    layout, under ``unattributed`` (the run did not record its store),
    checking every table reads back the same first; the runs moved."""
    shared = shared_folder(root, UNATTRIBUTED)
    moved = []

    for name, folder in runs.items():
        if not _keeps_its_own(folder):
            continue

        try:
            if _move(folder, shared):
                moved.append(name)
        except Exception:
            logger.exception("Could not move the market inputs of %s's run; it keeps "
                             "its own.", name)

    if moved:
        logger.info("Moved %d stored runs' market inputs to the shared layout.", len(moved))

    return moved


def _inputs(folder: Path) -> dict[str, Any] | None:
    """What the run in *folder* says of its shared inputs; None when there
    is no run there or it keeps them itself."""
    path = folder / RUN_FILE

    if not path.exists():
        return None

    found = json.loads(path.read_text(encoding="utf-8")).get("inputs")

    return dict(found) if found else None


def _keeps_its_own(folder: Path) -> bool:
    """Whether a run is kept in *folder* with its market inputs in it."""
    return (folder / RUN_FILE).exists() and _inputs(folder) is None


def _move(folder: Path,
          shared: SharedInputs) -> bool:
    """Write the run in *folder* again with its inputs in *shared*, swapped
    in once every table reads back the same; whether it held any."""
    run = RunData.from_record(folder)
    tables = {name: run.tables[name] for name in run.tables}

    if not any(name in tables for name in SHARED):
        return False

    staged = folder.with_name(folder.name + ".moving")
    shutil.rmtree(staged, ignore_errors=True)
    run.write(staged, shared=shared, store={"id": UNATTRIBUTED, "name": None})
    again = RunData.from_record(staged)

    for name, table in tables.items():
        if not _same(again.tables[name], table):
            shutil.rmtree(staged, ignore_errors=True)
            raise ValueError(f"the {name} table would not read back the same")

    _copy_extras(folder, staged)
    old = folder.with_name(folder.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    folder.rename(old)
    staged.rename(folder)
    shutil.rmtree(old, ignore_errors=True)

    return True


def _copy_extras(folder: Path,
                 staged: Path) -> None:
    """What a run's folder holds besides the run's own tables (a record's
    tables, kept analyses, factor work), copied to *staged*."""
    own = {f"{name}.parquet" for name in (read_manifest(folder) or {}).get("panels", {})}
    own.update({MANIFEST_FILE, RUN_FILE})

    for path in folder.iterdir():
        target = staged / path.name

        if path.name in own or target.exists():
            continue

        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def _same(found: pd.DataFrame,
          expected: pd.DataFrame) -> bool:
    return (found.equals(expected) and found.index.equals(expected.index)
            and list(found.dtypes.astype(str)) == list(expected.dtypes.astype(str)))
