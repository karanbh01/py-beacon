# src/beacon/data/store.py
"""
A persisted market-data store on disk.

A store is a directory holding a `manifest.json` and one Parquet file per
dataset: `market.parquet` (required), and `reference.parquet`,
`corporate_actions.parquet` and `features.parquet` when the data has them.
:func:`save` writes one from a `DataFetcher` and :func:`load` reads one back.
A server started with `python -m beacon.server` loads the store at
:func:`default_path` (or the path it is given), which is what
`python -m beacon.synthetic` writes.

Parquet keeps each column's type, so a store reads back without parsing a
date or a number, and packs smaller than gzipped CSV. A store saved by an
earlier version, as gzipped CSV (schema version 1), is converted the first
time it is loaded. `schema_version` is how the format changes without
stranding anyone's store: a store written by a newer version is refused
rather than misread.

## Byte-for-byte reproducibility

Two runs of the same generator produce the same store. Parquet as written
here carries no timestamps, the rows go into row groups of a fixed size, and
the manifest's keys are written sorted.

Timestamps are deliberately *not* recorded in the manifest. The store's age is
the file's mtime, and `DataFetcher` stamps its own refresh time when it loads.
Writing a creation time into the manifest would be a third answer to a
question that already has two, and would make every store byte-unique.
"""
# Why a store at all: data once reached the server only by being loaded into
# `MarketData` by a caller that built the process. `python -m beacon.server`
# builds its own process, so it had no such caller and always started
# data-less: every data endpoint answered `CONFIGURATION_ERROR`, and sync
# could not bootstrap because there was nothing to sync *into*.
#
# Why Parquet (BN-354): the store began as gzipped CSV, to need nothing
# beyond pandas, when 645k rows (512 names over five years) read in a second
# and a half. At 6,000 names over ten years, 11.4 million rows, reading and
# parsing the CSV took 15 s of a 17.7 s load; as Parquet the same rows read
# and build in 1.4 s, from 235 MB rather than 300.
#
# Reproducibility is what makes BN-114's determinism guarantee testable.
import gzip
import io
import json
import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError, StoreFormatError
from .base import MarketData, ReferenceData
from .corporate_actions import CorporateActions
from .features import FeatureData
from .fetcher import DEFAULT_FX_POLICY, DataFetcher
from .free_float import DEFAULT_FREE_FLOAT_BACKFILL_DAYS
from .kinds import LEGACY_RATE_COLUMN, SeriesError

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes. A store written by a newer Beacon is
# refused rather than misread; an older one is upgraded on load.
# Version 2 is Parquet (BN-354); version 1 was gzipped CSV.
STORE_SCHEMA_VERSION = 2

MANIFEST_NAME = "manifest.json"
MARKET_FILE = "market.parquet"
REFERENCE_FILE = "reference.parquet"
ACTIONS_FILE = "corporate_actions.parquet"
FEATURES_FILE = "features.parquet"

# Dataset name -> the file holding it. The names match
# `DataFetcher.DATASETS`, so coverage can measure a dataset without
# knowing this module's filenames.
FILE_FOR_DATASET = {
    "market": MARKET_FILE,
    "reference": REFERENCE_FILE,
    "corporate_actions": ACTIONS_FILE,
    "features": FEATURES_FILE,
}

# Each dataset's file in a version 1 store, read to convert it.
LEGACY_FILE_FOR_DATASET = {
    "market": "market.csv.gz",
    "reference": "reference.csv.gz",
    "corporate_actions": "corporate_actions.csv.gz",
    "features": "features.csv.gz",
}

# Matches DocumentStore's app name, so a machine has one Beacon directory
# rather than two that differ by a letter.
APP_NAME = "beacon"
STORE_DIRECTORY = "market-store"

# Environment variable consulted when no --data path is given.
DATA_PATH_ENV_VAR = "BEACON_DATA_PATH"

# Written into the manifest so `/data/coverage` can say where rows came from
# without guessing. Free-form: a downstream source this module has never heard
# of should be able to name itself rather than pick the closest lie.
SOURCE_SYNTHETIC = "synthetic"
SOURCE_LOCAL = "local"
SOURCE_YFINANCE = "yfinance"
SOURCE_IMPORTED = "imported"

# zstd at level 3, with float columns split into byte streams, which is
# what lets random-walk prices compress at all: on the 11.4 million rows of a
# 6,000-name store, 235 MB written in 4.7 s, against 266 MB without the split
# and 228 MB at level 9 in 15 s.
ZSTD_LEVEL = 3

# Rows to a row group. Fixed, so the same rows always make the same file, and
# how far a save's progress moves at a time.
CHUNK_ROWS = 1_000_000

# Called with the share of a save done, from 0 to 1, and what is being saved.
SaveProgress = Callable[[float, str], None]


@dataclass(frozen=True)
class StoreManifest:
    """What a store says about itself.

    Attributes:
        schema_version: On-disk shape, for migration on load.
        source: Where the rows came from: "synthetic", "yfinance", "local",
            "imported", or whatever wrote it.
        datasets: Which files are present, so a loader knows what to expect
            rather than probing for each one.
    """
    schema_version: int
    source: str
    datasets: tuple[str, ...]


def default_path() -> Path:
    """The store location a spawned server auto-loads.

    The platform app-data directory, beside the documents `DocumentStore`
    keeps. Requires `platformdirs`; passing an explicit path anywhere in this
    module needs nothing beyond pandas, because only the *default* location is
    a platform question.

    Returns:
        Path: The directory, which may not exist.
    """
    # Deliberately inside the function: `beacon.data` is on the core import
    # path, and hoisting this would make the whole data layer load a package
    # to answer a question only this function asks.
    import platformdirs  # noqa: PLC0415

    return Path(platformdirs.user_data_dir(APP_NAME)) / STORE_DIRECTORY



def size_on_disk(path: Path) -> int | None:
    """Total bytes a store occupies, or None if it is not readable.

    Every file in the directory, not just the ones this module writes: a store
    a future version adds a file to should report its real size rather than a
    figure that quietly excludes the new part.
    """
    if not path.is_dir():
        return None

    try:
        return sum(entry.stat().st_size
                   for entry in path.iterdir() if entry.is_file())
    except OSError as exc:
        logger.warning("Could not measure the store at %s: %s.", path, exc)

        return None


def dataset_size_on_disk(path: Path,
                         dataset: str) -> int | None:
    """Bytes one dataset's file occupies.

    Per dataset rather than the whole store, so a coverage pane showing a size
    against each row shows numbers that add up. Reporting the store total on
    every row would display the same figure once per dataset and make any sum
    of them wrong by that factor.

    Returns:
        int | None: The size, or None when the store does not hold that
        dataset ("there is no file" rather than "the file is empty"), when
        the dataset name is unknown, or when the file cannot be measured.

        Callers that sum these have to treat None as zero themselves, so a
        caller adding up sizes has to notice that a dataset is absent rather
        than have the absence quietly contribute a zero it never checked.
    """
    # Deliberate, and `test_an_unmeasurable_path_is_none` pins it.
    if dataset not in FILE_FOR_DATASET:
        return None

    target = _dataset_file(path, dataset)
    if target is None:
        return None

    try:
        return target.stat().st_size
    except OSError as exc:
        logger.warning("Could not measure %s: %s.", target, exc)

        return None


def _dataset_file(path: Path,
                  dataset: str) -> Path | None:
    """The file holding *dataset*: its Parquet file, else a version 1 store's
    CSV, else None."""
    for name in (FILE_FOR_DATASET[dataset], LEGACY_FILE_FOR_DATASET[dataset]):
        if (path / name).is_file():
            return path / name

    return None


def exists(path: Path) -> bool:
    """Whether a readable store lives at this path.

    A directory holding a manifest and market data. Market data is the floor:
    reference data and corporate actions are optional, but a store with no
    prices is not a data source, and reporting it as one would move the failure
    from startup to the first request.
    """
    return (path / MANIFEST_NAME).is_file() and _dataset_file(path, "market") is not None


def read_manifest(path: Path) -> StoreManifest:
    """Read a store's manifest.

    Raises:
        ConfigurationError: If it is missing, unreadable, or written by a
            newer py-beacon than this one.
    """
    manifest_path = path / MANIFEST_NAME

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "data_store",
            f"No manifest at {manifest_path}: this is not a Beacon data "
            f"store.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            "data_store",
            f"The manifest at {manifest_path} is not valid JSON: {exc}.") from exc

    version = int(payload.get("schema_version", 0))
    if version > STORE_SCHEMA_VERSION:
        raise ConfigurationError(
            "data_store",
            f"The store at {path} is version {version}, but this Beacon reads "
            f"up to version {STORE_SCHEMA_VERSION}. Upgrade Beacon or "
            f"regenerate the store.")

    return StoreManifest(schema_version=version,
                         source=str(payload.get("source", SOURCE_LOCAL)),
                         datasets=tuple(payload.get("datasets", ())))


def flatten_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Index levels back to columns, ready to write.

    The containers differ in whether they kept their key columns:
    `CorporateActions` sets its index with ``drop=False`` so the columns
    survive, while `MarketData` and `ReferenceData` consume theirs. Dropping
    the index when its names are already columns avoids writing each key twice,
    which `from_dataframe` would then read back as a duplicate-column frame.
    """
    duplicated = any(name in frame.columns for name in frame.index.names)

    return frame.reset_index(drop=duplicated)


def as_stored(frame: pd.DataFrame,
              dataset: str) -> pd.DataFrame:
    """*frame* with the types a store has always given back for it.

    A store was gzipped CSV before py-beacon 0.9.6, and a load gave back what
    reading that text gave: numbers as float64 or int64, a date column no container
    parses as text, a blank string as blank. Parquet keeps whatever it is
    given, so without this a store saved now would load other types from the
    same data, and the API would show them.
    """
    stored = frame.reset_index(drop=True)

    # Numbers exactly, as float64 or int64. The CSV store parsed them back
    # from text, and about one float64 in six came back a bit away from
    # itself; that rounding is not kept.
    for column in stored.columns:
        kind = stored[column].dtype.kind

        if kind == "f" and stored[column].dtype != "float64":
            stored[column] = stored[column].astype("float64")
        elif kind in "iu" and stored[column].dtype != "int64":
            stored[column] = stored[column].astype("int64")

    # Text and dates as the CSV store read them: a date no container parses
    # stays text, a blank stays blank. The market's identifiers and dates are
    # parsed by MarketData either way.
    texts = [column for column in stored.columns
             if stored[column].dtype.kind not in "fiub"
             and not (dataset == "market" and column in ("IDENTIFIER", "DATE"))]

    if texts and not stored.empty:
        text = stored[texts].to_csv(index=False, lineterminator="\n")
        stored[texts] = pd.read_csv(io.StringIO(text))

    return stored


def _write_frame(frame: pd.DataFrame,
                 path: Path,
                 progress: Callable[[float], None] | None = None,
                 delta_integers: bool = False) -> None:
    """Write one Parquet file reproducibly.

    Args:
        frame: What to write, its index left out.
        path: The file to write.
        progress: Called after each row group with the share of the rows
            written.
        delta_integers: Delta-pack whole-number columns, which halves a
            column of prices kept as whole cents.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pa.Table.from_pandas(frame, preserve_index=False)
    floats = [field.name for field in table.schema if pa.types.is_floating(field.type)]
    rows = table.num_rows
    integers = ([field.name for field in table.schema if pa.types.is_integer(field.type)]
                if delta_integers else [])
    # BN-403: a column delta-packed is kept out of the dictionary, which
    # Parquet would otherwise try first.
    encoding: dict[str, Any] = {"use_dictionary": [name for name in table.column_names
                                                   if name not in integers]} if integers else {}

    if integers:
        encoding["column_encoding"] = dict.fromkeys(integers, "DELTA_BINARY_PACKED")

    with pq.ParquetWriter(path, table.schema, compression="zstd",
                          compression_level=ZSTD_LEVEL,
                          use_byte_stream_split=floats or False, **encoding) as writer:
        # One pass even for an empty frame, which still writes its schema.
        for start in range(0, max(rows, 1), CHUNK_ROWS):
            writer.write_table(table.slice(start, CHUNK_ROWS), row_group_size=CHUNK_ROWS)

            if progress is not None:
                progress(min(start + CHUNK_ROWS, rows) / rows if rows else 1.0)


def outdated(path: Path) -> str | None:
    """Why the store at *path* is in a format this version no longer reads,
    or None. Reads only its market file's header.

    A store saved before py-beacon 0.7.0 keeps exchange rates in a ``RATE``
    column, which is refused since they moved to ``CLOSE``.
    """
    # BN-309, after BN-308 retired the RATE column. Only a version 1 store
    # can carry one: the column was gone before stores were Parquet.
    try:
        with gzip.open(path / LEGACY_FILE_FOR_DATASET["market"], "rt", encoding="utf-8",
                       newline="") as handle:
            header = handle.readline()
    except (OSError, EOFError, zlib.error, UnicodeDecodeError):
        return None

    columns = {column.strip().strip('"') for column in header.rstrip("\r\n").split(",")}

    if LEGACY_RATE_COLUMN not in columns:
        return None

    return ("It was saved before py-beacon 0.7.0, with exchange rates in a RATE "
            "column, which is no longer read. Regenerate or re-import it.")


def _read_frame(path: Path,
                columns: list[str] | None = None) -> pd.DataFrame:
    """Read one Parquet file back, all its columns or *columns*.

    Raises:
        ConfigurationError: If the file cannot be read, naming it (BN-233),
            so a damaged store is refused cleanly rather than stopping the
            server with a traceback.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    try:
        frame = pq.read_table(path, columns=columns).to_pandas()
    except (OSError, pa.ArrowException) as error:
        raise ConfigurationError(
            "data_store",
            f"{path.name} in {path.parent} cannot be read: {error}.") from error

    # A column of flags with gaps comes back holding None in the gaps, where
    # reading the CSV store gave NaN.
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].where(frame[column].notna(), np.nan)

    return frame


def _read_legacy_frame(path: Path) -> pd.DataFrame:
    """Read one gzipped CSV of a version 1 store.

    Raises:
        ConfigurationError: If the file cannot be read, naming it (BN-233).
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return pd.read_csv(handle)
    except (OSError, EOFError, zlib.error, UnicodeDecodeError,
            pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ConfigurationError(
            "data_store",
            f"{path.name} in {path.parent} cannot be read: {error}.") from error


def _build(path: Path,
           frame: pd.DataFrame,
           build: Callable[[pd.DataFrame], T]) -> T:
    """Turn a file's rows into data, naming the file if they don't fit.

    Raises:
        ConfigurationError: If the rows lack a required column or hold values
            of the wrong kind.
    """
    try:
        return build(frame)
    except (KeyError, ValueError, TypeError) as error:
        raise ConfigurationError(
            "data_store",
            f"{path.name} in {path.parent} does not hold the rows a data "
            f"store needs: {error}.") from error


def _read_dataset(path: Path,
                  dataset: str,
                  build: Callable[[pd.DataFrame], T]) -> T | None:
    """One dataset built from its file, Parquet or a version 1 store's CSV;
    None when the store does not hold it."""
    target = _dataset_file(path, dataset)

    if target is None:
        return None

    read = _read_frame if target.name == FILE_FOR_DATASET[dataset] else _read_legacy_frame

    return _build(target, read(target), build)


def _convert(path: Path,
             datasets: dict[str, MarketData | ReferenceData | CorporateActions
                            | FeatureData | None]) -> None:
    """Rewrite a version 1 store as Parquet, from the data just read from it.

    Each dataset is written as :func:`save` writes it, then the manifest says
    version 2, and only then do the CSVs go, so a failure part-way leaves a
    store that still reads. A store that cannot be written to (a read-only
    folder, say) stays as it is, with a warning, and is read from its CSVs
    each time.
    """
    try:
        for dataset, data in datasets.items():
            if data is not None:
                _write_frame(as_stored(flatten_index(data.data), dataset),
                             path / FILE_FOR_DATASET[dataset])

        manifest_path = path / MANIFEST_NAME
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = STORE_SCHEMA_VERSION
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    except OSError as error:
        logger.warning("Could not convert the data store at %s to Parquet, so it is "
                       "read from its CSV files: %s", path, error)
        return

    for name in LEGACY_FILE_FOR_DATASET.values():
        (path / name).unlink(missing_ok=True)

    logger.info("Converted the data store at %s to Parquet.", path)


def save(fetcher: DataFetcher,
         path: Path,
         source: str = SOURCE_LOCAL,
         progress: SaveProgress | None = None) -> Path:
    """Write a fetcher's data to a store directory.

    Args:
        fetcher: The data source to persist. Reference data, corporate
            actions and features are written only when present, so a
            market-only fetcher round-trips as a market-only store rather
            than as one carrying empty files.
        path: Directory to write into; created if absent.
        source: Recorded in the manifest and reported by `/data/coverage`.
        progress: Called as the files are written, with the share of the
            save done (by rows) and what is being saved.

    Returns:
        Path: The directory written.
    """
    path.mkdir(parents=True, exist_ok=True)

    actions = fetcher.corporate_actions
    features = fetcher.features
    parts = [("market", "market data", fetcher.market.data, MARKET_FILE),
             ("reference", "reference data",
              fetcher.reference.data if fetcher.reference is not None else None,
              REFERENCE_FILE),
             ("corporate_actions", "corporate actions",
              None if actions.is_empty else actions.data, ACTIONS_FILE),
             ("features", "features", None if features.is_empty else features.data,
              FEATURES_FILE)]
    written = [(name, label, frame, file) for name, label, frame, file in parts
               if frame is not None]
    total = sum(len(frame) for _, _, frame, _ in written) or 1
    done = 0

    for name, label, frame, file in written:
        _write_frame(as_stored(flatten_index(frame), name), path / file,
                     _share(progress, done, len(frame), total, f"Saving {label}"))
        done += len(frame)

    datasets = [name for name, _, _, _ in written]
    manifest = {"schema_version": STORE_SCHEMA_VERSION,
                "source": source,
                "datasets": datasets}
    (path / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger.info("Wrote a %s data store to %s (%s).",
                source, path, ", ".join(datasets))

    return path


def _share(progress: SaveProgress | None,
           done: int,
           rows: int,
           total: int,
           message: str) -> Callable[[float], None] | None:
    """One file's progress as a share of the whole save, or None."""
    if progress is None:
        return None

    report = progress

    return lambda fraction: report((done + fraction * rows) / total, message)


def save_features(features: FeatureData,
                  path: Path) -> None:
    """Write a store's feature table, leaving its other datasets as they are.

    Args:
        features: The whole feature table, not just new rows: the file is
            replaced.
        path: The store directory. It must already hold a manifest.

    Raises:
        ConfigurationError: If *path* is not a Beacon data store.
    """
    manifest_path = path / MANIFEST_NAME
    read_manifest(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    _write_frame(as_stored(flatten_index(features.data), "features"), path / FEATURES_FILE)

    datasets = list(payload.get("datasets", []))
    if "features" not in datasets:
        payload["datasets"] = [*datasets, "features"]
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")


def load(path: Path,
         fx_policy: str = DEFAULT_FX_POLICY,
         max_price_staleness_days: int | None = None,
         free_float_backfill_days: int = DEFAULT_FREE_FLOAT_BACKFILL_DAYS
         ) -> DataFetcher:
    """Read a store directory into a fetcher.

    Args:
        path: Directory written by :func:`save`.
        fx_policy: How the fetcher reads a rate on a day the pair printed
            none: see :data:`~beacon.data.fetcher.FX_CARRY_FORWARD` and
            :data:`~beacon.data.fetcher.FX_EXACT_DAY`. A modelling assumption
            rather than a store property, so it is chosen when the store is
            opened rather than written into it: the same rows converted under
            the other assumption are a legitimate second answer, and stamping
            one into the data would make it the only one.
        max_price_staleness_days: How many calendar days a name may go
            untraded before it is dropped; None keeps everything.
        free_float_backfill_days: How many calendar days back a blank free
            float may take the last reported value from. Chosen at open for
            the same reason as the two above.

    Returns:
        DataFetcher: Serving whatever the store holds.

    Raises:
        StoreFormatError: If the store was saved in a format this version no
            longer reads (see `outdated`).
        ConfigurationError: If the path is not a readable store, or its rows
            contradict each other (see `beacon.data.kinds`).
        ValueError: If *fx_policy* is not a known policy, or either day
            count is out of range.
    """
    # The three settings are BN-207 (fx_policy), BN-211 (staleness) and
    # BN-219 (free-float backfill).
    if not exists(path):
        raise ConfigurationError(
            "data_store",
            f"No data store at {path}: expected {MANIFEST_NAME} and "
            f"{MARKET_FILE}.")

    manifest = read_manifest(path)
    reason = outdated(path)

    if reason is not None:
        raise StoreFormatError("data_store", f"The data store at {path}: {reason}")

    market = _read_dataset(path, "market", MarketData.from_dataframe)
    reference = _read_dataset(path, "reference", ReferenceData.from_dataframe)
    actions = _read_dataset(path, "corporate_actions", CorporateActions.from_dataframe)
    features = _read_dataset(path, "features", FeatureData.from_dataframe)

    if manifest.schema_version < STORE_SCHEMA_VERSION:
        _convert(path, {"market": market, "reference": reference,
                        "corporate_actions": actions, "features": features})

    if market is None:
        raise ConfigurationError("data_store", f"No market data in the data store at {path}.")

    logger.info("Loaded %d identifier(s) from the %s data store at %s.",
                len(market.identifiers), manifest.source, path)

    # The fetcher checks the store's rows against each other: what reference
    # data says a series is, against its prices (BN-308). A store that fails
    # is reported like any other unreadable store, so a server starting on it
    # comes up without data rather than not at all.
    try:
        fetcher = DataFetcher(market, reference, actions, features,
                              fx_policy=fx_policy,
                              max_price_staleness_days=max_price_staleness_days,
                              free_float_backfill_days=free_float_backfill_days)
    except SeriesError as error:
        raise ConfigurationError("data_store",
                                 f"The data store at {path} cannot be read: {error}") from error

    # Stamped here because this is the only place that knows: the manifest says
    # who wrote the rows, and the path is what `/data/coverage` measures on
    # disk. A fetcher built in-process keeps None for both, which is honest.
    fetcher.record_origin(manifest.source, path)

    return fetcher
