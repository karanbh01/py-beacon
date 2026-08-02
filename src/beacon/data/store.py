# src/beacon/data/store.py
"""
A persisted market-data store on disk.

Until now data reached the server only by being loaded into `MarketData` by a
caller that built the process. `python -m beacon.server` builds its own process,
so it had no such caller and always started data-less: every data endpoint
answered `CONFIGURATION_ERROR`, and sync could not bootstrap because there was
nothing to sync *into*. This is the format that gives a spawned server
something to find.

## Why CSV and gzip

A columnar format would load faster and pack smaller. It would also put a
binary dependency (pyarrow, ~40MB) in front of the one thing that has to work
before anything else does. The store is read once at startup, and 645k rows —
512 names over five years — take about a second and a half through
`read_csv`, which is not the difference between a usable desktop application
and an unusable one.

What CSV buys is worth more here: the store opens in any text editor after a
`gunzip`, so "the server started with no data" is a question you can answer by
looking, and the format needs nothing beyond pandas. If load time ever becomes
the complaint, `schema_version` is how the format changes without stranding
anyone's store.

## Byte-for-byte reproducibility

Two runs of the same generator must produce the same store, or BN-114's
determinism guarantee is untestable. Three things would otherwise break it,
and each is pinned below:

* **Line endings.** `to_csv` follows the platform by default, so the same
  frame written on Windows and Linux differs in every row. Pinned to ``\\n``.
* **The gzip header carries an mtime.** Two writes a second apart produce
  different bytes for identical content. Pinned to 0.
* **Manifest key order.** Written sorted.

Timestamps are deliberately *not* recorded in the manifest. The store's age is
the file's mtime, and `DataFetcher` stamps its own refresh time when it loads —
writing a creation time into the manifest would be a third answer to a question
that already has two, and would make every store byte-unique.
"""
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .._optional import require
from ..exceptions import ConfigurationError
from .base import MarketData, ReferenceData
from .corporate_actions import CorporateActions
from .fetcher import DataFetcher

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes. A store written by a newer Beacon is
# refused rather than misread; an older one is upgraded on load.
STORE_SCHEMA_VERSION = 1

MANIFEST_NAME = "manifest.json"
MARKET_FILE = "market.csv.gz"
REFERENCE_FILE = "reference.csv.gz"
ACTIONS_FILE = "corporate_actions.csv.gz"

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

# Passed to every to_csv call. Without the line terminator the same frame
# written on two platforms differs in every row.
_CSV_OPTIONS = {"index": False, "lineterminator": "\n"}

# mtime=0 or the gzip header stamps the write time into the bytes.
_GZIP_OPTIONS = {"method": "gzip", "mtime": 0}


@dataclass(frozen=True)
class StoreManifest:
    """What a store says about itself.

    Attributes:
        schema_version: On-disk shape, for migration on load.
        source: Where the rows came from — "synthetic", "yfinance", "local",
            or whatever wrote it.
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
    require("platformdirs", "The default data-store location")

    # Deliberately inside the function: `beacon.data` is on the core import
    # path, and hoisting this would make the whole data layer need an optional
    # package to answer a question only this function asks.
    import platformdirs  # noqa: PLC0415

    return Path(platformdirs.user_data_dir(APP_NAME)) / STORE_DIRECTORY


def exists(path: Path) -> bool:
    """Whether a readable store lives at this path.

    A directory holding a manifest and market data. Market data is the floor:
    reference data and corporate actions are optional, but a store with no
    prices is not a data source, and reporting it as one would move the failure
    from startup to the first request.
    """
    return (path / MANIFEST_NAME).is_file() and (path / MARKET_FILE).is_file()


def read_manifest(path: Path) -> StoreManifest:
    """Read a store's manifest.

    Raises:
        ConfigurationError: If it is missing, unreadable, or written by a
            newer Beacon than this one.
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


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    """Index levels back to columns, ready to write.

    The containers differ in whether they kept their key columns:
    `CorporateActions` sets its index with ``drop=False`` so the columns
    survive, while `MarketData` and `ReferenceData` consume theirs. Dropping
    the index when its names are already columns avoids writing each key twice,
    which `from_dataframe` would then read back as a duplicate-column frame.
    """
    duplicated = any(name in frame.columns for name in frame.index.names)

    return frame.reset_index(drop=duplicated)


def _write_frame(frame: pd.DataFrame,
                 path: Path) -> None:
    """Write one gzipped CSV reproducibly."""
    frame.to_csv(path, compression=_GZIP_OPTIONS, **_CSV_OPTIONS)


def _read_frame(path: Path) -> pd.DataFrame:
    """Read one gzipped CSV back."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle)


def save(fetcher: DataFetcher,
         path: Path,
         source: str = SOURCE_LOCAL) -> Path:
    """Write a fetcher's data to a store directory.

    Args:
        fetcher: The data source to persist. Reference data and corporate
            actions are written only when present, so a market-only fetcher
            round-trips as a market-only store rather than as one carrying two
            empty files.
        path: Directory to write into; created if absent.
        source: Recorded in the manifest and reported by `/data/coverage`.

    Returns:
        Path: The directory written.
    """
    path.mkdir(parents=True, exist_ok=True)

    datasets = ["market"]
    _write_frame(_flatten(fetcher.market.data), path / MARKET_FILE)

    if fetcher.reference is not None:
        datasets.append("reference")
        _write_frame(_flatten(fetcher.reference.data), path / REFERENCE_FILE)

    actions = fetcher.corporate_actions
    if not actions.is_empty:
        datasets.append("corporate_actions")
        _write_frame(_flatten(actions.data), path / ACTIONS_FILE)

    manifest = {"schema_version": STORE_SCHEMA_VERSION,
                "source": source,
                "datasets": datasets}
    (path / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger.info("Wrote a %s data store to %s (%s).",
                source, path, ", ".join(datasets))

    return path


def load(path: Path) -> DataFetcher:
    """Read a store directory into a fetcher.

    Args:
        path: Directory written by :func:`save`.

    Returns:
        DataFetcher: Serving whatever the store holds.

    Raises:
        ConfigurationError: If the path is not a readable store.
    """
    if not exists(path):
        raise ConfigurationError(
            "data_store",
            f"No data store at {path}: expected {MANIFEST_NAME} and "
            f"{MARKET_FILE}.")

    manifest = read_manifest(path)

    market = MarketData.from_dataframe(_read_frame(path / MARKET_FILE))

    reference = None
    if (path / REFERENCE_FILE).is_file():
        reference = ReferenceData.from_dataframe(_read_frame(path / REFERENCE_FILE))

    actions = None
    if (path / ACTIONS_FILE).is_file():
        actions = CorporateActions.from_dataframe(_read_frame(path / ACTIONS_FILE))

    logger.info("Loaded %d identifier(s) from the %s data store at %s.",
                len(market.identifiers), manifest.source, path)

    return DataFetcher(market, reference, actions)
