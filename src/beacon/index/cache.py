# src/beacon/index/cache.py
"""
A persistent, content-addressed cache of calculated :class:`IndexResult`s.

A result is stored under a fingerprint of the four things the calculation is a
pure function of — definition, the store behind the fetcher, window, library
version — so there is deliberately NO invalidation logic: changed inputs never
match, and stale entries age out by size-capped pruning.

The safety rule: **cache only what can be keyed completely; anything else
calculates fresh, every time.** An incomplete key would mean silently stale
numbers, so anything the key cannot capture (an unregistered rule class, a
parameter that does not serialise, a fetcher with no on-disk store behind it)
makes :func:`fingerprint` return None, and no key means no cache — at the cost
of a recalculation the caller would have paid anyway.
:func:`explain_uncacheable` answers *why*.

Storage is one directory per fingerprint: panels in the data store's own
reproducible gzipped-CSV format, plus a ``manifest.json`` keeping the key
parts in the clear so "why didn't this hit?" is answerable by looking. Entries
are staged and renamed into place; on read, anything missing or unparseable is
a miss and the corrupt entry is removed — the cache never raises into a
calculation. There is no cache schema version: the library version is part of
every key, so a format change rides on the version bump that ships it.
"""
import hashlib
import json
import logging
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from .._optional import require
from ..catalogue import SELECTION, WEIGHTING, classes, parameters_of
from ..data import store
from ..data.fetcher import DataFetcher
from .capping import CapReport
from .constructor import IndexDefinition
from .result import _DAILY_WEIGHT_DTYPES, IndexResult

logger = logging.getLogger(__name__)

# Lives beside the market-store directory under the same app name, so a
# machine has one Beacon directory rather than several that differ by a word.
CACHE_DIRECTORY = "index_cache"

MANIFEST_NAME = "manifest.json"
LEVELS_FILE = "levels.csv.gz"
DIVISOR_FILE = "divisor.csv.gz"
DAILY_WEIGHTS_FILE = "daily_weights.csv.gz"
SNAPSHOTS_FILE = "snapshots.json"

# The panels an entry holds, for the manifest's size table and for pruning.
DATA_FILES = (LEVELS_FILE, DIVISOR_FILE, DAILY_WEIGHTS_FILE, SNAPSHOTS_FILE)

# Total size the cache may occupy before the least-recently-used entries are
# evicted. Pruned on every put, in the spirit of the job registry's retention.
MAX_CACHE_BYTES = 512 * 1024 * 1024

# Keys are sha256 hexdigests and nothing else may name an entry directory:
# get() deletes what it cannot parse, so an arbitrary string reaching the
# filesystem as a path could delete something that is not ours.
_KEY_FORMAT = re.compile(r"[0-9a-f]{64}")

_DATE_COLUMN = "DATE"
_LEVEL_COLUMN = "LEVEL"
_DIVISOR_COLUMN = "DIVISOR"


def default_root() -> Path:
    """The cache location used when no explicit root is given.

    Requires `platformdirs`, imported inside the function for the same reason
    `store.default_path` does: only the *default* location is a platform
    question, and the core import path must not need an optional package.
    """
    require("platformdirs", "The default index-cache location")

    import platformdirs  # noqa: PLC0415

    return Path(platformdirs.user_data_dir(store.APP_NAME)) / CACHE_DIRECTORY


# -- fingerprinting ----------------------------------------------------------


def fingerprint(definition: IndexDefinition,
                fetcher: DataFetcher,
                start_date: str | None,
                end_date: str) -> str | None:
    """The cache key for one calculation, or None when it cannot be keyed.

    A sha256 hexdigest over the canonical JSON dump of the key parts. The
    window arguments are the calculator's own (*start_date* None means the
    base date) and are normalised, so two spellings of the same window share
    a key. None means uncacheable: the reason is logged at DEBUG and available
    from :func:`explain_uncacheable`.
    """
    parts = key_parts(definition, fetcher, start_date, end_date)
    if parts is None:
        return None

    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def key_parts(definition: IndexDefinition,
              fetcher: DataFetcher,
              start_date: str | None,
              end_date: str) -> dict[str, Any] | None:
    """The four key parts in the clear — what :func:`fingerprint` hashes and
    what an entry's manifest records — or None when uncacheable."""
    parts, _ = _key_parts(definition, fetcher, start_date, end_date)

    return parts


def explain_uncacheable(definition: IndexDefinition,
                        fetcher: DataFetcher,
                        start_date: str | None,
                        end_date: str) -> str | None:
    """Why this calculation cannot be cached, or None when it can."""
    _, reason = _key_parts(definition, fetcher, start_date, end_date)

    return reason


def _key_parts(definition: IndexDefinition,
               fetcher: DataFetcher,
               start_date: str | None,
               end_date: str) -> tuple[dict[str, Any] | None, str | None]:
    """Build the key payload, or the reason there cannot be one.

    The reason is DEBUG-logged on the way out, so a silent None is traceable.
    """
    definition_part, reason = _definition_payload(definition)

    data_part = window = None
    if definition_part is not None:
        data_part, reason = _data_identity(fetcher)
    if data_part is not None:
        window, reason = _window_payload(definition, start_date, end_date)

    if window is None:
        logger.debug("Index result is uncacheable: %s", reason)
        return None, reason

    return {"definition": definition_part,
            "data": data_part,
            "window": window,
            "beacon_version": __version__}, None


def _definition_payload(definition: IndexDefinition) -> tuple[dict[str, Any] | None, str | None]:
    """Every constructor field of the definition, rules and scheme included —
    or None with the reason, when a rule or the scheme cannot be keyed."""
    rules = []
    for rule in definition.eligibility_rules:
        payload, reason = _configured_payload(rule, SELECTION, "eligibility rule")
        if payload is None:
            return None, reason

        rules.append(payload)

    scheme, reason = _configured_payload(definition.weighting_scheme,
                                         WEIGHTING,
                                         "weighting scheme")
    if scheme is None:
        return None, reason

    universe = definition.universe_identifiers

    return {"index_id": definition.index_id,
            "index_name": definition.index_name,
            "base_date": definition.base_date.isoformat(),
            "base_value": definition.base_value,
            "currency": definition.currency,
            "rebalancing_frequency": definition.rebalancing_frequency,
            "description": definition.description,
            "universe_identifiers": list(universe) if universe is not None else None,
            "max_constituent_weight": definition.max_constituent_weight,
            "rebalance_day_rule": definition.rebalance_day_rule,
            "calendar": definition.calendar,
            "return_type": definition.return_type,
            "withholding_tax_rate": definition.withholding_tax_rate,
            "effective_lag_sessions": definition.effective_lag_sessions,
            "eligibility_rules": rules,
            "weighting_scheme": scheme}, None


def _configured_payload(instance: Any,
                        kind: str,
                        role: str) -> tuple[dict[str, Any] | None, str | None]:
    """One rule or scheme as its registered name plus parameter values.

    Parameter names come from the catalogue's constructor introspection, each
    value read off the instance attribute of the same name — the convention
    every registered type follows. Whatever breaks the convention (an
    unregistered class, a missing attribute, a value JSON cannot carry) is
    exactly the incomplete key this module refuses to build.
    """
    cls = type(instance)

    if classes(kind).get(cls.__name__) is not cls:
        return None, (f"the {role} {cls.__name__} is not registered in the "
                      f"catalogue, so its configuration cannot be keyed")

    values: dict[str, Any] = {}
    for parameter in parameters_of(cls):
        if not hasattr(instance, parameter.name):
            return None, (f"the {role} {cls.__name__} keeps no attribute for "
                          f"its constructor parameter '{parameter.name}'")

        value = getattr(instance, parameter.name)
        if not _is_jsonable(value):
            return None, (f"the {role} {cls.__name__} parameter "
                          f"'{parameter.name}' does not serialise to JSON")

        values[parameter.name] = value

    return {"type": cls.__name__, "params": values}, None


def _data_identity(fetcher: DataFetcher) -> tuple[dict[str, Any] | None, str | None]:
    """The store behind the fetcher: its path plus the manifest's stamp.

    The stamp is the manifest's content hash *and* its mtime: `store.save`
    rewrites the manifest on every write, so regenerating a store — even to
    byte-identical content — moves the stamp and misses. That is the
    conservative direction; a rewrite never produces a stale hit, at worst a
    fresh calculation. An in-memory fetcher has no store path and no stable
    identity, so it is uncacheable — rightly, tests and mocks should not cache.
    """
    path = fetcher.store_path
    if path is None:
        return None, ("the fetcher has no store path: in-memory data has no "
                      "stable identity to key on")

    manifest = path / store.MANIFEST_NAME
    try:
        content = manifest.read_bytes()
        modified = manifest.stat().st_mtime_ns
    except OSError as exc:
        return None, f"the store manifest at {manifest} is unreadable: {exc}"

    return {"store_path": str(path.resolve()),
            "manifest_sha256": hashlib.sha256(content).hexdigest(),
            "manifest_mtime_ns": modified}, None


def _window_payload(definition: IndexDefinition,
                    start_date: str | None,
                    end_date: str) -> tuple[dict[str, str] | None, str | None]:
    """The calculation window, normalised the way the calculator resolves it."""
    try:
        start = (pd.Timestamp(start_date) if start_date is not None
                 else definition.base_date)
        end = pd.Timestamp(end_date)
    except (TypeError, ValueError) as exc:
        return None, f"the window dates do not parse: {exc}"

    if pd.isna(start) or pd.isna(end):
        return None, "the window dates do not parse: NaT"

    return {"start": start.isoformat(), "end": end.isoformat()}, None


def _is_jsonable(value: Any) -> bool:
    """Whether the canonical dump can carry this value."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False

    return True


# -- entry serialisation -----------------------------------------------------


def _now() -> str:
    """A sortable UTC stamp for created_at / last_used."""
    return datetime.now(UTC).isoformat()


def _write_json(path: Path,
                payload: dict[str, Any]) -> None:
    """One JSON file, written the way the data store writes its manifest."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_series(series: pd.Series,
                  column: str,
                  path: Path) -> None:
    """A date-indexed series as a two-column reproducible gzipped CSV."""
    frame = pd.DataFrame({_DATE_COLUMN: series.index,
                          column: series.to_numpy(dtype="float64")})

    store._write_frame(frame, path)


def _read_series(path: Path,
                 column: str) -> pd.Series:
    frame = store._read_frame(path)
    index = pd.DatetimeIndex(pd.to_datetime(frame[_DATE_COLUMN])).rename(None)

    return pd.Series(frame[column].to_numpy(dtype="float64"), index=index)


def _read_panel(path: Path) -> pd.DataFrame:
    """The daily weights panel, back in its BN-153 storage dtypes."""
    frame = store._read_frame(path)
    frame[_DATE_COLUMN] = pd.to_datetime(frame[_DATE_COLUMN])

    return frame.astype(_DAILY_WEIGHT_DTYPES)


def _snapshots_payload(result: IndexResult) -> dict[str, Any]:
    """Everything that is a dict keyed by Timestamp, dates as ISO strings."""
    return {
        "index_id": result.index_id,
        "constituent_snapshots": {date.isoformat(): list(ids)
                                  for date, ids in result.constituent_snapshots.items()},
        "weight_snapshots": {date.isoformat(): dict(weights)
                             for date, weights in result.weight_snapshots.items()},
        "announcement_dates": {effective.isoformat(): announced.isoformat()
                               for effective, announced in result.announcement_dates.items()},
        "cap_reports": {date.isoformat(): {"cap": report.cap,
                                           "capped": dict(report.capped),
                                           "redistributed": report.redistributed,
                                           "passes": report.passes,
                                           "uncapped_weights": dict(report.uncapped_weights)}
                        for date, report in result.cap_reports.items()},
    }


def _read_entry(entry: Path) -> IndexResult:
    """Rebuild an IndexResult from one entry directory. Raises on anything
    unexpected — a missing file, truncated gzip, bad JSON — and leaves the
    caller to translate that into a miss."""
    # Parsed for validation only: an entry whose manifest is gone or garbled
    # is corrupt, not merely untidy.
    _read_json(entry / MANIFEST_NAME)

    snapshots = _read_json(entry / SNAPSHOTS_FILE)

    return IndexResult(
        index_id=str(snapshots["index_id"]),
        index_levels=_read_series(entry / LEVELS_FILE, _LEVEL_COLUMN),
        divisor_history=_read_series(entry / DIVISOR_FILE, _DIVISOR_COLUMN),
        constituent_snapshots={pd.Timestamp(date): [str(asset) for asset in ids]
                               for date, ids in snapshots["constituent_snapshots"].items()},
        weight_snapshots={pd.Timestamp(date): {str(asset): float(weight)
                                               for asset, weight in weights.items()}
                          for date, weights in snapshots["weight_snapshots"].items()},
        cap_reports={pd.Timestamp(date): CapReport(**report)
                     for date, report in snapshots["cap_reports"].items()},
        announcement_dates={pd.Timestamp(effective): pd.Timestamp(announced)
                            for effective, announced in snapshots["announcement_dates"].items()},
        daily_weights=_read_panel(entry / DAILY_WEIGHTS_FILE))


# -- the cache ---------------------------------------------------------------


class IndexResultCache:
    """Filesystem cache of IndexResults, keyed by :func:`fingerprint`.

    Reading never raises: a corrupt or half-present entry is a miss, removed
    on sight. Writing never raises either — a cache write is a convenience,
    and failing one must not fail the calculation that produced the result.
    """

    def __init__(self,
                 root: Path | None = None):
        """
        Args:
            root: Directory the cache lives in, created on first write. None
                uses :func:`default_root`, which needs `platformdirs`.
        """
        self._root = Path(root) if root is not None else default_root()

    @property
    def root(self) -> Path:
        """Where this cache keeps its entries."""
        return self._root

    def entry_path(self,
                   key: str) -> Path:
        """The directory one key's entry occupies (which may not exist)."""
        return self._root / key

    def get(self,
            key: str) -> IndexResult | None:
        """Read a cached result, or None on a miss.

        A hit refreshes the ``last_used`` stamp pruning orders evictions by.
        The returned result has no DataFetcher bound; a consumer that needs
        asset views re-binds via :meth:`IndexResult.with_data`.
        """
        if not _KEY_FORMAT.fullmatch(key):
            logger.debug("Not a cache key: %r.", key)
            return None

        entry = self.entry_path(key)
        if not entry.is_dir():
            logger.debug("Index cache miss for %s.", key)
            return None

        try:
            result = _read_entry(entry)
        except Exception as exc:
            logger.warning("Dropping the corrupt index-cache entry at %s: %s",
                           entry, exc)
            shutil.rmtree(entry, ignore_errors=True)

            return None

        self._touch(entry)
        logger.debug("Index cache hit for %s.", key)

        return result

    def put(self,
            key: str,
            result: IndexResult,
            parts: dict[str, Any] | None = None) -> None:
        """Store a result under its fingerprint, then prune to the size cap.

        The caller owns the key/result pairing — this module cannot re-derive
        the inputs from the result. *parts* is the :func:`key_parts` payload,
        recorded in the entry's manifest in the clear so a stored entry can
        say what it was keyed on; optional, the fingerprint alone serves hits.
        """
        if not _KEY_FORMAT.fullmatch(key):
            logger.debug("Refusing to cache under invalid key %r.", key)
            return

        entry = self.entry_path(key)
        if entry.exists():
            logger.debug("Index cache already holds %s.", key)
            return

        staging = self._root / f".staging-{uuid.uuid4().hex}"
        try:
            staging.mkdir(parents=True)
            total = self._write_entry(staging, key, result, parts)
            staging.rename(entry)
        except Exception as exc:
            logger.warning("Could not cache index result '%s' under %s: %s",
                           result.index_id, key, exc)

            return
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        logger.info("Cached index result '%s' under %s (%d bytes).",
                    result.index_id, key, total)
        self._prune()

    def clear(self) -> int:
        """Remove every entry. Returns how many were removed."""
        removed = 0
        for _, _, entry in self._entries():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1

        return removed

    def size_on_disk(self) -> int:
        """Total bytes the cache's entries occupy."""
        return sum(size for _, size, _ in self._entries())

    # -- internals -----------------------------------------------------------

    def _write_entry(self,
                     staging: Path,
                     key: str,
                     result: IndexResult,
                     parts: dict[str, Any] | None) -> int:
        """Write one entry's files into the staging directory."""
        _write_series(result.index_levels, _LEVEL_COLUMN, staging / LEVELS_FILE)
        _write_series(result.divisor_history, _DIVISOR_COLUMN, staging / DIVISOR_FILE)
        store._write_frame(result.daily_weights, staging / DAILY_WEIGHTS_FILE)
        _write_json(staging / SNAPSHOTS_FILE, _snapshots_payload(result))

        sizes = {name: (staging / name).stat().st_size for name in DATA_FILES}

        stamp = _now()
        _write_json(staging / MANIFEST_NAME, {"key": key,
                                              "key_parts": parts,
                                              "created_at": stamp,
                                              "last_used": stamp,
                                              "sizes": sizes})

        return sum(sizes.values())

    def _touch(self,
               entry: Path) -> None:
        """Refresh last_used on a hit. Best effort — a failure costs only
        eviction order, never the hit itself."""
        manifest_path = entry / MANIFEST_NAME
        try:
            manifest = _read_json(manifest_path)
            manifest["last_used"] = _now()
            _write_json(manifest_path, manifest)
        except (OSError, ValueError) as exc:
            logger.warning("Could not update last_used for %s: %s", entry, exc)

    def _entries(self) -> list[tuple[str, int, Path]]:
        """Every entry as (last_used, size, path). Staging dirs excluded."""
        if not self._root.is_dir():
            return []

        found = []
        for child in self._root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue

            found.append((self._last_used(child), self._size_of(child), child))

        return found

    @staticmethod
    def _last_used(entry: Path) -> str:
        """The manifest's last_used stamp; unreadable sorts oldest, so a
        broken entry is the first thing pruning reclaims."""
        try:
            return str(_read_json(entry / MANIFEST_NAME).get("last_used", ""))
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _size_of(entry: Path) -> int:
        try:
            return sum(item.stat().st_size
                       for item in entry.iterdir() if item.is_file())
        except OSError:
            return 0

    def _prune(self) -> None:
        """Evict least-recently-used entries until the size cap holds."""
        entries = self._entries()
        total = sum(size for _, size, _ in entries)
        if total <= MAX_CACHE_BYTES:
            return

        evicted = 0
        for _, size, entry in sorted(entries):
            shutil.rmtree(entry, ignore_errors=True)
            total -= size
            evicted += 1

            if total <= MAX_CACHE_BYTES:
                break

        logger.info("Pruned %d index-cache entry(s) beyond the %d-byte cap.",
                    evicted, MAX_CACHE_BYTES)
