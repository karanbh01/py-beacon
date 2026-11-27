# src/beacon/server/analytics_cache.py
"""
The last stored run's analytics, kept between requests, so opening one name
after another in Drilldown reuses the run's loaded tables and the work done
over the whole run (each portfolio's weights, the names' returns, the style
box at each rebalance) rather than doing it again for each.

    cache = AnalyticsCache()
    with cache.using(folder, source, data_version) as analytics:
        analytics.instrument.profile("AAPL")

One run is kept at a time, the one last asked about; another run's request
builds its own in its place. The analytics are built again whenever the run's
files change (a new backtest, an analysis rerun) or the data does (a store
loaded or refreshed). Requests take turns on it, so two never work the same
figure out at once.
"""
# BN-481 (Karan, 2026-11-27): Drilldown built every name's profile from a
# freshly loaded run.
import os
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import Request

from ..analysis.analytics import Analytics
from ..data.fetcher import DataFetcher
from .active_data import active_data
from .instrument import stored_analytics


class AnalyticsCache:
    """The analytics of the run last asked about."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[Any, ...] | None = None
        self._analytics: Analytics | None = None

    @contextmanager
    def using(self,
              folder: Path,
              source: DataFetcher | None,
              data_version: str) -> Iterator[Analytics]:
        """The analytics of the run kept in *folder* on the data *source*
        (`stored_analytics`), as kept from an earlier request when neither
        has changed since, else built afresh and kept."""
        key = (str(folder), data_version, run_stamp(folder))

        with self._lock:
            if key != self._key or self._analytics is None:
                # The run before is let go first, so two are never held.
                self._key, self._analytics = None, None
                self._analytics = stored_analytics(folder, source)
                self._key = key

            yield self._analytics

    def clear(self) -> None:
        """Let go of the run kept."""
        with self._lock:
            self._key, self._analytics = None, None


def kept_analytics(request: Request,
                   folder: Path) -> AbstractContextManager[Analytics]:
    """The analytics of the run kept in *folder* on the data being served,
    through the app's cache (`AnalyticsCache.using`)."""
    cache: AnalyticsCache = request.app.state.analytics_cache
    data = active_data(request)

    return cache.using(folder, data.fetcher, data.data_version)


def run_stamp(folder: Path) -> frozenset[tuple[str, int, int]]:
    """Every file under *folder*, with when it was last written and its size,
    which changes whenever the run is written again or its analysis is."""
    found = set()

    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)

            # Gone since the folder was listed: the run is being written.
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue

            found.add((os.path.relpath(path, folder), stat.st_mtime_ns, stat.st_size))

    return frozenset(found)


__all__ = ["AnalyticsCache", "kept_analytics", "run_stamp"]
