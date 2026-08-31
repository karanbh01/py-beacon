# src/beacon/sources.py
"""
The process-level data source: what answers when nothing was bound.

    import beacon
    beacon.use(fetcher)          # point the process somewhere, once
    p.asset("AAA").prices(...)   # every unbound portfolio reads it

Resolution order (decision 16): a source **bound to the object wins** — a
backtest result's views always read the data its run used — and this module
is the fallback for objects that have none:

1. a source set explicitly with :func:`use`;
2. otherwise the default store on disk — the one `python -m beacon.synthetic`
   writes and the server auto-loads — loaded lazily on first need and cached;
3. neither → :class:`DataSourceError` naming both fixes, because "no data"
   discovered deep inside a price lookup is useless without being told what
   to do about it.

## Why this is not called `data`

`beacon.expressions.data` is the expression root (a symbolic, unbound field
reference) and `beacon.data` is the data package. A third `data` naming a
*bound* source would collide with both — the package collision was already
hit once (BN-142) and is pinned by a test. `use` says what it does.

## Deliberately process-global, and only a fallback

An ambient source makes interactive work frictionless and makes results
depend on process state — both are true. The design keeps the second harm
away from anything that matters: objects that must be reproducible (backtest
results) carry their own binding, which always wins, so the ambient source
only ever answers for objects that never recorded one.
"""
import logging
import threading

from .data.fetcher import DataFetcher
from .exceptions import DataSourceError

logger = logging.getLogger(__name__)

class _ProcessSource:
    """The one holder for the process's ambient source.

    A tiny object rather than module globals: the two slots change together
    under one lock, and rebinding attributes on an instance does not need
    `global` statements the linter rightly dislikes.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.explicit: DataFetcher | None = None
        self.default: DataFetcher | None = None


_state = _ProcessSource()


def use(fetcher: DataFetcher | None) -> None:
    """Set the process's data source; `None` clears it.

    Args:
        fetcher: What unbound reads should resolve against, or None to fall
            back to the default store again.
    """
    with _state.lock:
        _state.explicit = fetcher

    logger.info("Process data source %s.",
                "set" if fetcher is not None else "cleared")


def resolve() -> DataFetcher:
    """The data source an unbound read should use.

    Returns:
        DataFetcher: The explicit source when one was set, else the default
        store, loaded lazily on first call and cached for the process.

    Raises:
        DataSourceError: When neither exists. The message names both fixes.
    """
    with _state.lock:
        if _state.explicit is not None:
            return _state.explicit

        if _state.default is not None:
            return _state.default

        return _load_default()


def _load_default() -> DataFetcher:
    """Load the default store once. Caller holds the lock."""
    # Imported here rather than at module scope so that importing `beacon`
    # (which imports this module for `use`) stays clean of the store's
    # machinery until somebody actually needs the fallback.
    from .data import store  # noqa: PLC0415

    try:
        path = store.default_path()
        loadable = store.exists(path)
    except Exception:
        loadable = False
        path = None

    if not loadable or path is None:
        raise DataSourceError(
            "No data source. Call beacon.use(fetcher) to point this process "
            "at one, or generate the default store with "
            "`python -m beacon.synthetic`.")

    logger.info("Loading the default store from %s.", path)
    _state.default = store.load(path)

    return _state.default


def _reset_for_tests() -> None:
    """Drop both sources. Test scaffolding, not API."""
    with _state.lock:
        _state.explicit = None
        _state.default = None
