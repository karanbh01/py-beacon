# src/beacon/data/ingest.py
"""
Fetching market and reference data from an external source.

The `data` extra (yfinance) was declared and unused: data arrived only by being
loaded into `MarketData` at startup, which is why
`POST /data/coverage/{dataset}/sync` returned 501. This is the path that makes
it real.

## The downloader is injected

Everything here takes a *downloader* — a callable from
``(identifier, start, end)`` to a date-indexed OHLCV frame. The yfinance-backed
one is built by :func:`yfinance_downloader`, and that is the only place the
optional dependency is touched.

The point is not abstraction for its own sake. It means this module imports and
runs with no network and no extra installed, so the reshaping, the partial-
failure handling and the progress reporting are all testable against a fake
that returns known numbers — none of which could be tested honestly against a
live market-data service.

## One identifier at a time, and failures do not spread

A sync over five hundred names must not collapse because one was delisted last
month. Each identifier is fetched separately, a failure is recorded against
that identifier alone, and the run continues. The result says what came back
and what did not, so a caller can tell a successful sync from a sync that
succeeded at nothing.

Fetching them one at a time also makes progress mean something: "142 of 500"
is a real statement, whereas a single bulk request can only ever report 0% and
then 100%.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .._optional import require
from ..exceptions import DataNotFoundError

logger = logging.getLogger(__name__)

# What a downloader is: identifier, start, end -> date-indexed OHLCV frame.
Downloader = Callable[[str, str | None, str | None], pd.DataFrame]

# Source column to the name MarketData uses. Anything not listed is dropped
# rather than carried through under a foreign name — a column called
# "Adj Close" sitting beside "CLOSE" is an invitation to read the wrong one.
COLUMN_MAP = {
    "Open": "OPEN",
    "High": "HIGH",
    "Low": "LOW",
    "Close": "CLOSE",
    "Volume": "VOLUME",
}

# Reference fields worth taking, mapped from yfinance's info dictionary.
REFERENCE_MAP = {
    "longName": "NAME",
    "currency": "CURRENCY",
    "exchange": "EXCHANGE",
    "sector": "SECTOR",
    "industry": "INDUSTRY",
    "country": "COUNTRY",
}

# The validity start stamped on ingested reference records. Reference data
# needs a DATE_FROM and a download carries no history of when a field changed,
# so it is recorded as valid from the start of the requested window.
DEFAULT_VALID_FROM = "1900-01-01"


@dataclass
class IngestResult:
    """What a sync managed to fetch.

    Attributes:
        market: Long-form market data, ready for ``MarketData.from_dataframe``.
            Empty when nothing was fetched.
        reference: Long-form reference data. Empty when none was requested or
            none came back.
        fetched: Identifiers that returned data.
        failed: Identifier to the reason it did not, so a caller can tell a
            successful sync from one that succeeded at nothing.
    """
    market: pd.DataFrame
    reference: pd.DataFrame = field(default_factory=pd.DataFrame)
    fetched: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        """Market-data rows fetched."""
        return len(self.market)

    @property
    def succeeded(self) -> bool:
        """Whether anything at all came back."""
        return bool(self.fetched)

    def summary(self) -> dict[str, object]:
        """A JSON-ready description, for a job result or a log line."""
        return {"fetched": len(self.fetched),
                "failed": len(self.failed),
                "rows": self.rows,
                "identifiers": list(self.fetched),
                "errors": dict(self.failed)}


def yfinance_downloader() -> Downloader:
    """Build a downloader backed by yfinance.

    Returns:
        Downloader: Callable fetching one identifier's history.

    Raises:
        MissingDependencyError: If yfinance is not installed. Raised here
            rather than at import, so this module stays usable with an injected
            downloader and no extra installed.
    """
    yfinance = require("yfinance", "Market-data ingestion")

    def download(identifier: str,
                 start: str | None,
                 end: str | None) -> pd.DataFrame:
        ticker = yfinance.Ticker(identifier)
        frame: pd.DataFrame = ticker.history(start=start, end=end, auto_adjust=False)

        return frame

    return download


def yfinance_reference_downloader() -> Callable[[str], dict[str, object]]:
    """Build a reference-data downloader backed by yfinance.

    Returns:
        Callable: Takes an identifier and returns its info mapping.

    Raises:
        MissingDependencyError: If yfinance is not installed.
    """
    yfinance = require("yfinance", "Reference-data ingestion")

    def download(identifier: str) -> dict[str, object]:
        info: dict[str, object] = yfinance.Ticker(identifier).info

        return info

    return download


def normalise_history(identifier: str,
                      frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape a downloaded OHLCV frame into the long form MarketData expects.

    Args:
        identifier: The instrument the frame belongs to.
        frame: Date-indexed frame with Open/High/Low/Close/Volume columns.

    Returns:
        pd.DataFrame: Rows of ``IDENTIFIER``, ``DATE`` and the mapped columns.
        Empty when the input is empty.

    Raises:
        DataNotFoundError: If there is no close price. Every other column is
            optional — a series with no volume is still a usable price series —
            but the calculator and the engine both read CLOSE, so a frame
            without one would be accepted here and fail much later.
    """
    if frame.empty:
        return pd.DataFrame(columns=["IDENTIFIER", "DATE", *COLUMN_MAP.values()])

    present = {source: target for source, target in COLUMN_MAP.items()
               if source in frame.columns}

    if "CLOSE" not in present.values():
        raise DataNotFoundError(f"a close price for '{identifier}'",
                                source="ingestion")

    tidy = frame[list(present)].rename(columns=present).copy()

    # Downloaded indexes are often timezone-aware; MarketData compares against
    # naive timestamps, and a mixed-awareness comparison raises rather than
    # quietly misaligning.
    index = pd.DatetimeIndex(tidy.index)
    if index.tz is not None:
        index = index.tz_localize(None)

    tidy.insert(0, "DATE", index.normalize())
    tidy.insert(0, "IDENTIFIER", identifier)

    return tidy.reset_index(drop=True)


def normalise_reference(identifier: str,
                        info: dict[str, object],
                        valid_from: str = DEFAULT_VALID_FROM) -> dict[str, object]:
    """Reshape a downloaded info mapping into a reference-data row.

    Args:
        identifier: The instrument.
        info: The source's field mapping.
        valid_from: DATE_FROM to stamp on the record.

    Returns:
        dict: A reference row. Fields the source did not supply are simply
        absent rather than filled with a placeholder.
    """
    row: dict[str, object] = {"IDENTIFIER": identifier,
                              "DATE_FROM": valid_from,
                              "DATE_TO": None}

    for source, target in REFERENCE_MAP.items():
        value = info.get(source)
        if value is not None:
            row[target] = value

    return row


def ingest_market_data(identifiers: list[str],
                       downloader: Downloader,
                       start: str | None = None,
                       end: str | None = None,
                       on_progress: Callable[[int, int, str], None] | None = None
                       ) -> IngestResult:
    """Fetch history for several identifiers.

    Args:
        identifiers: What to fetch.
        downloader: Where to fetch it from.
        start: Inclusive start date.
        end: Inclusive end date.
        on_progress: Called with ``(done, total, identifier)`` after each one,
            so a job can report real progress rather than 0% then 100%.

    Returns:
        IngestResult: The combined data and a per-identifier outcome.
    """
    frames: list[pd.DataFrame] = []
    fetched: list[str] = []
    failed: dict[str, str] = {}

    for position, identifier in enumerate(identifiers, start=1):
        _fetch_one(identifier, downloader, start, end, frames, fetched, failed)

        if on_progress is not None:
            on_progress(position, len(identifiers), identifier)

    market = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["IDENTIFIER", "DATE", *COLUMN_MAP.values()]))

    logger.info(
        f"Ingested {len(market)} row(s) for {len(fetched)} of "
        f"{len(identifiers)} identifier(s); {len(failed)} failed.")

    return IngestResult(market=market, fetched=fetched, failed=failed)


def _fetch_one(identifier: str,
               downloader: Downloader,
               start: str | None,
               end: str | None,
               frames: list[pd.DataFrame],
               fetched: list[str],
               failed: dict[str, str]) -> None:
    """Fetch one identifier, recording the outcome either way.

    Broad except on purpose: a downloader talks to a network service and can
    fail in as many ways as that service can, and the whole point of fetching
    one at a time is that none of them stop the run. The reason is kept against
    the identifier rather than swallowed.
    """
    try:
        frame = normalise_history(identifier, downloader(identifier, start, end))
    except Exception as exc:
        logger.warning(f"Could not fetch '{identifier}': {exc}")
        failed[identifier] = str(exc)
        return

    if frame.empty:
        failed[identifier] = "no data returned"
        return

    frames.append(frame)
    fetched.append(identifier)


def ingest_reference_data(identifiers: list[str],
                          downloader: Callable[[str], dict[str, object]],
                          valid_from: str = DEFAULT_VALID_FROM,
                          on_progress: Callable[[int, int, str], None] | None = None
                          ) -> IngestResult:
    """Fetch reference records for several identifiers.

    Args:
        identifiers: What to fetch.
        downloader: Where to fetch it from.
        valid_from: DATE_FROM to stamp on each record.
        on_progress: Called with ``(done, total, identifier)`` after each one.

    Returns:
        IngestResult: The reference rows and a per-identifier outcome.
    """
    rows: list[dict[str, object]] = []
    fetched: list[str] = []
    failed: dict[str, str] = {}

    for position, identifier in enumerate(identifiers, start=1):
        try:
            info = downloader(identifier)
        except Exception as exc:
            logger.warning(f"Could not fetch reference data for '{identifier}': {exc}")
            failed[identifier] = str(exc)
        else:
            rows.append(normalise_reference(identifier, info, valid_from))
            fetched.append(identifier)

        if on_progress is not None:
            on_progress(position, len(identifiers), identifier)

    reference = (pd.DataFrame(rows) if rows
                 else pd.DataFrame(columns=["IDENTIFIER", "DATE_FROM", "DATE_TO"]))

    logger.info(
        f"Ingested reference data for {len(fetched)} of {len(identifiers)} "
        f"identifier(s); {len(failed)} failed.")

    return IngestResult(market=pd.DataFrame(), reference=reference,
                        fetched=fetched, failed=failed)
