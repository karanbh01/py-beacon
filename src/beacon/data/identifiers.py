# src/beacon/data/identifiers.py
"""
Searching and enumerating the identifiers a data source actually holds.

Nothing in the API could answer "which identifiers do you have, and which of
them match what the user is typing". `/data/coverage` counts them,
`/data/reference` looks up ones the caller already knows, and a universe's
members are a membership list rather than coverage. So a client wanting
type-ahead had to build its own index from the union of every universe — which
silently omits any identifier no universe happens to contain, and is empty when
none are configured.

## Built once, searched many times

This is called on every keystroke. A scan that reads reference data per request
would be doing the expensive part — pulling names out of pandas — over and over
for an answer that only changes when the data does.

So the index is built once and cached against a fingerprint of the fetcher's
refresh timestamps. A sync moves those, the fingerprint changes, and the next
request rebuilds. Nothing else invalidates it, because nothing else can change
which identifiers exist.

Matching strings are **folded at build time**, not per request. That is the
single biggest thing keeping a 100k-row search inside its budget: lowercasing a
hundred thousand strings per keystroke costs more than the matching does.

## Ranking is the server's job

Once `limit` is applied the caller cannot re-rank what it was not sent, so the
order has to be right here. Best first:

1. exact identifier match
2. identifier prefix
3. name prefix
4. identifier substring
5. name substring

with an alphabetical tie-break inside each tier. Someone typing `CMP0` wants
`CMP000`, not the first company whose description happens to contain that
fragment.

## Why `total` forces a full scan

`total` is the number of matches *before* the limit, so the client can say
"showing 20 of 340". That cannot be known without examining every candidate, so
there is no early exit — the limit bounds what is returned, never what is
looked at. Worth knowing before trying to optimise the loop away.
"""
import logging
from dataclasses import dataclass, field

from .fetcher import (
    ACTIONS_DATASET,
    MARKET_DATASET,
    REFERENCE_DATASET,
    DataFetcher,
)

logger = logging.getLogger(__name__)

# Reference columns worth carrying into a suggestion row. They disambiguate two
# similar tickers in the right-hand meta; none is required, and an absent one is
# simply null rather than a reason to drop the row.
NAME_COLUMN = "NAME"
EXCHANGE_COLUMN = "EXCHANGE"
CURRENCY_COLUMN = "CURRENCY"

# Match tiers, best first. Numeric so a sort key can carry them directly.
EXACT_IDENTIFIER = 0
IDENTIFIER_PREFIX = 1
NAME_PREFIX = 2
IDENTIFIER_SUBSTRING = 3
NAME_SUBSTRING = 4
NO_MATCH = 5

DEFAULT_LIMIT = 20

# The ceiling a request cannot raise. Enumeration is a legitimate use — a client
# indexing the whole store once — but an unbounded limit lets one request ask
# the server to serialise everything it holds.
MAX_LIMIT = 1000


@dataclass(frozen=True)
class IdentifierEntry:
    """One identifier the data source holds.

    Attributes:
        identifier: The symbol itself.
        name: Display name, or None when reference data carries none. A row
            without a name is still a useful suggestion, so it is returned
            rather than dropped.
        datasets: Which datasets actually cover it. This is what lets a client
            offer a reference-only name in a reference view and mark it
            unavailable for prices, rather than suggesting something it cannot
            then serve.
        exchange: Where it trades, when known.
        currency: Its denomination, when known.
    """
    identifier: str
    name: str | None = None
    datasets: tuple[str, ...] = ()
    exchange: str | None = None
    currency: str | None = None

    # Lowercased once at build time. Folding per request is what makes a
    # 100k-row search miss its budget.
    folded_identifier: str = field(default="", repr=False, compare=False)
    folded_name: str = field(default="", repr=False, compare=False)

    def rank_against(self,
                     query: str) -> int:
        """Which match tier this entry falls into for a folded query.

        Returns:
            int: One of the tier constants, or NO_MATCH.
        """
        if self.folded_identifier == query:
            return EXACT_IDENTIFIER

        if self.folded_identifier.startswith(query):
            return IDENTIFIER_PREFIX

        if self.folded_name and self.folded_name.startswith(query):
            return NAME_PREFIX

        if query in self.folded_identifier:
            return IDENTIFIER_SUBSTRING

        if self.folded_name and query in self.folded_name:
            return NAME_SUBSTRING

        return NO_MATCH


@dataclass(frozen=True)
class SearchResult:
    """What a search found.

    Attributes:
        entries: The rows to return, ranked and windowed.
        total: Matches *before* the limit, so a client can say "showing 20 of
            340".
        truncated: Whether the limit hid anything. Derivable from `total` and
            the window, but explicit beats arithmetic at a call site.
    """
    entries: tuple[IdentifierEntry, ...]
    total: int
    truncated: bool


class IdentifierIndex:
    """An in-memory index of the identifiers a fetcher holds.

    Args:
        entries: Pre-built rows, ordered alphabetically by identifier.
        version: A fingerprint of the data this was built from, used both to
            decide when to rebuild and as an ETag.
    """

    def __init__(self,
                 entries: tuple[IdentifierEntry, ...],
                 version: str):
        self.entries = entries
        self.version = version

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def empty(cls) -> "IdentifierIndex":
        """An index over no data.

        A server with no data source still answers this endpoint — "nothing
        matches" and "this engine is misconfigured" are different statements,
        and an empty suggestion list must not look like a broken install.
        """
        return cls((), "empty")

    @classmethod
    def build(cls,
              fetcher: DataFetcher) -> "IdentifierIndex":
        """Index everything a fetcher holds.

        Args:
            fetcher: The data source.

        Returns:
            IdentifierIndex: Entries ordered by identifier, so an enumeration
            with no query is alphabetical and every tie-break below is stable
            without re-sorting.
        """
        coverage: dict[str, set[str]] = {}

        for identifier in fetcher.identifiers:
            coverage.setdefault(identifier, set()).add(MARKET_DATASET)

        for identifier in fetcher.reference_identifiers or []:
            coverage.setdefault(identifier, set()).add(REFERENCE_DATASET)

        for identifier in fetcher.corporate_actions.identifiers:
            coverage.setdefault(identifier, set()).add(ACTIONS_DATASET)

        details = cls._reference_details(fetcher)

        entries = []
        for identifier in sorted(coverage):
            name, exchange, currency = details.get(identifier, (None, None, None))

            entries.append(IdentifierEntry(
                identifier=identifier,
                name=name,
                datasets=tuple(sorted(coverage[identifier])),
                exchange=exchange,
                currency=currency,
                folded_identifier=identifier.casefold(),
                folded_name=(name or "").casefold()))

        index = cls(tuple(entries), fingerprint(fetcher))

        logger.info("Indexed %d identifier(s) for search (version %s).",
                    len(entries), index.version)

        return index

    @staticmethod
    def _reference_details(
            fetcher: DataFetcher) -> dict[str, tuple[str | None, str | None,
                                                     str | None]]:
        """Name, exchange and currency per identifier, in one pass.

        Read from the whole reference frame rather than by querying each
        identifier: a hundred thousand single-row lookups is the cost this
        index exists to pay once instead of per request.
        """
        reference = fetcher.reference
        if reference is None:
            return {}

        frame = reference.data
        columns = [column for column in (NAME_COLUMN, EXCHANGE_COLUMN,
                                         CURRENCY_COLUMN)
                   if column in frame.columns]
        if not columns:
            return {}

        details: dict[str, tuple[str | None, str | None, str | None]] = {}

        for identifier, row in frame[columns].iterrows():
            key = str(identifier)
            # setdefault, not assignment: an identifier with several validity
            # windows appears more than once, and the first record is the one
            # the single-name endpoint would return.
            details.setdefault(key, (_text(row.get(NAME_COLUMN)),
                                     _text(row.get(EXCHANGE_COLUMN)),
                                     _text(row.get(CURRENCY_COLUMN))))

        return details

    def search(self,
               query: str | None = None,
               limit: int = DEFAULT_LIMIT,
               offset: int = 0,
               datasets: tuple[str, ...] = ()) -> SearchResult:
        """Find identifiers matching a fragment, or enumerate them all.

        Args:
            query: Fragment to match against identifier and name. None or
                empty means no filter — the first `limit` in index order.
            limit: Maximum rows to return.
            offset: Rows to skip, for walking a full enumeration.
            datasets: Only return identifiers covered by *all* of these.

        Returns:
            SearchResult: Ranked rows, plus the total before the limit.
        """
        # No copy in the common case: filtering 100k rows to build a list the
        # loop below would walk anyway is pure overhead when nothing is filtered.
        if datasets:
            wanted = set(datasets)
            candidates: tuple[IdentifierEntry, ...] | list[IdentifierEntry] = [
                entry for entry in self.entries
                if wanted.issubset(entry.datasets)]
        else:
            candidates = self.entries

        folded = (query or "").strip().casefold()

        if not folded:
            # Already alphabetical from the build, so enumeration needs no sort.
            window = list(candidates[offset:offset + limit])

            return SearchResult(tuple(window), len(candidates),
                                len(candidates) > offset + len(window))

        # One bucket per tier. Because `candidates` is alphabetical, each bucket
        # comes out alphabetical for free — so the tie-break needs no sort, and
        # concatenating the buckets in tier order *is* the ranking.
        #
        # That matters more than it looks. A query like "company a" matches
        # almost every name, and sorting a hundred thousand matched rows to
        # return twenty of them was the difference between 97ms and 3ms.
        exact: list[IdentifierEntry] = []
        identifier_prefix: list[IdentifierEntry] = []
        name_prefix: list[IdentifierEntry] = []
        identifier_substring: list[IdentifierEntry] = []
        name_substring: list[IdentifierEntry] = []

        # Ranking is inlined rather than calling `rank_against` per row: a
        # method call per entry is real money at this size. The method stays as
        # the readable statement of the rule, and a test pins the two together.
        for entry in candidates:
            folded_identifier = entry.folded_identifier

            if folded_identifier == folded:
                exact.append(entry)
            elif folded_identifier.startswith(folded):
                identifier_prefix.append(entry)
            else:
                folded_name = entry.folded_name

                if folded_name and folded_name.startswith(folded):
                    name_prefix.append(entry)
                elif folded in folded_identifier:
                    identifier_substring.append(entry)
                elif folded_name and folded in folded_name:
                    name_substring.append(entry)

        tiers = (exact, identifier_prefix, name_prefix,
                 identifier_substring, name_substring)
        total = sum(len(tier) for tier in tiers)

        return SearchResult(_window(tiers, offset, limit), total,
                            total > offset + min(limit, max(total - offset, 0)))


def _window(tiers: tuple[list[IdentifierEntry], ...],
            offset: int,
            limit: int) -> tuple[IdentifierEntry, ...]:
    """The requested slice across the tiers, in rank order.

    Walks tier by tier taking only what the window needs, so a query matching
    a hundred thousand rows still materialises twenty.
    """
    window: list[IdentifierEntry] = []
    skipped = 0

    for tier in tiers:
        if len(window) >= limit:
            break

        if skipped + len(tier) <= offset:
            skipped += len(tier)
            continue

        start = max(offset - skipped, 0)
        window.extend(tier[start:start + (limit - len(window))])
        skipped += len(tier)

    return tuple(window)


def _text(value: object) -> str | None:
    """A reference cell as a string, or None when it is absent or blank."""
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return None

    return text


def fingerprint(fetcher: DataFetcher) -> str:
    """A short version tag for the data an index would be built from.

    Made from the per-dataset refresh timestamps alone. Every path that can
    change which identifiers exist — both merge methods — records a refresh, so
    the timestamps are a complete signal.

    Deliberately *not* derived from the identifier count: reading that means a
    `unique()` over the whole market panel, which on a hundred thousand names
    is exactly the work a per-request cache key must not do. It also has to be
    computable without building the index, or checking the cache would cost as
    much as missing it.

    Used both to decide when to rebuild and as the ETag a client revalidates
    against.
    """
    from hashlib import blake2s  # noqa: PLC0415

    stamps = []
    for dataset in (MARKET_DATASET, REFERENCE_DATASET, ACTIONS_DATASET):
        moment = fetcher.last_refreshed(dataset)
        stamps.append(moment.isoformat() if moment else "-")

    return blake2s("|".join(stamps).encode("utf-8"), digest_size=8).hexdigest()
