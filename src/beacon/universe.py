# src/beacon/universe.py
"""
Building a universe by filtering, rather than by listing.

    from beacon import universe
    from beacon.expressions import data

    members = universe.where((data.reference.region == "North America")
                             & (data.market.market_cap > 1e9),
                             fetcher)

The same expressions that drive index rules (BN-142), so there is one way of
naming a datapoint and it works everywhere.

## Frozen and live are different objects

The distinction a user has to be told about rather than left to discover.

A universe saved as **a list of identifiers** is a fact about a moment. It
does not change when the data does, which is what you want for a published
index whose membership was fixed at a review — and it cannot be refreshed,
because nothing records how it was chosen.

A universe saved as **an expression** is a question. Re-evaluating it next
month gives a different answer, which is what you want for "every US name
above a billion" — and it can never tell you what it contained last month,
because it does not store that.

Neither is more correct. What would be wrong is a universe that looks like one
and behaves like the other, so `resolve` takes a date and says plainly what it
resolved at, and the server records `mode` on the document.

## Resolution is point-in-time

`where(expression, fetcher, date)` answers as of `date`, through the same
resolver an index rule uses. A universe built "as of last March" contains what
was knowable last March — including names that have since been delisted, and
excluding ones that had not yet listed.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data.fetcher import DataFetcher
from .expressions.core import Expression, from_dict
from .expressions.resolve import resolve

logger = logging.getLogger(__name__)

# A universe that stores its members, versus one that stores its question.
FROZEN = "frozen"
LIVE = "live"
MODES = (FROZEN, LIVE)


def where(expression: Expression,
          fetcher: DataFetcher,
          date: pd.Timestamp | str | None = None,
          identifiers: list[str] | None = None,
          on_missing: bool = False) -> list[str]:
    """The instruments satisfying an expression, as of a date.

    Args:
        expression: The filter.
        fetcher: The data to resolve against.
        date: When to stand. Defaults to the end of the loaded data — not to
            today, because a store loaded from a file has a last date and
            answering against a calendar the data does not reach would report
            every name as having no value.
        identifiers: Candidates. Defaults to everything the store covers.
        on_missing: Whether a name with no value for a field is included.

    Returns:
        list[str]: Matching identifiers, in the order the candidates were
        given, so two runs over the same data produce the same list.
    """
    as_of = _standing_date(fetcher, date)
    candidates = identifiers if identifiers is not None else _all_names(fetcher)

    members = [name for name in candidates
               if resolve(expression, name, as_of, fetcher,
                          on_missing=on_missing)]

    logger.info("Universe filter selected %d of %d name(s) as of %s.",
                len(members), len(candidates), as_of.strftime("%Y-%m-%d"))

    return members


@dataclass(frozen=True)
class FilteredUniverse:
    """A universe and the question that produced it.

    Carries both the expression and the membership it resolved to, so a caller
    can save either — and `mode` says which of the two the document means.
    """
    expression: Expression
    identifiers: list[str]
    as_of: pd.Timestamp
    mode: str = LIVE
    candidates: int = field(default=0)

    @property
    def is_live(self) -> bool:
        """Whether re-evaluating is meant to change the answer."""
        return self.mode == LIVE

    def as_document(self) -> dict[str, Any]:
        """The parts a stored universe keeps."""
        return {"filter": self.expression.to_dict(), "mode": self.mode,
                "identifiers": list(self.identifiers),
                "as_of": self.as_of.strftime("%Y-%m-%d")}


def build(expression: Expression,
          fetcher: DataFetcher,
          date: pd.Timestamp | str | None = None,
          mode: str = LIVE,
          on_missing: bool = False) -> FilteredUniverse:
    """Resolve an expression and keep the question alongside the answer.

    Args:
        expression: The filter.
        fetcher: The data.
        date: When to stand.
        mode: `LIVE` to re-evaluate on read, `FROZEN` to keep the membership.
        on_missing: Whether uncovered names are included.

    Returns:
        FilteredUniverse: Expression, members, and which of the two counts.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")

    as_of = _standing_date(fetcher, date)
    candidates = _all_names(fetcher)

    return FilteredUniverse(
        expression=expression,
        identifiers=where(expression, fetcher, as_of, candidates, on_missing),
        as_of=as_of,
        mode=mode,
        candidates=len(candidates))


def resolve_document(document: dict[str, Any],
                     fetcher: DataFetcher,
                     date: pd.Timestamp | str | None = None) -> list[str]:
    """The members of a stored universe.

    A frozen document answers with what it stored; a live one re-evaluates its
    filter. That branch is the whole distinction, and putting it here means
    one place decides rather than every caller remembering.
    """
    stored = document.get("filter")

    if document.get("mode", FROZEN) != LIVE or not isinstance(stored, dict):
        return [str(name) for name in document.get("identifiers") or []]

    return where(from_dict(stored), fetcher, date)


def _all_names(fetcher: DataFetcher) -> list[str]:
    """Every instrument the store covers.

    Reference identifiers where there are any, because a universe is a set of
    *instruments* and the market data also carries an identifier per FX pair --
    which would otherwise be offered as a universe member.
    """
    return list(fetcher.reference_identifiers or fetcher.identifiers)


def _standing_date(fetcher: DataFetcher,
                   date: pd.Timestamp | str | None) -> pd.Timestamp:
    """The date a filter is resolved at."""
    if date is not None:
        return pd.Timestamp(date)

    return fetcher.date_range[1]
