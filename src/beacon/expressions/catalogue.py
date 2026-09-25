# src/beacon/expressions/catalogue.py
"""
Every datapoint an expression can name, listed from the loaded store.

`GET /data/features/catalogue` publishes the feature fields. This module lists
the rest (market, reference and corporate-action fields) alongside them, so a
client builds **one** field picker rather than one per dataset, and the picker
and the expression API cannot disagree about what exists.

Fields are read from the store, not only from the declarations in
`namespaces.py`: a store may carry columns the declarations do not know about,
and those are the columns a user loaded themselves. The declarations still
decide two things the data cannot say: which market fields are *derived*
(computed per request, so stored nowhere), and which action fields exist
(`kind` and `status` are computed when the API returns an action, not stored).
"""
# The feature catalogue endpoint arrived in BN-137; this module is the other
# half of that picker.
from typing import Any

from ..data.fetcher import DataFetcher
from .namespaces import (
    ACTION_COLUMNS,
    ACTIONS,
    DERIVED_COLUMNS,
    FEATURES,
    MARKET,
    REFERENCE,
)

# Stored columns that identify a row rather than describe an instrument. They
# are part of every frame and screening on them is meaningless, so listing
# them in a field picker is noise.
NOT_SCREENABLE = ("IDENTIFIER", "DATE", "DATE_FROM", "DATE_TO", "EX_DATE")


def describe_fields(fetcher: DataFetcher) -> list[dict[str, Any]]:
    """Every field a client can offer, in picker order.

    Args:
        fetcher: The loaded store.

    Returns:
        list[dict]: One entry per datapoint, each naming its namespace, its
        path, and whether it is derived.
    """
    entries = []

    entries.extend(_market(fetcher))
    entries.extend(_reference(fetcher))
    entries.extend(_actions())
    entries.extend(_features(fetcher))

    return entries


def _entry(namespace: str,
           name: str,
           dataset: str | None = None,
           derived: bool = False) -> dict[str, Any]:
    """One descriptor."""
    path = ".".join(part for part in (namespace, dataset, name) if part)

    return {"path": path, "namespace": namespace, "name": name,
            "dataset": dataset, "derived": derived}


def _market(fetcher: DataFetcher) -> list[dict[str, Any]]:
    stored = [_entry(MARKET, column.lower())
              for column in fetcher.market_columns
              if column not in NOT_SCREENABLE]
    derived = [_entry(MARKET, name, derived=True) for name in DERIVED_COLUMNS]

    return stored + derived


def _reference(fetcher: DataFetcher) -> list[dict[str, Any]]:
    return [_entry(REFERENCE, column.lower())
            for column in (fetcher.reference_columns or [])
            if column not in NOT_SCREENABLE]


def _actions() -> list[dict[str, Any]]:
    return [_entry(ACTIONS, name) for name in ACTION_COLUMNS]


def _features(fetcher: DataFetcher) -> list[dict[str, Any]]:
    return [_entry(FEATURES, name, dataset=dataset)
            for dataset in fetcher.feature_types()
            for name in fetcher.feature_fields(dataset)]
