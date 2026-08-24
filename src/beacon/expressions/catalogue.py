# src/beacon/expressions/catalogue.py
"""
Every datapoint an expression can name, listed from the loaded store.

`GET /data/features/catalogue` (BN-137) already publishes the feature half.
This is the other half — market, reference and corporate-action fields — so a
client builds **one** field picker rather than one per dataset, and so the
picker and the expression API cannot disagree about what exists.

## Read from the store, not from the declaration

`namespaces.py` declares what is known in advance; a store may carry more. A
catalogue built from the declaration would omit exactly the columns a user
loaded themselves — the ones they are most likely to be looking for.

The declaration is still used for two things the data cannot say: which market
fields are *derived* (they are columns nowhere, being computed per request),
and which action fields exist (`kind` and `status` are computed on the way out
of the API rather than stored).
"""
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
