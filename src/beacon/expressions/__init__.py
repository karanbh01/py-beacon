# src/beacon/expressions/__init__.py
"""
Expressions: a typed, autocompleting way to refer to a datapoint.

    from beacon.expressions import data

    data.reference.sector == "Financials"
    data.market.adv_3m > 1_000_000

An expression is a tree, not a value. It serialises into the `params` of a
stored rule, so a screen written in Python and one built in the client are the
same document.
"""
from .core import (
    All,
    Any_,
    Comparison,
    Expression,
    Field,
    Not,
    distinct_fields_in,
    fields_in,
    from_dict,
)
from .namespaces import Data, Features, FeatureType, Namespace, column_for, data

__all__ = [
    "All",
    "Any_",
    "Comparison",
    "Data",
    "Expression",
    "FeatureType",
    "Features",
    "Field",
    "Namespace",
    "Not",
    "column_for",
    "data",
    "distinct_fields_in",
    "fields_in",
    "from_dict",
]
