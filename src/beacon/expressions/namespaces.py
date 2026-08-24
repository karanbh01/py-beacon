# src/beacon/expressions/namespaces.py
"""
Where `data.market.close` and `data.features.fundamentals.revenue` come from.

    from beacon.expressions import data

    data.reference.sector == "Financials"
    data.market.market_cap > 1e9
    data.features.fundamentals.revenue > 1e9

## Imported from `beacon.expressions`, not from `beacon`

`from beacon import data` would be the obvious spelling and it cannot work:
`beacon.data` is already the data *package*, and importing any of its
submodules rebinds that name on the parent. An expression root living there
would be whichever won the import race — which is exactly what happened, with
tests passing alone and failing after anything that imported
`beacon.data.store`. `beacon.data` raises a message naming this module for the
three namespaces where the mistake is plausible.

`data` is a **description, not a dataset**. It is a module-level symbol bound
to nothing, because an expression is written before there is anything to
evaluate it against — in a script, in a saved definition, in a client. Binding
it to a loaded store would make the import order matter and the same screen
mean different things in two processes.

## Declared where there is a contract, open where there is not

The split is what makes autocomplete possible without a generation step.

**Market and reference columns are declared.** They are a documented contract
(`beacon.data`), so they can be listed here and complete everywhere — in
Jupyter, in an IDE, in `dir()` — and they cannot drift, because this list *is*
the contract rather than a copy of it.

**Feature types and fields are open.** Somebody loads `satellite_imagery`
tomorrow and it has to work with no code change, so that half accepts any
attribute and is checked against the loaded data instead (BN-141).

Declared does not mean closed. A store carrying an extra reference column
resolves too — the declaration is what is *known in advance*, not what is
allowed.

## Lower case here, upper case in storage

`data.reference.sector` resolves to the `SECTOR` column. The API should read
like Python and the store should read like a data feed, and mapping between
them is one line here rather than a convention every caller has to remember.

## Derived fields resolve like stored ones

`adv_3m`, `market_cap` and `free_float_market_cap` are computed per request
rather than stored (BN-133). A user should not have to know which side of that
line a datapoint falls on, so they live in the market namespace beside the
stored columns and carry a flag saying they are derived.
"""
from typing import Any

from ..exceptions import UnknownDatasetError
from .core import Field

# The namespaces themselves.
MARKET = "market"
REFERENCE = "reference"
ACTIONS = "actions"
FEATURES = "features"

# Declared market columns: the stored price/volume series every market data
# frame carries.
MARKET_COLUMNS = (
    "open", "high", "low", "close", "volume",
    "shares_outstanding", "free_float",
)

# Computed per request rather than stored. Listed beside the stored columns
# because a caller should not have to know which is which.
DERIVED_COLUMNS = ("adv_3m", "market_cap", "free_float_market_cap")

# Declared reference dimensions.
REFERENCE_COLUMNS = (
    "name", "sector", "sub_industry", "region", "exchange", "currency",
    "country_listing", "country_domicile",
)

# Declared corporate-action fields.
ACTION_COLUMNS = ("type", "kind", "value", "ex_date", "pay_date", "status")


class Namespace:
    """One dataset's fields, reached by attribute.

    Declared names complete and are documented; anything else still resolves,
    because a store may carry columns this list has never heard of. What is
    *not* allowed is a private or dunder name, which would otherwise turn a
    typo like `data.market.__deepcopy__` into a field.
    """

    def __init__(self,
                 namespace: str,
                 columns: tuple[str, ...] = (),
                 derived: tuple[str, ...] = ()):
        self._namespace = namespace
        self._columns = columns
        self._derived = derived

    def __getattr__(self,
                    name: str) -> Field:
        # `__getattr__` runs for anything not found normally, which includes
        # every dunder Python looks up on an object it is copying, pickling or
        # inspecting. Returning a Field for those makes the namespace behave
        # bizarrely under a debugger, so they are refused as missing.
        if name.startswith("_"):
            raise AttributeError(name)

        return Field(self._namespace, name)

    def __dir__(self) -> list[str]:
        """What Jupyter and IPython complete on.

        Only the declared names: an open namespace has no list to offer, and
        suggesting nothing is better than suggesting something wrong.
        """
        return sorted({*self._columns, *self._derived})

    @property
    def declared(self) -> tuple[str, ...]:
        """The documented columns, derived ones included."""
        return tuple(sorted({*self._columns, *self._derived}))

    def is_derived(self,
                   name: str) -> bool:
        """Whether a field is computed per request rather than stored."""
        return name in self._derived

    def __repr__(self) -> str:
        return f"<data.{self._namespace}>"


class FeatureType(Namespace):
    """One feature dataset — `data.features.fundamentals`.

    Wholly open: the fields a dataset carries are whatever was loaded, and a
    fixed list here would be wrong the first time somebody imported their own.
    """

    def __init__(self,
                 dataset: str):
        super().__init__(FEATURES)
        self._dataset = dataset

    def __getattr__(self,
                    name: str) -> Field:
        if name.startswith("_"):
            raise AttributeError(name)

        return Field(FEATURES, name, dataset=self._dataset)

    def __repr__(self) -> str:
        return f"<data.features.{self._dataset}>"


class Features:
    """The feature namespace, which nests by dataset type.

    `data.features.fundamentals.revenue`, not `data.features.revenue`.

    `TYPE` is what separates datasets sharing one table (BN-134): `revenue`
    from a vendor and `revenue` from a user's own model are different series.
    Flattening them here would leave the API unable to say which it meant at
    exactly the moment the user is choosing between them.
    """

    def __getattr__(self,
                    name: str) -> FeatureType:
        if name.startswith("_"):
            raise AttributeError(name)

        return FeatureType(name)

    def __dir__(self) -> list[str]:
        """Nothing to offer.

        Which feature types exist is a property of the loaded data, not of
        this module, and inventing a plausible list would complete names that
        do not resolve.
        """
        return []

    def __repr__(self) -> str:
        return "<data.features>"


class Data:
    """The root: `data`.

    Deliberately a small fixed set of namespaces. Unlike the fields inside
    them, the datasets Beacon holds are a closed contract — a typo like
    `data.refrence.sector` should fail at the attribute rather than build a
    field in a namespace nothing will ever resolve.
    """

    def __init__(self) -> None:
        self.market = Namespace(MARKET, MARKET_COLUMNS, DERIVED_COLUMNS)
        self.reference = Namespace(REFERENCE, REFERENCE_COLUMNS)
        self.actions = Namespace(ACTIONS, ACTION_COLUMNS)
        self.features = Features()

    def __getattr__(self,
                    name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        raise UnknownDatasetError(
            f"there is no '{name}' dataset. Expected one of "
            f"{', '.join(NAMESPACES)}.")

    def __dir__(self) -> list[str]:
        return list(NAMESPACES)

    def __repr__(self) -> str:
        return "<beacon.data>"


NAMESPACES = (MARKET, REFERENCE, ACTIONS, FEATURES)

# The module-level root. One instance, bound to no dataset.
data = Data()


def column_for(field: Field) -> str:
    """The stored column a field names.

    Upper case for market, reference and action columns, which are stored that
    way; feature fields keep their case, because a feature's `FIELD` is
    whatever the loader wrote and upper-casing it would stop matching.
    """
    if field.namespace == FEATURES:
        return field.name

    return field.name.upper()
