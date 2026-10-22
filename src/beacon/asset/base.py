# src/beacon/asset/base.py
"""
Module defining the base class for financial assets.
"""
from dataclasses import dataclass, fields
from typing import Any


# BN-185 made the index pipeline equity-only.
@dataclass(frozen=True)
class Asset:
    """Base class for a financial asset. An immutable metadata container.

    Every field must be non-empty, or construction raises `ValueError`.

    **The index pipeline accepts only :class:`~beacon.asset.equity.Equity`.**
    Subclasses such as :class:`~beacon.asset.bond.Bond` and
    :class:`~beacon.asset.commodity.Commodity` are usable as metadata, but a
    universe containing one is refused by selection, weighting, market values
    and corporate-action handling alike; see
    :func:`~beacon.asset.equity.require_equity`.
    """
    name: str
    currency: str
    asset_id: str = ""
    asset_type: str = ""

    def __init_subclass__(cls,
                          **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Before `@dataclass` sees the subclass, so it keeps this hash rather
        # than generating one of its own (BN-351).
        cls.__hash__ = Asset.__hash__  # type: ignore[method-assign]

    def __post_init__(self) -> None:
        for field_name in ('asset_id', 'asset_type', 'name', 'currency'):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty.")

        # Last, after a subclass has filled its defaults in.
        object.__setattr__(self, "_hash", self._field_hash())

    def _field_hash(self) -> int:
        """The hash `@dataclass` generates: of the compared fields, in order."""
        return hash(tuple(getattr(self, field.name) for field in fields(self)
                          if (field.compare if field.hash is None else field.hash)))

    # Computed once (BN-351): the generated hash builds a tuple of every field
    # on each call, and the index's daily loop hashes its holdings millions of
    # times over a long run. The value is the same, so dicts and sets of
    # assets behave exactly as before.
    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined, no-any-return]

    # String hashes differ between processes, so a pickled asset carries its
    # fields only and works its hash out again where it lands.
    def __getstate__(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items() if name != "_hash"}

    def __setstate__(self,
                     state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        object.__setattr__(self, "_hash", self._field_hash())
