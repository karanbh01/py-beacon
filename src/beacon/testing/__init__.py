# src/beacon/testing/__init__.py
"""
Deterministic synthetic data for tests, examples and documentation.

Shipped rather than kept under `tests/` because the documentation gallery and
the examples need the same numbers the tests use. A chart in the docs built
from a different helper than the baseline it is compared against is a chart
nobody can debug, and asking a reader to invent their own data before they can
run an example is friction with no upside.

    from beacon.testing import dataset

    fetcher = dataset.data_fetcher()
    prices = dataset.prices()

Everything here is generated from fixed constants and returns a fresh copy, so
two callers always see identical frames and neither can disturb the other. See
`dataset` for why the arithmetic avoids `exp` — the short version is that
bit-identical output across operating systems is a requirement, not a bonus.

Core-only: pandas and numpy, no optional dependency, because a test that needs
data should not first need an extra installed.
"""
from . import dataset
from .dataset import (
    BASE_CURRENCY,
    CONSTITUENTS,
    END,
    FX_PAIR,
    SEED,
    START,
    UNIVERSE,
    Constituent,
    data_fetcher,
    equal_weights,
    fx_rates,
    market_data,
    market_frame,
    prices,
    reference_data,
    reference_frame,
    returns,
    sectors,
    trading_days,
)

__all__ = [
    "BASE_CURRENCY",
    "CONSTITUENTS",
    "END",
    "FX_PAIR",
    "SEED",
    "START",
    "UNIVERSE",
    "Constituent",
    "data_fetcher",
    "dataset",
    "equal_weights",
    "fx_rates",
    "market_data",
    "market_frame",
    "prices",
    "reference_data",
    "reference_frame",
    "returns",
    "sectors",
    "trading_days",
]
