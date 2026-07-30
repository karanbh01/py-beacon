# src/beacon/testing/dataset.py
"""
The canonical synthetic dataset.

One fixed universe, one fixed date span, one set of price paths. Every test,
example and documentation page that needs "some data" should use this rather
than growing another local helper, because scaffolding written six times in six
slightly different shapes means six subtly different answers and no way to tell
which one a baseline was built against.

## Reproducibility, and why the maths is deliberately dull

The paths are built from `+` and `*` only — no `exp`, no `log`. That is not
an accident and not a simplification:

* numpy's `Generator` produces bit-identical draws across platforms for a given
  seed, because the ziggurat sampler is numpy's own C rather than the system
  library.
* IEEE 754 pins the results of `+`, `-`, `*` and `/` exactly, so every platform
  agrees on them to the last bit.
* `exp` and `log` are *not* pinned that way. They come from the platform's libm
  and are allowed to differ in the last unit in the last place, which
  compounding then amplifies.

So returns are simple rather than logarithmic and prices compound by
multiplication. A geometric path built the textbook way through `exp` would be
reproducible on one machine and off by a hair on another — invisible in a unit
test, and exactly the kind of thing that makes an image-regression baseline
fail on a different runner for no reason anyone can see.

## What is in it

Six assets chosen so the data has something to say:

* **AAA** and **BBB** are close substitutes — high beta, small idiosyncratic
  noise — so correlation, shrinkage and substitution effects have a real signal
  to find.
* **CCC** is defensive: low beta, low volatility, and only weakly correlated
  with the rest. Tuned that way on purpose — with a merely low-volatility CCC
  the minimum-variance portfolio put 100% into it, and a corner solution tests
  an optimiser far less than an interior one. As it stands the answer blends
  three names and beats the least volatile single asset, so diversification has
  to be working for the numbers to come out right.
* **DDD** is the volatile high-flyer, **EEE** the laggard, so return and risk
  rankings disagree and an optimiser has a genuine trade-off.
* **FFF** trades in GBP, so anything touching FX has a case that exercises it.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..data.base import MarketData, ReferenceData
from ..data.fetcher import DataFetcher

# Changing any of these changes every baseline built on this dataset. That is
# the point of naming them here: it should be a deliberate, visible act.
SEED = 20240101
START = "2023-01-02"
END = "2025-12-31"

BASE_CURRENCY = "USD"
FX_PAIR = "GBPUSD"

# Prices are rounded to this many decimals. Belt and braces on top of the
# arithmetic choice above: it also keeps a CSV round trip exact.
PRICE_DECIMALS = 6

# Daily market factor volatility. Everything else is expressed relative to it.
MARKET_VOLATILITY = 0.008
MARKET_DRIFT = 0.0003


@dataclass(frozen=True)
class Constituent:
    """One synthetic company and the behaviour its price path should show.

    Attributes:
        identifier: Ticker.
        name: Display name.
        sector: Classification, for group constraints and breakdowns.
        currency: Trading currency.
        initial_price: Price on the first date.
        beta: Sensitivity to the common market factor.
        idiosyncratic: Daily volatility of the asset's own noise.
        drift: Daily expected return on top of the market.
        shares_outstanding: For market-cap weighting.
        free_float: Fraction of shares actually investable.
    """
    identifier: str
    name: str
    sector: str
    currency: str
    initial_price: float
    beta: float
    idiosyncratic: float
    drift: float
    shares_outstanding: int
    free_float: float


CONSTITUENTS = (
    Constituent("AAA", "Alpha Industries", "Technology", "USD",
                100.0, 1.10, 0.004, 0.00030, 1_000_000_000, 0.90),
    Constituent("BBB", "Beta Systems", "Technology", "USD",
                75.0, 1.05, 0.004, 0.00025, 800_000_000, 0.85),
    Constituent("CCC", "Gamma Utilities", "Utilities", "USD",
                50.0, 0.25, 0.005, 0.00012, 1_500_000_000, 1.00),
    Constituent("DDD", "Delta Dynamics", "Industrials", "USD",
                200.0, 1.35, 0.011, 0.00045, 300_000_000, 0.75),
    Constituent("EEE", "Epsilon Retail", "Consumer", "USD",
                25.0, 0.85, 0.007, 0.00005, 2_000_000_000, 0.95),
    Constituent("FFF", "Zeta Holdings", "Financials", "GBP",
                60.0, 0.95, 0.005, 0.00018, 600_000_000, 0.80),
)

UNIVERSE = tuple(constituent.identifier for constituent in CONSTITUENTS)


def trading_days() -> pd.DatetimeIndex:
    """The dataset's business-day calendar."""
    return pd.bdate_range(START, END)


@lru_cache(maxsize=1)
def _paths() -> pd.DataFrame:
    """Generate every price path once.

    Cached because generation is not free and every accessor needs it. Callers
    never see this frame — they get copies — so one test cannot mutate the
    dataset another test is about to read.
    """
    dates = trading_days()
    periods = len(dates)
    generator = np.random.default_rng(SEED)

    # Drawn first, and once, so the market factor is identical regardless of
    # how many constituents follow it.
    market = MARKET_DRIFT + MARKET_VOLATILITY * generator.standard_normal(periods)

    prices = {}
    for constituent in CONSTITUENTS:
        noise = generator.standard_normal(periods)
        returns = (constituent.drift
                   + constituent.beta * market
                   + constituent.idiosyncratic * noise)

        # Simple compounding, no exponentials: see the module docstring.
        path = constituent.initial_price * np.cumprod(1.0 + returns)
        prices[constituent.identifier] = np.round(path, PRICE_DECIMALS)

    return pd.DataFrame(prices, index=dates)


@lru_cache(maxsize=1)
def _fx_path() -> pd.Series:
    """A GBPUSD series, so FX conversion has something to convert."""
    dates = trading_days()
    generator = np.random.default_rng(SEED + 1)

    returns = 0.00002 + 0.004 * generator.standard_normal(len(dates))
    path = 1.25 * np.cumprod(1.0 + returns)

    return pd.Series(np.round(path, PRICE_DECIMALS), index=dates, name=FX_PAIR)


def prices() -> pd.DataFrame:
    """Closing prices, dates on the index and identifiers on the columns.

    Returns:
        pd.DataFrame: A copy, so callers may modify it freely.
    """
    return _paths().copy()


def returns() -> pd.DataFrame:
    """Daily simple returns, with the first (undefined) row dropped."""
    return prices().pct_change().dropna(how="all")


def fx_rates() -> pd.Series:
    """The GBPUSD series."""
    return _fx_path().copy()


def market_frame() -> pd.DataFrame:
    """The long-form market data, in the shape MarketData expects.

    One row per identifier per date, carrying OHLC, volume, shares outstanding
    and free float — enough for market-cap weighting and for a price_column
    override to have something else to point at.
    """
    frames = [_constituent_rows(constituent) for constituent in CONSTITUENTS]
    frames.append(_fx_rows())

    return pd.concat(frames, ignore_index=True)


def _constituent_rows(constituent: Constituent) -> pd.DataFrame:
    """Long-form rows for one constituent."""
    close = _paths()[constituent.identifier]

    # Derived from close by fixed ratios rather than drawn separately: an
    # intraday range that could invert (low above high) would be a distracting
    # thing to discover inside an unrelated test.
    return pd.DataFrame({
        "IDENTIFIER": constituent.identifier,
        "DATE": close.index,
        "OPEN": np.round(close.to_numpy() * 0.999, PRICE_DECIMALS),
        "HIGH": np.round(close.to_numpy() * 1.008, PRICE_DECIMALS),
        "LOW": np.round(close.to_numpy() * 0.992, PRICE_DECIMALS),
        "CLOSE": close.to_numpy(),
        "VOLUME": 1_000_000,
        "SHARES_OUTSTANDING": constituent.shares_outstanding,
        "FREE_FLOAT": constituent.free_float,
    })


def _fx_rows() -> pd.DataFrame:
    """The FX pair, stored as its own identifier like any other series."""
    series = _fx_path()

    return pd.DataFrame({
        "IDENTIFIER": FX_PAIR,
        "DATE": series.index,
        "OPEN": series.to_numpy(),
        "HIGH": series.to_numpy(),
        "LOW": series.to_numpy(),
        "CLOSE": series.to_numpy(),
        "VOLUME": 0,
        "SHARES_OUTSTANDING": np.nan,
        "FREE_FLOAT": np.nan,
    })


def reference_frame() -> pd.DataFrame:
    """The reference data, valid across the whole span.

    One open-ended validity row per constituent. Point-in-time classification
    changes are BN-98's territory; this stays deliberately simple so a test
    that does not care about validity windows does not have to think about
    them.
    """
    return pd.DataFrame([
        {"IDENTIFIER": constituent.identifier,
         "NAME": constituent.name,
         "CURRENCY": constituent.currency,
         "EXCHANGE": "LSE" if constituent.currency == "GBP" else "NYSE",
         "SECTOR": constituent.sector,
         "DATE_FROM": pd.Timestamp(START),
         "DATE_TO": pd.NaT}
        for constituent in CONSTITUENTS
    ])


def market_data() -> MarketData:
    """The dataset as a MarketData container."""
    return MarketData.from_dataframe(market_frame())


def reference_data() -> ReferenceData:
    """The dataset as a ReferenceData container."""
    return ReferenceData.from_dataframe(reference_frame())


def data_fetcher() -> DataFetcher:
    """The dataset as a DataFetcher — the usual entry point.

    Returns:
        DataFetcher: Wired to the full universe, the FX pair and the reference
        data, ready to hand to an IndexCalculator or a BacktestEngine.
    """
    return DataFetcher(market_data=market_data(),
                       reference_data=reference_data())


def sectors() -> dict[str, list[str]]:
    """Constituents grouped by sector, for group-constraint tests."""
    grouped: dict[str, list[str]] = {}
    for constituent in CONSTITUENTS:
        grouped.setdefault(constituent.sector, []).append(constituent.identifier)

    return grouped


def equal_weights() -> dict[str, float]:
    """An equally weighted portfolio over the universe."""
    return dict.fromkeys(UNIVERSE, 1.0 / len(UNIVERSE))
