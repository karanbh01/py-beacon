# src/beacon/synthetic/dataset.py
"""
Assembling a synthetic universe into the containers the rest of Beacon reads.

This is the layer that turns four panels into a `DataFetcher`, and the one
place that knows the whole dataset is meant to be mutually consistent: the
reference data names the same identifiers the market data prices, the action
history refers only to those identifiers, and the shares outstanding move on
exactly the dates the split actions record.

## This is not `beacon.testing.dataset`

They are deliberately different things and neither should grow into the other.

`beacon.testing.dataset` is a tiny frozen fixture — five names, a few hundred
days, price paths built from `+` and `*` alone so the numbers are bit-identical
on every platform. Chart baselines and unit tests depend on those exact values,
so it must never change.

This module is a *generator*: hundreds of names, years of history, drawn from a
model whose parameters are meant to be tuned as the model improves. Nothing
should assert on its exact values. Use the fixture when a test needs a known
number, and this when something needs to look like a market.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import store
from ..data.base import MarketData, ReferenceData
from ..data.corporate_actions import CorporateActions
from ..data.features import FeatureData
from ..data.fetcher import DataFetcher
from . import features as features_module
from . import fx, listings, prices, returns, universe

logger = logging.getLogger(__name__)

# Matches the universe pane in the client, so the default generation fills it.
DEFAULT_ASSETS = 512

# A fixed window, so `generate(SyntheticConfig())` is reproducible on any day.
# The CLI defaults differently — it ends today, because demo data that stopped
# eighteen months ago reads as stale in every freshness indicator.
DEFAULT_START = "2019-12-31"
DEFAULT_END = "2024-12-31"

DEFAULT_SEED = 42

# Annualised, and roughly the long-run US figures. Both configurable: the
# premium is the single number that decides whether five years of generated
# history looks like a bull market or a lost decade.
DEFAULT_RISK_FREE_RATE = 0.03
DEFAULT_EQUITY_PREMIUM = 0.06


@dataclass(frozen=True)
class SyntheticConfig:
    """What to generate.

    Attributes:
        assets: How many names.
        start: First date of the panel.
        end: Last date of the panel.
        seed: Everything random is drawn from this, so the same seed and the
            same dates produce the same dataset.
        risk_free_rate: Annualised, the base of the CAPM drift.
        equity_premium: Annualised excess return on a beta-one name.
        currency: Reporting currency for every name.
        delisting_rate: Annualised hazard of a name leaving the universe.
            Zero gives the constant, survivorship-biased universe every
            dataset had before BN-130.
        listing_rate: Annualised hazard of a name having joined partway
            through rather than at the start.
        features: Generate fundamental ratios and a little alternative data.
            On by default and cheap -- about a tenth of the market panel --
            because a dataset with nothing to screen on cannot exercise a
            feature rule, and a store that silently lacks one is harder to
            diagnose than one that costs a few hundred thousand rows.
    """
    assets: int = DEFAULT_ASSETS
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    seed: int = DEFAULT_SEED
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    equity_premium: float = DEFAULT_EQUITY_PREMIUM
    currency: str = universe.DEFAULT_CURRENCY
    delisting_rate: float = listings.ANNUAL_DELISTING_RATE
    listing_rate: float = listings.ANNUAL_LISTING_RATE
    features: bool = True

    def __post_init__(self) -> None:
        if self.assets < 1:
            raise ValueError(f"assets must be at least 1, got {self.assets}.")
        if pd.Timestamp(self.end) <= pd.Timestamp(self.start):
            raise ValueError(
                f"end ({self.end}) must fall after start ({self.start}).")


@dataclass(frozen=True)
class SyntheticDataset:
    """A generated universe and everything drawn from it.

    Attributes:
        market: OHLCV, shares outstanding and free float.
        reference: Names, classification, exchange and currency.
        actions: Dividends and splits, matching the price path.
        universe: The per-name parameters behind the draw. Exposed because the
            targets are what a statistical check should compare against — the
            realised figures are a sample from them, not the same thing.
        returns: The economic total returns the prices were built from. Not
            recoverable from `market` alone without undoing the dividends and
            splits, which is precisely what a coherence test does.
        features: Fundamental ratios and alternative data, derived from the
            prices above so a valuation screen and a price screen agree.
    """
    market: MarketData
    reference: ReferenceData
    actions: CorporateActions
    universe: pd.DataFrame
    returns: pd.DataFrame
    features: FeatureData = field(default_factory=FeatureData.empty)

    def fetcher(self) -> DataFetcher:
        """A `DataFetcher` over the generated data."""
        return DataFetcher(self.market, self.reference, self.actions,
                           self.features)


def generate(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate a synthetic dataset.

    Args:
        config: What to generate; defaults to 512 names over a fixed five-year
            window.

    Returns:
        SyntheticDataset: The panels and the parameters behind them.
    """
    settings = config if config is not None else SyntheticConfig()

    # One generator threaded through every draw. Separate generators per stage
    # would make each stage reproducible on its own and the dataset as a whole
    # reproducible only by accident.
    rng = np.random.default_rng(settings.seed)

    dates = pd.bdate_range(settings.start, settings.end)
    names = universe.build(settings.assets, rng, dates=dates,
                           currency=settings.currency,
                           delisting_rate=settings.delisting_rate,
                           listing_rate=settings.listing_rate)

    panel = returns.simulate(names, dates, rng,
                             risk_free_rate=settings.risk_free_rate,
                             equity_premium=settings.equity_premium)

    market_frame, action_frame = prices.build(names, panel, rng)

    # FX pairs live in the market data as identifiers of their own, which is
    # how `fetch_fx_rates` finds them. Concatenated rather than merged: an
    # exchange rate has no volume, no shares outstanding and no free float,
    # and inventing zeros for those would make an FX row answer questions it
    # has no answer to.
    rates = fx.build(dates, rng)

    # Derived from the prices above rather than drawn beside them: a P/E that
    # contradicts the price series in the same dataset would make a valuation
    # screen and a price screen disagree about the same company.
    close = market_frame.pivot(index="DATE", columns="IDENTIFIER",
                               values="CLOSE") if settings.features else None
    feature_rows = (features_module.build(names, close, panel, rng)
                    if settings.features else None)

    logger.info("Generated %d identifier(s) over %d business days (seed %d).",
                settings.assets, len(dates), settings.seed)

    return SyntheticDataset(
        market=MarketData.from_dataframe(
            pd.concat([market_frame, rates], ignore_index=True)),
        reference=ReferenceData.from_dataframe(
            universe.reference_frame(names, settings.start)),
        actions=CorporateActions.from_dataframe(action_frame),
        features=(FeatureData.from_dataframe(feature_rows)
                  if feature_rows is not None else FeatureData.empty()),
        universe=names,
        returns=panel)


def write(config: SyntheticConfig,
          path: Path) -> Path:
    """Generate a dataset and write it as a Beacon data store.

    Args:
        config: What to generate.
        path: Store directory, created if absent.

    Returns:
        Path: The directory written.
    """
    dataset = generate(config)

    return store.save(dataset.fetcher(), path, source=store.SOURCE_SYNTHETIC)
