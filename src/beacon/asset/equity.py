# src/beacon/asset/equity.py
"""
Module defining the Equity asset class, and the index pipeline's equity-only gate.
"""
from dataclasses import dataclass

from ..exceptions import CalculationError
from .base import Asset


# BN-185 made the index pipeline equity-only.
@dataclass(frozen=True)
class Equity(Asset):
    """Represents an equity security.

    `asset_id` defaults to `ticker` and `asset_type` to ``"EQUITY"``;
    `ticker` and `exchange` must be non-empty.

    **This is the only asset type the index pipeline accepts.**
    Selection, weighting, market values and corporate-action divisor
    adjustments all read market data keyed by :attr:`ticker` and all reason in
    terms of shares outstanding and free float, none of which the other
    :class:`~beacon.asset.base.Asset` subclasses carry. A constituent that is
    not an equity is refused by :func:`require_equity` at whichever of those
    layers meets it first, rather than admitted, skipped, or zeroed.
    """
    ticker: str = ""
    exchange: str = ""
    isin: str | None = None
    sector: str | None = None
    country: str | None = None

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker cannot be empty.")
        if not self.exchange:
            raise ValueError("exchange cannot be empty.")
        if not self.asset_id:
            object.__setattr__(self, 'asset_id', self.ticker)
        if not self.asset_type:
            object.__setattr__(self, 'asset_type', 'EQUITY')
        super().__post_init__()


# BN-185: one answer, at six sites that used to give four. Selection admitted a
# non-equity, weighting raised, the market-value path skipped it with a
# warning or valued it at zero, and the corporate-action path returned the
# divisor unchanged in silence: four readings of one question, drifted apart
# because nothing exercised them.
#
# Every one of the lenient answers is the same substitution wearing a
# different hat: an index computed over a subset of its own universe, with
# levels that sum, weights that normalise and a backtest that tracks, so
# nothing downstream looks wrong enough for anyone to ask. If a non-equity
# ever does reach the calculation, that is a defect in the universe worth
# surfacing rather than absorbing.
def require_equity(asset: Asset,
                   calculation_name: str,
                   action: str) -> Equity:
    """Return *asset* as an :class:`Equity`, or raise if it is not one.

    Every stage of the index pipeline (selection, weighting, market values and
    corporate-action adjustments) calls this, so a non-equity constituent is
    refused rather than skipped, zeroed or silently admitted.

    Args:
        asset: The constituent to check.
        calculation_name: What is refusing: a rule, scheme, or stage name.
        action: The passive verb phrase completing "it cannot ..." ("be
            weighted", "be valued"), so the message says what could not be done.

    Returns:
        Equity: *asset*, narrowed, so callers get the ticker-bearing type back
        rather than checking and casting separately.

    Raises:
        CalculationError: If *asset* is not an :class:`Equity`. The message
            names the asset and its actual type, because "not an equity" alone
            does not say which name in the universe to go and look at.
    """
    if isinstance(asset, Equity):
        return asset

    raise CalculationError(
        calculation_name=calculation_name,
        details=(f"constituent '{asset.asset_id}' is a {type(asset).__name__}, "
                 f"not an equity, so it cannot {action}. The index pipeline is "
                 f"equity-only: every stage prices, weights and adjusts through "
                 f"an equity ticker, and admitting this name anyway would "
                 f"publish an index over a subset of its own universe without "
                 f"saying so. Remove it from the universe."))
