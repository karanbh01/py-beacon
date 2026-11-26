# src/beacon/backtest/trading_record.py
"""
What the engine did at each rebalance: `result.trading`.

    result.trading.steps          what each rebalance's implementation did
    result.trading.pricing        the session each rebalance priced from
    result.trading.unfilled       the orders it could not fill
    result.trading.replication    an IndexTracking run's replication, else None
    result.trading.active         an ActiveStrategy's choices, else None

Recorded data only: what was measured from it is under `result.analytics`.
"""
# BN-460 (Karan, 2026-11-24): the engine's record in one place; the result's
# flat names warn until 0.12.0.
from dataclasses import dataclass, field

import pandas as pd

from ..index.result import PriceGap
from ..portfolio.base import Portfolio
from ..strategy.active import ActiveStep
from ..strategy.tracking import ReplicationStep
from .implementation import RebalanceStep

# Why an order went unfilled (BN-252, BN-266).
CASH_SHORT = "cash"
NO_PRICE = "no price"
EXECUTION_LIMIT = "execution limit"


@dataclass(frozen=True)
class UnfilledOrder:
    """An order the simulation could not execute in full.

    Recorded on the result rather than only logged: a partially filled
    rebalance leaves the portfolio off its target weights, and a caller
    comparing tracking error against expectations needs to know that happened
    rather than reading it as a modelling result.

    Attributes:
        date: The rebalance date.
        asset_id: Asset that could not be fully traded.
        requested_quantity: Quantity the rebalance asked for; 0.0 when the
            name could not be priced to size an order.
        filled_quantity: Quantity actually traded; 0.0 when nothing was.
        price: Execution price used; NaN when there was none.
        shortfall_value: Notional value that went unfilled; for a name with
            no price, the value the rebalance aimed to hold. NaN for an order
            an execution limit left unfinished.
        reason: Why: ``"cash"`` (not enough to buy it all), ``"no price"``
            (the name could not be priced on the day) or
            ``"execution limit"`` (still working when the next rebalance
            replaced it or the run ended).
    """
    date: pd.Timestamp
    asset_id: str
    requested_quantity: float
    filled_quantity: float
    price: float
    shortfall_value: float
    reason: str = CASH_SHORT


# RebalancePricing is BN-183. PriceGap, also BN-183, moved to the index
# layer with BN-251, when the index started recording gaps too.
@dataclass(frozen=True)
class RebalancePricing:
    """What one rebalance priced from.

    A rebalance scheduled on a day the market was shut still trades: it prices
    from the session in force through the closure. Recording both dates lets a
    reader tell that case apart from an ordinary rebalance. `date` and
    `priced_from` are equal for the ordinary rebalance, which is what makes an
    unequal pair worth reading.

    Attributes:
        date: The rebalance date from the weight schedule.
        priced_from: The session its prices were read from.
    """
    date: pd.Timestamp
    priced_from: pd.Timestamp


@dataclass
class TradingRecord:
    """What the engine did at each rebalance.

    Attributes:
        steps: What the screening and redistribution stages did at each
            rebalance: the target, the names removed and by what, and the
            weights traded to.
        pricing: What each rebalance priced from, in date order. `date` and
            `priced_from` differ only where the schedule landed on a day the
            market was shut.
        unfilled: Buys the simulation could not execute in full. Empty for a
            run where every rebalance leg filled, so a non-empty list is
            itself the signal that the portfolio drifted off target for a
            reason other than price movement.
        price_gaps: Days a name had no bar on a session its calendar says was
            open, and was therefore marked at a carried-forward price. A
            market holiday is not a gap, since nothing is missing on a day
            nothing traded.
        replication: For an index held through `IndexTracking`, what the
            replication did at each rebalance: the weights, the holdings and
            the ex-ante tracking error. None for any other strategy.
        active: For an `ActiveStrategy`, what it did at each rebalance: the
            weights, the benchmark's, the ex-ante tracking error, the active
            share and the scores. None for any other strategy.
    """
    portfolio: Portfolio = field(repr=False)
    steps: list[RebalanceStep] = field(default_factory=list)
    pricing: list[RebalancePricing] = field(default_factory=list)
    unfilled: list[UnfilledOrder] = field(default_factory=list)
    price_gaps: list[PriceGap] = field(default_factory=list)
    replication: list[ReplicationStep] | None = None
    active: list[ActiveStep] | None = None

    @property
    def unfilled_value(self) -> float:
        """The notional that went unfilled across the run."""
        return float(sum(order.shortfall_value for order in self.unfilled))

    @property
    def total_costs(self) -> float:
        """The transaction costs paid over the run, in its currency."""
        return float(sum(transaction.transaction_cost
                         for transaction in self.portfolio.transactions))


__all__ = ["CASH_SHORT", "EXECUTION_LIMIT", "NO_PRICE", "RebalancePricing", "TradingRecord",
           "UnfilledOrder"]
