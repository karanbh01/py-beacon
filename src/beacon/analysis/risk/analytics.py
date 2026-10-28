# src/beacon/analysis/risk/analytics.py
"""
A run's risk analytics in one place: its factors, value at risk, risk over
time, concentration and liquidity, each built when first asked for and kept.
"""
# BN-397.
from collections.abc import Sequence
from functools import cached_property

from ...data.fetcher import DataFetcher
from ...strategy.factors import Factor
from ..data import RunData
from ..performance.settings import AnalysisSettings
from .concentration import ConcentrationAnalytics
from .contributions import RiskOverTime
from .factors import FactorAnalytics
from .liquidity import LiquidityAnalytics
from .var import RiskOfLoss


class RiskAnalytics:
    """A run's risk.

    Args:
        run: The run's data.
        source: The data the run read, which the factors and traded values
            are measured from.
        factors: The factors; `beacon.strategy.default_factors` when None.
        settings: The window and conventions the value at risk is measured
            with.
    """

    def __init__(self,
                 run: RunData,
                 source: DataFetcher,
                 factors: Sequence[Factor] | None = None,
                 settings: AnalysisSettings | None = None):
        self.run = run
        self.source = source
        self._factors = list(factors) if factors is not None else None
        self.settings = settings

    @cached_property
    def factors(self) -> FactorAnalytics:
        """The run's factor exposures, factor returns and betas."""
        return FactorAnalytics(self.run, self.source, self._factors)

    @cached_property
    def loss(self) -> RiskOfLoss:
        """The run's value at risk, expected shortfall and stress windows."""
        return RiskOfLoss(self.run, self.settings)

    @cached_property
    def over_time(self) -> RiskOverTime:
        """The run's expected and measured risk, and where it came from."""
        return RiskOverTime(self.factors)

    @cached_property
    def concentration(self) -> ConcentrationAnalytics:
        """The run's concentration by weight, risk and outcome."""
        return ConcentrationAnalytics(self.run, self.over_time)

    @cached_property
    def liquidity(self) -> LiquidityAnalytics:
        """The run's liquidity at each rebalance and month end."""
        return LiquidityAnalytics(self.run, self.source)
