# src/beacon/analysis/reports/context.py
"""
What a run's statistics are worked out from: the run, the choices the
analysis makes, and each analysis of it, built the first time a statistic
asks for it.

    context = Context(run, options, source)
    context.book("index").headline()
    context.relative("benchmark").regression()

A statistic asks the context for what it needs; one the run cannot give,
such as a comparator it does not have, raises `Unavailable` saying why.
"""
# BN-398.
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cached_property
from typing import Any

import pandas as pd

from ...data.fetcher import DataFetcher
from ...strategy.factors import Factor
from ..attribution.analytics import Attribution
from ..class_panels import CLASS_DAYS
from ..data import RunData
from ..performance.analytics import PerformanceAnalytics
from ..performance.comparisons import aligned
from ..performance.crises import CrisisAnalytics
from ..performance.portfolio import PortfolioAnalytics
from ..performance.regimes import RegimeAnalytics
from ..performance.relative_analytics import RelativeAnalytics
from ..risk.concentration import ConcentrationAnalytics
from ..risk.contributions import RiskOverTime
from ..risk.factors import FactorAnalytics
from ..risk.liquidity import LiquidityAnalytics
from ..risk.var import RiskOfLoss
from .options import BENCHMARK, INDEX, OPTIMISED, AnalysisOptions
from .statistics import (
    NO_BENCHMARK,
    NO_DATA_STORE,
    NO_INDEX,
    NO_SHARE_CLASSES,
    Unavailable,
)

FUND = "fund"
CLASS = "class/"

# The comparators, in the order their figures are given.
COMPARATORS = (INDEX, OPTIMISED, BENCHMARK)


class Context:
    """A run and every analysis of it a statistic can read.

    Args:
        run: The run.
        options: The choices the analysis makes.
        source: The data store the run was made from, which factor,
            liquidity and risk figures read; None when it is not loaded.
        factors: The factors risk and attribution use; the defaults when
            None.
    """

    def __init__(self,
                 run: RunData,
                 options: AnalysisOptions | None = None,
                 source: DataFetcher | None = None,
                 factors: Sequence[Factor] | None = None):
        self.run = run
        self.options = options or AnalysisOptions()
        self.source = source
        self.factors = list(factors) if factors is not None else None
        self._kept: dict[tuple[Any, ...], Any] = {}

    def kept(self,
             key: tuple[Any, ...],
             build: Callable[[], Any]) -> Any:
        """What *build* gives, worked out once under *key*, so statistics
        read from the same call share it."""
        if key not in self._kept:
            self._kept[key] = build()

        return self._kept[key]

    # -- performance -------------------------------------------------------

    @cached_property
    def performance(self) -> PerformanceAnalytics:
        """The fund's performance."""
        return PerformanceAnalytics(self.run, self.options.window)

    @cached_property
    def comparators(self) -> list[str]:
        """The comparators the run has, in order."""
        return [name for name in COMPARATORS if name in self.run.comparators]

    @cached_property
    def tracked(self) -> str | None:
        """The comparator relative figures are measured against: the one
        asked for, else the optimised index when the run solved one, else
        the index; None when the run has none of them."""
        chosen = self.options.window.tracked or (OPTIMISED if OPTIMISED in self.comparators
                                                  else INDEX)

        return chosen if chosen in self.comparators else None

    def require(self,
                comparator: str) -> str:
        """*comparator*, or `Unavailable` when the run lacks it."""
        if comparator not in self.comparators:
            raise Unavailable(NO_BENCHMARK if comparator == BENCHMARK else NO_INDEX)

        return comparator

    def require_tracked(self) -> str:
        """The tracked comparator, or `Unavailable` when the run has none."""
        if self.tracked is None:
            raise Unavailable(NO_BENCHMARK if self.options.window.tracked == BENCHMARK
                              else NO_INDEX)

        return self.tracked

    @cached_property
    def classes(self) -> list[str]:
        """The run's share classes, by name; none for a run without them."""
        days = self.run.tables.get(CLASS_DAYS) if CLASS_DAYS in self.run.tables else None

        return [] if days is None or days.empty else list(pd.unique(days["CLASS"].astype(str)))

    def books(self) -> list[str]:
        """Every book a figure of returns is given for: the fund, each
        comparator and each share class."""
        return [FUND, *self.comparators, *(CLASS + name for name in self.classes)]

    def book(self,
             name: str) -> PerformanceAnalytics:
        """The performance of book *name*: the fund, a comparator, or a
        share class (``class/<name>``)."""
        if name == FUND:
            return self.performance

        return self._books[name] if name in self._books else self._book(name)

    @cached_property
    def _books(self) -> dict[str, PerformanceAnalytics]:
        return {}

    def _book(self,
              name: str) -> PerformanceAnalytics:
        if name.startswith(CLASS):
            returns = self._class_returns(name[len(CLASS):])
        else:
            levels = self.run.comparators[self.require(name)]
            returns = aligned(self.run.returns, levels)[1]

        run = RunData.of_returns(returns, self.run.periods_per_year, self.run.risk_free_rate,
                                 currency=self.run.currency)
        found = PerformanceAnalytics(run, replace(self.options.window, tracked=None))
        self._books[name] = found

        return found

    def _class_returns(self,
                       name: str) -> pd.Series:
        """A share class's daily returns: its NAV per unit, with what it
        paid out added back."""
        if name not in self.classes:
            raise Unavailable(NO_SHARE_CLASSES)

        days = self.run.tables[CLASS_DAYS]
        own = days[days["CLASS"].astype(str) == name]
        nav = own["NAV_PER_UNIT"].astype(float)
        paid = (own["DISTRIBUTIONS"].astype(float) / own["UNITS"].astype(float)).fillna(0.0)
        returns = ((nav + paid) / nav.shift(1) - 1.0).fillna(0.0)
        returns.index = pd.DatetimeIndex(returns.index)

        return returns

    def relative(self,
                 comparator: str | None = None) -> RelativeAnalytics:
        """The fund against *comparator*: the tracked one when None."""
        name = self.require(comparator) if comparator is not None else self.require_tracked()

        if name not in self._relatives:
            self._relatives[name] = RelativeAnalytics(self.run, name, self.options.window)

        return self._relatives[name]

    @cached_property
    def _relatives(self) -> dict[str, RelativeAnalytics]:
        return {}

    def market(self) -> str:
        """The comparator crises and regimes are read from."""
        chosen = self.options.crises.market or (BENCHMARK if BENCHMARK in self.comparators
                                                else INDEX)

        return self.require(chosen)

    @cached_property
    def crises(self) -> CrisisAnalytics:
        """The run's crises, against the tracked comparator."""
        return CrisisAnalytics(self.run, self.require_tracked(), self.market(),
                               threshold=self.options.crises.threshold,
                               settings=self.options.window)

    @cached_property
    def regimes(self) -> RegimeAnalytics:
        """The run's regimes, against the tracked comparator."""
        return RegimeAnalytics(self.run, self.require_tracked(), self.market(),
                               settings=self.options.window)

    # -- portfolio ---------------------------------------------------------

    @cached_property
    def portfolio(self) -> PortfolioAnalytics:
        """The run's portfolio figures."""
        return PortfolioAnalytics(self.run)

    # -- risk --------------------------------------------------------------

    def require_source(self) -> DataFetcher:
        """The data store, or `Unavailable` when it is not loaded."""
        if self.source is None:
            raise Unavailable(NO_DATA_STORE)

        return self.source

    @cached_property
    def factor_analytics(self) -> FactorAnalytics:
        """The run's factor exposures, returns and betas."""
        return FactorAnalytics(self.run, self.require_source(), self.factors)

    @cached_property
    def loss(self) -> RiskOfLoss:
        """The run's value at risk, expected shortfall and stress windows."""
        return RiskOfLoss(self.run, self.options.window)

    @cached_property
    def risk_over_time(self) -> RiskOverTime:
        """The run's expected and measured risk."""
        return RiskOverTime(self.factor_analytics, window=self.options.risk_model.window)

    @cached_property
    def concentration(self) -> ConcentrationAnalytics:
        """The run's concentration; by risk only when the data store is
        loaded."""
        return ConcentrationAnalytics(self.run, self.risk_over_time if self.source is not None
                                      else None)

    @cached_property
    def liquidity(self) -> LiquidityAnalytics:
        """How fast the run's books could be sold."""
        options = self.options.liquidity

        return LiquidityAnalytics(self.run, self.require_source(),
                                  participation=options.participation,
                                  lookback_days=options.lookback_days)

    # -- attribution -------------------------------------------------------

    @cached_property
    def attribution(self) -> Attribution:
        """The run's attribution."""
        return Attribution(self.run)

    def window(self) -> tuple[Any, Any]:
        """The analysis window's start and end, None where open."""
        return self.options.window.start, self.options.window.end
