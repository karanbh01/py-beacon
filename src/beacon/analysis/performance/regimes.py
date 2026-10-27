# src/beacon/analysis/performance/regimes.py
"""
How a run did in each kind of market: months the market rose and fell, calm
and volatile stretches, and bear markets and corrections.

    regimes = RegimeAnalytics(RunData.from_record(folder), market=sp500_levels)
    regimes.regimes()["volatile"]

Each regime is a set of days, found on a broad market series as the crises
are; over its days the fund is measured against its comparator. A regime's
return compounds its days, as if invested on those days alone, and is
annualised from their mean so regimes of different lengths compare.
"""
# BN-381. The crises' machinery (BN-380) over other regimes. Corrections are
# falls of 10% to 20%, so a bear market is not also a correction.
from dataclasses import dataclass

import pandas as pd

from ..data import BENCHMARK, INDEX, RunData
from . import formulas
from .analytics import PerformanceAnalytics
from .comparisons import aligned
from .drawdowns import episodes
from .settings import AnalysisSettings

# The market's falls that make a bear market and a correction.
BEAR, CORRECTION = 0.20, 0.10

# The rolling volatility calm and volatile days are told apart by: its days,
# and the share of days at each end.
VOLATILITY_WINDOW, FIFTH = 63, 0.2

REGIMES = ("up months", "down months", "calm", "volatile", "bear markets", "corrections")


@dataclass(frozen=True)
class Regime:
    """The fund against its comparator over one regime's days.

    Attributes:
        days: The regime's days the run had.
        fund: The fund's return compounded over them.
        comparator: The comparator's.
        difference: The fund's less the comparator's.
        fund_annual: The fund's mean daily return, a year.
        comparator_annual: The comparator's.
        beta: The fund's beta to the comparator over them.
        alpha: The fund's mean daily return beyond its beta, in excess of the
            risk-free rate, a year (Jensen's alpha over the regime).
    """
    days: int
    fund: float
    comparator: float
    difference: float
    fund_annual: float
    comparator_annual: float
    beta: float
    alpha: float


def up_and_down_months(market: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Which days fall in a month the market rose, and in one it fell,
    from its daily returns."""
    months = pd.DatetimeIndex(market.index).strftime("%Y-%m")
    monthly = formulas.period_returns(market, formulas.MONTHLY)
    by_day = pd.Series(monthly.reindex(months).to_numpy(), index=market.index)

    return by_day > 0, by_day < 0


def calm_and_volatile(market: pd.Series,
                      window: int = VOLATILITY_WINDOW,
                      fifth: float = FIFTH) -> tuple[pd.Series, pd.Series]:
    """Which days the market's rolling volatility was in its lowest fifth,
    and in its highest; days before the first full window are neither."""
    volatility = market.rolling(window).std()
    measured = volatility.dropna()

    if measured.empty:
        none = pd.Series(False, index=market.index)
        return none, none

    low, high = measured.quantile(fifth), measured.quantile(1.0 - fifth)

    return (volatility <= low).fillna(False), (volatility >= high).fillna(False)


def falls(levels: pd.Series,
          deeper_than: float,
          no_deeper_than: float | None = None) -> pd.Series:
    """Which days fall in one of the market's declines, peak to trough, of
    more than *deeper_than* (and at most *no_deeper_than*)."""
    inside = pd.Series(False, index=levels.index)

    for drawdown in episodes(levels):
        size = -drawdown.depth

        if size > deeper_than and (no_deeper_than is None or size <= no_deeper_than):
            inside |= (levels.index > drawdown.peak) & (levels.index <= drawdown.trough)

    return inside


def regime_of(ours: pd.Series,
              theirs: pd.Series,
              days: pd.Series,
              daily_rate: float,
              periods_per_year: int) -> Regime | None:
    """The fund against its comparator over *days*; None without any."""
    chosen = days.reindex(ours.index, fill_value=False).to_numpy(dtype=bool)

    if not chosen.any():
        return None

    fund, comparator = ours[chosen], theirs[chosen]
    total, theirs_total = formulas.compounded_return(fund), formulas.compounded_return(comparator)
    beta = formulas.beta(fund, comparator)
    excess, theirs_excess = fund - daily_rate, comparator - daily_rate

    return Regime(days=int(chosen.sum()), fund=total, comparator=theirs_total,
                  difference=total - theirs_total,
                  fund_annual=float(fund.mean() * periods_per_year),
                  comparator_annual=float(comparator.mean() * periods_per_year),
                  beta=beta,
                  alpha=float((excess.mean() - beta * theirs_excess.mean()) * periods_per_year))


class RegimeAnalytics:
    """How a run did in each kind of market, against one comparator.

    Args:
        run: The run's data.
        comparator: What the fund is measured against, by name or levels;
            the run's index by default.
        market: The broad market series the regimes are found on, by name of
            one of the run's comparators or by its levels; the run's
            benchmark when it has one, else its index.
        settings: What the figures are measured over and with.

    Raises:
        KeyError: If the run has no comparator of a name given.
    """

    def __init__(self,
                 run: RunData,
                 comparator: str | pd.Series = INDEX,
                 market: str | pd.Series | None = None,
                 settings: AnalysisSettings | None = None):
        performance = PerformanceAnalytics(run, settings)
        self.periods_per_year = performance.periods_per_year
        self.daily_rate = formulas.daily_rate(performance.risk_free_rate, self.periods_per_year)
        levels = run.comparators[comparator] if isinstance(comparator, str) else comparator
        self.ours, self.theirs = aligned(performance.returns, levels)
        chosen = market if market is not None else (BENCHMARK if BENCHMARK in run.comparators
                                                   else INDEX)
        market_levels = run.comparators[chosen] if isinstance(chosen, str) else chosen
        _, self.market = aligned(performance.returns, market_levels)
        self.market_levels = (1.0 + self.market).cumprod()

    def days(self) -> dict[str, pd.Series]:
        """Which days fall in each regime, by name."""
        up, down = up_and_down_months(self.market)
        calm, volatile = calm_and_volatile(self.market)

        return {"up months": up, "down months": down, "calm": calm, "volatile": volatile,
                "bear markets": falls(self.market_levels, BEAR),
                "corrections": falls(self.market_levels, CORRECTION, BEAR)}

    def regimes(self) -> dict[str, Regime | None]:
        """The fund against its comparator in each regime, by name; None for
        a regime the run never saw."""
        return {name: regime_of(self.ours, self.theirs, days, self.daily_rate,
                                self.periods_per_year)
                for name, days in self.days().items()}
