# src/beacon/analysis/risk/factors.py
"""
A run's factors: each name's exposures at each rebalance, the factors' daily
returns, and the betas of the fund, its comparators and its active return to
them.

**Exposures** are each factor's scores (see `beacon.strategy.Factor`) at
each rebalance, for every name the fund, its index or its benchmark held,
standardised over the universe: the index's names that day by default, or
names given.

**Factor returns** are estimated each day by a cross-sectional regression of
the universe's returns on their exposures at the last rebalance before the
day, with a market intercept, each name weighted by its market cap:

    r_i = f_market + sum_k z_ik f_k + e_i

so a factor's return is what a unit of exposure earned that day with the
others held fixed. A factor naming its own series (``returns``) takes that
series' daily change instead.

**Betas** regress a book's daily return over the risk-free rate on the
factor returns, with an intercept, by ordinary least squares; the active
return (the fund's less its index's) is regressed on the same factors.

**Holdings-based exposures** are a book's weights times its names'
exposures: at each rebalance, from its weights at the close that day, or
each day, from that day's weights and the last rebalance's exposures. Cash
and names without exposures add nothing. A book's active exposure is its
exposure less its index's, or another comparator's.
"""
# BN-390 added the holdings-based exposures.
# BN-387. Agreed with Karan on 2026-10-27.
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from ...backtest.histories import history
from ...data.base import as_of_position
from ...data.fetcher import DataFetcher
from ...exceptions import CalculationError
from ...risk.model import MARKET_FACTOR
from ...strategy.base import StrategyContext
from ...strategy.factors import Factor, FactorInputs, default_factors
from ..data import INDEX, RunData
from ..panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from ..performance import formulas
from ..performance.comparisons import aligned
from ..performance.portfolio import PortfolioAnalytics
from .model import RebalanceRiskModels
from .read_once import PricesReadOnce

logger = logging.getLogger(__name__)

# The longest trailing window a factor reads at a rebalance: momentum's year.
LONGEST_LOOKBACK = 252

FUND = "fund"
ACTIVE = "active"
ALPHA = "alpha"

# The fewest days a regression is fitted on.
MINIMUM_DAYS = 20


@dataclass(frozen=True)
class FactorRegression:
    """A book's returns regressed on the factors.

    Attributes:
        coefficients: For the intercept (``alpha``, a day) and each factor,
            its coefficient, standard error and t-statistic.
        r_squared: The share of the returns' variance the factors explain.
        observations: The days fitted on.
        periods_per_year: The days in the run's year.
    """
    coefficients: pd.DataFrame
    r_squared: float
    observations: int
    periods_per_year: int

    @property
    def betas(self) -> pd.Series:
        """Each factor's beta."""
        return self.coefficients["coefficient"].drop(ALPHA)

    @property
    def alpha_a_year(self) -> float:
        """The intercept over a year: what the book earned beyond its
        exposures, a day times the days in a year."""
        return float(self.coefficients.loc[ALPHA, "coefficient"] * self.periods_per_year)


class FactorAnalytics:
    """A run's factor exposures, factor returns and betas.

    Args:
        run: The run's data, with its name returns, weights and reference.
        source: The data the run read, which the factors are scored from.
        factors: The factors; `beacon.strategy.default_factors` when None.
        universe: The names exposures are standardised over; the index's
            names at each rebalance when None.

    Raises:
        ValueError: If two factors share a name, or one is called market.
    """

    def __init__(self,
                 run: RunData,
                 source: DataFetcher,
                 factors: Sequence[Factor] | None = None,
                 universe: Sequence[str] | None = None):
        self.run = run
        self.source = source
        self.factors = list(factors) if factors is not None else default_factors()
        names = [factor.name for factor in self.factors]

        if len(set(names)) != len(names) or MARKET_FACTOR in names:
            raise ValueError(f"Factors need distinct names other than {MARKET_FACTOR!r}, "
                             f"got {names}.")

        self.universe = list(universe) if universe is not None else None
        self.context = StrategyContext(self._prices_once(source), run.currency)
        self._models = RebalanceRiskModels(run)
        self._exposures: dict[pd.Timestamp, pd.DataFrame] = {}
        self._caps: dict[pd.Timestamp, pd.Series] = {}
        self._fallbacks: dict[pd.Timestamp, dict[str, dict[tuple[str, ...], tuple[str, ...]]]] = {}
        self._returns: pd.DataFrame | None = None

    def _prices_once(self,
                     source: DataFetcher) -> DataFetcher:
        """*source*, with every name's closes over the run (and the year and
        a half before it a trailing return reads) read once, rather than at
        each rebalance (BN-397: 11 s of 19 at 6,000 names)."""
        dates = self.dates

        if not dates:
            return source

        names = sorted({str(name) for book in (FUND_WEIGHTS, INDEX_WEIGHTS)
                        if (table := self.run.tables.get(book)) is not None
                        for name in table.columns})
        start = dates[0] - pd.Timedelta(days=int(LONGEST_LOOKBACK * 1.6) + 10)

        return cast(DataFetcher, PricesReadOnce(source, names, start, dates[-1],
                                                self.run.currency, dates))

    @property
    def names(self) -> list[str]:
        """The factors' names, in order."""
        return [factor.name for factor in self.factors]

    @property
    def dates(self) -> list[pd.Timestamp]:
        """The run's rebalances."""
        reference = self.run.tables[REFERENCE]

        return sorted(pd.Timestamp(day) for day in reference.index.get_level_values(0).unique())

    def exposures(self,
                  date: str | pd.Timestamp) -> pd.DataFrame:
        """Each held name's exposure to each factor at the rebalance on
        *date*: names on the index, factors on the columns."""
        day = pd.Timestamp(date)

        if day not in self._exposures:
            if day not in self.dates:
                raise KeyError(f"{day.date()} is not one of the run's rebalances.")

            self._exposures[day] = self._scored(day)

        return self._exposures[day]

    def fallbacks(self,
                  date: str | pd.Timestamp) -> dict[str, dict[tuple[str, ...], tuple[str, ...]]]:
        """For each factor centred within groups, the groups at the rebalance
        on *date* too small to centre on, and the wider group used."""
        day = pd.Timestamp(date)
        self.exposures(day)

        return self._fallbacks[day]

    def universe_on(self,
                    date: pd.Timestamp) -> list[str]:
        """The names standardised over at the rebalance on *date*."""
        if self.universe is not None:
            return self.universe

        weights = self.run.tables.get(INDEX_WEIGHTS)
        at = None if weights is None else as_of_position(weights.index, date)

        if weights is None or at is None:
            return self._models.held(date)

        row = weights.iloc[at]

        return sorted(str(name) for name in row.index[row.fillna(0.0) > 0.0])

    def factor_returns(self) -> pd.DataFrame:
        """Each day's factor returns, the market's first, from the day after
        the first rebalance."""
        if self._returns is None:
            self._returns = self._imported(self._estimated())

        return self._returns

    def regression(self,
                   book: str = FUND) -> FactorRegression:
        """The book's returns (``fund``, a comparator's name such as
        ``index``, or ``active``) regressed on the factor returns over the
        whole run."""
        target, factors = self._paired(book)

        return regressed(target, factors, self.run.periods_per_year)

    def rolling_betas(self,
                      book: str = FUND,
                      window: int = 252) -> pd.DataFrame:
        """Each factor's beta over the *window* days to each day."""
        target, factors = self._paired(book)
        rows = {}

        for end in range(window, len(target) + 1):
            fit = regressed(target.iloc[end - window:end], factors.iloc[end - window:end],
                             self.run.periods_per_year)
            rows[target.index[end - 1]] = fit.betas

        return pd.DataFrame.from_dict(rows, orient="index", columns=list(factors.columns))

    def holdings_exposures(self,
                           book: str = FUND,
                           daily: bool = False) -> pd.DataFrame:
        """A book's exposure to each factor (``fund``, ``index`` or
        ``benchmark``): at each rebalance, or each day when *daily*.

        Raises:
            KeyError: If the run kept no weights for the book.
        """
        weights = PortfolioAnalytics(self.run).weights(book)

        if weights is None or weights.empty:
            raise KeyError(f"The run kept no weights for its {book}.")

        weights = weights.sort_index().fillna(0.0)
        rebalances = pd.DatetimeIndex(self.dates)
        days = weights.index if daily else rebalances
        rows = {}

        for day in days:
            on = as_of_position(weights.index, day)
            at = as_of_position(rebalances, day)

            if on is None or at is None:
                continue

            exposures = self.exposures(rebalances[at])
            held = weights.iloc[on]
            held.index = held.index.astype(str)
            rows[pd.Timestamp(day)] = exposures.T @ held.reindex(exposures.index).fillna(0.0)

        return pd.DataFrame.from_dict(rows, orient="index", columns=self.names)

    def active_exposures(self,
                         against: str = INDEX,
                         daily: bool = False) -> pd.DataFrame:
        """The fund's exposures less a comparator's, on the days both have."""
        ours = self.holdings_exposures(FUND, daily)
        theirs = self.holdings_exposures(against, daily)
        days = ours.index.intersection(theirs.index)

        return ours.loc[days] - theirs.loc[days]

    def _scored(self,
                day: pd.Timestamp) -> pd.DataFrame:
        """Every held name's exposures on *day*, and the caps they used."""
        universe = self.universe_on(day)
        held = sorted(set(self._models.held(day)) | set(universe))
        inputs = FactorInputs(self.context, day, held)
        table = pd.DataFrame({factor.name: factor.exposures(held, day, self.context,
                                                            universe=universe, inputs=inputs)
                              for factor in self.factors}, index=held)
        self._caps[day] = pd.Series(inputs.caps()).reindex(held).fillna(0.0)
        self._fallbacks[day] = inputs.fallbacks
        self._warn_if_unpriced(day)

        return table[self.names]

    def _warn_if_unpriced(self,
                          day: pd.Timestamp) -> None:
        """Say when names the fund held on *day* have no market cap in the
        data given: a run read from other data than its own, most likely."""
        weights = self.run.tables.get(FUND_WEIGHTS)
        at = None if weights is None else as_of_position(weights.index, day)

        if weights is None or at is None:
            return

        row = weights.iloc[at].fillna(0.0)
        row.index = row.index.astype(str)
        held = row[row > 0.0]
        unpriced = held[self._caps[day].reindex(held.index).fillna(0.0) <= 0.0]

        if len(unpriced):
            logger.warning("[%s] %d names the fund held (%.1f%% of it) have no market cap in "
                           "the data given; is it the data the run read?", day.date(),
                           len(unpriced), 100.0 * float(unpriced.sum()))

    def _estimated(self) -> pd.DataFrame:
        """The factor returns from the daily cross-sectional regressions,
        worked out a stretch between rebalances at a time: on each stretch
        the names, exposures and caps are the same, so every day with the
        same names priced is one fit."""
        # BN-411: a fit and several reindexes a day took 6.1 s at 6,000 names.
        returns = self.run.tables[NAME_RETURNS].sort_index()
        dates = pd.DatetimeIndex(self.dates)
        stretches: dict[int, list[int]] = {}

        for row, day in enumerate(returns.index):
            at = as_of_position(dates, day - pd.Timedelta(days=1))

            if at is not None:
                stretches.setdefault(at, []).append(row)

        rows: dict[pd.Timestamp, list[float]] = {}

        for at, positions in stretches.items():
            rebalance = dates[at]
            exposures = self.exposures(rebalance)
            names = [name for name in self.universe_on(rebalance) if name in exposures.index]
            block = returns.iloc[positions].reindex(columns=names).to_numpy(dtype=float)
            caps = self._caps[rebalance].reindex(names).to_numpy(dtype=float)
            fitted = _cross_sections(block, exposures.loc[names].to_numpy(dtype=float), caps)

            rows.update({returns.index[positions[day]]: found for day, found in fitted.items()})

        found = pd.DataFrame.from_dict(dict(sorted(rows.items())), orient="index",
                                       columns=[MARKET_FACTOR, *self.names])

        return found

    def _imported(self,
                  estimated: pd.DataFrame) -> pd.DataFrame:
        """*estimated*, with each factor naming its own series taking that
        series' daily change."""
        for factor in self.factors:
            if factor.returns is None:
                continue

            levels = history(self.source, factor.returns, None).closes.sort_index()

            if levels.dropna().empty:
                raise CalculationError("Factor returns", f"{factor.name}'s series "
                                                         f"{factor.returns!r} has no levels")

            estimated[factor.name] = levels.pct_change().reindex(estimated.index)

        return estimated

    def _paired(self,
                book: str) -> tuple[pd.Series, pd.DataFrame]:
        """The book's daily returns to regress, over the risk-free rate but
        for the active return, and the factor returns, on the days both
        have."""
        factors = self.factor_returns().dropna()
        rate = formulas.daily_rate(self.run.risk_free_rate, self.run.periods_per_year)

        if book == FUND:
            target = self.run.returns - rate
        elif book == ACTIVE:
            ours, theirs = aligned(self.run.returns, self._levels(INDEX))
            target = ours - theirs
        else:
            _, theirs = aligned(self.run.returns, self._levels(book))
            target = theirs - rate

        days = target.index.intersection(factors.index)

        if len(days) < MINIMUM_DAYS:
            raise CalculationError("Factor betas", f"{len(days)} days of {book} and factor "
                                                   f"returns, fewer than {MINIMUM_DAYS}")

        return target.loc[days].astype(float), factors.loc[days]

    def _levels(self,
                book: str) -> pd.Series:
        """A comparator's levels."""
        if book not in self.run.comparators:
            raise KeyError(f"The run has no {book}: it has "
                           f"{', '.join(self.run.comparators) or 'no comparators'}.")

        return self.run.comparators[book]


def _cross_sections(returns: np.ndarray,
                    exposures: np.ndarray,
                    caps: np.ndarray) -> dict[int, list[float]]:
    """Each day's factor returns over one stretch: the cap-weighted
    least-squares fit of the names' returns (days by names) on a market
    intercept and their exposures, by day; a day with fewer names than
    unknowns is left out. Days with the same names priced share one fit."""
    known = ~np.isnan(returns) & (caps > 0.0)[None, :]
    patterns, groups = np.unique(known, axis=0, return_inverse=True)
    design = np.column_stack([np.ones(len(caps)), exposures])
    fitted: dict[int, list[float]] = {}

    for group, pattern in enumerate(patterns):
        days = np.flatnonzero(groups.ravel() == group)

        if pattern.sum() <= exposures.shape[1] + 1:
            continue

        weights = caps[pattern]
        root = np.sqrt(weights / weights.sum())
        solved, *_ = np.linalg.lstsq(design[pattern] * root[:, None],
                                     returns[np.ix_(days, np.flatnonzero(pattern))].T
                                     * root[:, None], rcond=None)
        fitted.update({int(day): [float(value) for value in solved[:, column]]
                       for column, day in enumerate(days)})

    return fitted


def regressed(target: pd.Series,
               factors: pd.DataFrame,
               periods_per_year: int) -> FactorRegression:
    """*target* regressed on *factors* with an intercept, by ordinary least
    squares."""
    design = np.column_stack([np.ones(len(target)), factors.to_numpy(dtype=float)])
    values = target.to_numpy(dtype=float)
    solved, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ solved
    freedom = max(len(values) - design.shape[1], 1)
    variance = float(residuals @ residuals) / freedom
    covariance = variance * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    spread = float(((values - values.mean()) ** 2).sum())
    names = [ALPHA, *factors.columns]
    table = pd.DataFrame({"coefficient": solved, "std_error": errors,
                          "t_stat": np.divide(solved, errors, out=np.zeros_like(solved),
                                              where=errors > 0.0)}, index=names)

    return FactorRegression(coefficients=table,
                            r_squared=1.0 - float(residuals @ residuals) / spread
                            if spread > 0.0 else 0.0,
                            observations=len(values), periods_per_year=periods_per_year)
