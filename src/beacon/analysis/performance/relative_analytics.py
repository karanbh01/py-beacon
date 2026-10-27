# src/beacon/analysis/performance/relative_analytics.py
"""
A run against one comparator, its index or its benchmark: how much of its
return the comparator explains and what is left (alpha, beta, R-squared, and
the t-statistics that say whether the rest is more than luck), how it
behaved when the comparator rose and fell, how often and by how much it got
ahead, and its stretches of falling behind.

    relative = RelativeAnalytics(RunData.from_record(folder), "benchmark")
    relative.figures()

The two are compared on the days both have, as the run's comparisons are:
the comparator's first day returns nothing from its level that day, and so,
when the comparator starts after the run, does the run's.
"""
# BN-377. The forms chosen with Karan on 2026-10-27: Jensen's alpha from the
# daily excess returns, capture on months as Morningstar measures it, the
# batting average on months, the downside tracking error over every day.
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data import INDEX, RunData
from . import formulas
from .analytics import PerformanceAnalytics
from .comparisons import relative_figures
from .drawdowns import WORST, DrawdownFigures, drawdown_figures
from .settings import AnalysisSettings


@dataclass(frozen=True)
class Regression:
    """The fund's daily excess returns regressed on the comparator's.

    Attributes:
        alpha: The intercept, a year: the return the comparator does not
            explain.
        beta: The slope: the fund's sensitivity to the comparator.
        r_squared: The share of the fund's variance the comparator explains.
        alpha_t: Alpha over its standard error.
        alpha_p: The two-sided probability of an alpha this far from 0 by
            chance.
        residual_risk: The spread of what the comparator does not explain, a
            year.
    """
    alpha: float
    beta: float
    r_squared: float
    alpha_t: float | None
    alpha_p: float | None
    residual_risk: float


@dataclass(frozen=True)
class YearDifference:
    """One calendar year's returns and their difference.

    Attributes:
        year: The year.
        fund: The fund's return in it.
        comparator: The comparator's.
        difference: The fund's less the comparator's, in percentage points.
    """
    year: str
    fund: float
    comparator: float
    difference: float


class RelativeAnalytics:
    """A run against one comparator.

    Args:
        run: The run's data.
        comparator: The comparator, by name (``index``, ``optimised`` or
            ``benchmark``) or by its levels.
        settings: What the figures are measured over and with; the run's own
            when None.

    Raises:
        KeyError: If the run has no comparator of that name.
    """

    def __init__(self,
                 run: RunData,
                 comparator: str | pd.Series = INDEX,
                 settings: AnalysisSettings | None = None):
        self.performance = PerformanceAnalytics(run, settings)
        self.levels = (run.comparators[comparator] if isinstance(comparator, str)
                       else comparator)
        self.ours, self.theirs = _aligned(self.performance.returns, self.levels)

    @property
    def periods_per_year(self) -> int:
        """How many daily returns make a year."""
        return self.performance.periods_per_year

    @property
    def active(self) -> pd.Series:
        """The fund's daily return less the comparator's."""
        return self.ours - self.theirs

    def regression(self) -> Regression | None:
        """Jensen's regression of the fund's daily excess returns on the
        comparator's; None with fewer than three days or a comparator that
        does not vary."""
        rate = formulas.daily_rate(self.performance.risk_free_rate, self.periods_per_year)
        y = (self.ours - rate).to_numpy(dtype=float)
        x = (self.theirs - rate).to_numpy(dtype=float)
        count = len(y)

        if count < 3 or not float(np.std(x, ddof=1)) > formulas.NEGLIGIBLE_SPREAD:
            return None

        beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
        alpha = float(y.mean() - beta * x.mean())
        residuals = y - alpha - beta * x
        squares = float(np.sum((y - y.mean()) ** 2))
        unexplained = float(np.sum(residuals ** 2))
        noise = unexplained / (count - 2)
        error = np.sqrt(noise * (1.0 / count + x.mean() ** 2 / float(np.sum((x - x.mean()) ** 2))))
        t = float(alpha / error) if error > 0 else None

        return Regression(alpha=alpha * self.periods_per_year,
                          beta=beta,
                          r_squared=1.0 - unexplained / squares if squares else 0.0,
                          alpha_t=t,
                          alpha_p=_two_sided_p(t, count - 2),
                          residual_risk=float(np.sqrt(noise * self.periods_per_year)))

    def treynor(self) -> float | None:
        """The annual mean excess return per unit of beta; None without a
        beta."""
        regression = self.regression()

        if regression is None or regression.beta == 0:
            return None

        rate = formulas.daily_rate(self.performance.risk_free_rate, self.periods_per_year)

        return float((self.ours - rate).mean() * self.periods_per_year / regression.beta)

    def information_ratio_t(self) -> dict[str, float | None]:
        """The information ratio's t-statistic, IR times the root of the
        years, and its two-sided probability by chance."""
        ratio = formulas.information_ratio(self.active, self.periods_per_year)

        if ratio is None:
            return {"t": None, "p": None}

        t = ratio * float(np.sqrt(len(self.active) / self.periods_per_year))

        return {"t": t, "p": _two_sided_p(t, len(self.active) - 1)}

    def capture(self) -> dict[str, float | None]:
        """Up capture, down capture and their ratio, on months: the fund's
        geometric mean monthly return in the months the comparator rose (fell)
        over the comparator's in them."""
        ours = formulas.period_returns(self.ours, formulas.MONTHLY)
        theirs = formulas.period_returns(self.theirs, formulas.MONTHLY)
        up = _capture(ours[theirs > 0], theirs[theirs > 0])
        down = _capture(ours[theirs < 0], theirs[theirs < 0])

        return {"up": up, "down": down,
                "ratio": up / down if up is not None and down else None}

    def batting_average(self) -> float | None:
        """The share of months the fund's return beat the comparator's."""
        return formulas.relative_hit_rate_monthly(self.ours, self.theirs)

    def relative_levels(self) -> pd.Series:
        """The fund's wealth over the comparator's, from 1 the business day
        before the first day compared."""
        if self.ours.empty:
            return self.ours

        wealth = (1.0 + self.ours).cumprod() / (1.0 + self.theirs).cumprod()
        eve = pd.Timestamp(wealth.index[0]) - pd.offsets.BDay(1)

        return pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([eve])), wealth])

    def relative_drawdowns(self,
                           worst: int = WORST) -> DrawdownFigures:
        """The drawdowns of the fund's wealth relative to the comparator's:
        each a stretch of falling behind."""
        levels = self.relative_levels()
        total = float(levels.iloc[-1] - 1.0) if len(levels) else 0.0

        return drawdown_figures(levels, total, worst)

    def downside_tracking_error(self) -> float:
        """The root mean square of the days the fund fell behind, every day
        counted, a year."""
        values = np.minimum(self.active.to_numpy(dtype=float), 0.0)

        return float(np.sqrt(np.mean(values ** 2) * self.periods_per_year)) if len(values) else 0.0

    def tracking_difference_by_year(self) -> tuple[list[YearDifference], float | None]:
        """Each calendar year's return less the comparator's, and the
        standard deviation of those differences (None with fewer than two
        years)."""
        ours = formulas.period_returns(self.ours, formulas.ANNUAL)
        theirs = formulas.period_returns(self.theirs, formulas.ANNUAL)
        years = [YearDifference(year=str(year), fund=float(ours[year]),
                                comparator=float(theirs[year]),
                                difference=float(ours[year] - theirs[year]))
                 for year in ours.index]
        differences = pd.Series([year.difference for year in years], dtype=float)

        return years, (float(differences.std()) if len(differences) > 1 else None)

    def figures(self) -> dict[str, Any]:
        """Every relative figure, by name: today's comparison figures
        (relative return, tracking error, information ratio, correlation,
        beta, monthly hit rate) and the rest."""
        regression = self.regression()
        years, spread = self.tracking_difference_by_year()

        return {**relative_figures(self.ours, self.theirs, self.periods_per_year),
                "alpha": regression.alpha if regression else None,
                "r_squared": regression.r_squared if regression else None,
                "alpha_t": regression.alpha_t if regression else None,
                "alpha_p": regression.alpha_p if regression else None,
                "residual_risk": regression.residual_risk if regression else None,
                "information_ratio_t": self.information_ratio_t(),
                "treynor": self.treynor(),
                "capture": self.capture(),
                "batting_average": self.batting_average(),
                "downside_tracking_error": self.downside_tracking_error(),
                "relative_drawdowns": self.relative_drawdowns(),
                "tracking_difference_by_year": years,
                "tracking_difference_spread": spread}


def _aligned(returns: pd.Series,
             levels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The fund's and the comparator's daily returns on the days both have,
    as the run's comparisons align them."""
    levels = levels.dropna()
    window = returns.index.intersection(levels.index)

    if len(window) < 2:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        return empty, empty

    ours = returns.loc[window].astype(float)
    theirs = formulas.returns_from_start(levels.loc[window])

    if window[0] != returns.index[0]:
        ours = ours.copy()
        ours.iloc[0] = 0.0

    return ours, theirs


def _capture(ours: pd.Series,
             theirs: pd.Series) -> float | None:
    """The fund's geometric mean return over the comparator's, over the same
    months; None without any, or when the comparator's is 0."""
    if ours.empty:
        return None

    mean_ours = float((1.0 + ours).prod() ** (1.0 / len(ours)) - 1.0)
    mean_theirs = float((1.0 + theirs).prod() ** (1.0 / len(theirs)) - 1.0)

    return mean_ours / mean_theirs if mean_theirs != 0 else None


def _two_sided_p(t: float | None,
                 freedom: int) -> float | None:
    """The probability of a t-statistic at least this far from 0 by chance."""
    if t is None or freedom < 1:
        return None

    # scipy only when asked: it stays off `import beacon`'s path.
    from scipy import stats  # noqa: PLC0415

    return float(2.0 * stats.t.sf(abs(t), freedom))
