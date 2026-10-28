# src/beacon/analysis/risk/var.py
"""
Value at risk and expected shortfall, and how each book fared in the run's
stress windows.

Both are losses, reported as positive numbers: a one-day 99% value at risk
of 0.02 says a day's loss is worse than 2% one day in a hundred.

- **Historical**: from the returns themselves. Value at risk is minus their
  ``1 - level`` quantile (linearly interpolated); expected shortfall is minus
  the mean of the returns at or below that quantile.
- **Normal**: from their mean and standard deviation, as if normal.
  ``VaR = -(mean + z sd)`` with ``z`` the normal's ``1 - level`` quantile,
  and ``ES = -mean + sd phi(z) / (1 - level)``.
- **Cornish-Fisher**: the normal's, with ``z`` moved by the returns' skew S
  and excess kurtosis K, ``z + (z^2 - 1) S / 6 + (z^3 - 3z) K / 24 -
  (2z^3 - 5z) S^2 / 36``, so fat tails and a lean to one side count. The
  expansion holds for moderate tails (a skew within about 1 either side and
  an excess kurtosis up to about 6); far beyond them it can fall below even
  the normal's figure, and the historical one is the better guide.

Ten days is the square root of ten times one day, the Basel convention.

**Stress windows** are the run's crises (`beacon.analysis.performance.crises`:
the market's falls of 15% or more, or named episodes), and each book's return
and deepest fall over each.
"""
# BN-388. Agreed with Karan on 2026-10-27.
import math
from collections.abc import Sequence
from statistics import NormalDist

import pandas as pd

from ..data import RunData
from ..performance import formulas
from ..performance.analytics import PerformanceAnalytics
from ..performance.comparisons import aligned
from ..performance.crises import THRESHOLD, CrisisAnalytics, Episode
from ..performance.distribution import excess_kurtosis, skew
from ..performance.settings import AnalysisSettings

HISTORICAL = "historical"
NORMAL = "normal"
CORNISH_FISHER = "cornish_fisher"
VAR_METHODS = (HISTORICAL, NORMAL, CORNISH_FISHER)
ES_METHODS = (HISTORICAL, NORMAL)

LEVELS = (0.95, 0.99)
HORIZONS = (1, 10)

FUND = "fund"
ACTIVE = "active"

# The fewest returns a figure is measured on.
MINIMUM_RETURNS = 20

STANDARD = NormalDist()


def value_at_risk(returns: pd.Series,
                  level: float = 0.95,
                  method: str = HISTORICAL,
                  horizon: int = 1) -> float | None:
    """The loss worse than which a *horizon*-day period falls with
    probability ``1 - level``; None with fewer than 20 returns.

    Raises:
        ValueError: If the method is unknown or the level is not between 0
            and 1.
    """
    _check(method, VAR_METHODS, level)
    values = returns.dropna().astype(float)

    if len(values) < MINIMUM_RETURNS:
        return None

    if method == HISTORICAL:
        one_day = -float(values.quantile(1.0 - level))
    else:
        z = STANDARD.inv_cdf(1.0 - level)
        z = _cornish_fisher(z, values) if method == CORNISH_FISHER else z
        one_day = -(float(values.mean()) + z * float(values.std()))

    return one_day * math.sqrt(horizon)


def expected_shortfall(returns: pd.Series,
                       level: float = 0.95,
                       method: str = HISTORICAL,
                       horizon: int = 1) -> float | None:
    """The mean loss on the periods worse than the value at risk; None with
    fewer than 20 returns.

    Raises:
        ValueError: If the method is unknown or the level is not between 0
            and 1.
    """
    _check(method, ES_METHODS, level)
    values = returns.dropna().astype(float)

    if len(values) < MINIMUM_RETURNS:
        return None

    if method == HISTORICAL:
        cut = float(values.quantile(1.0 - level))
        one_day = -float(values[values <= cut].mean())
    else:
        z = STANDARD.inv_cdf(1.0 - level)
        one_day = -float(values.mean()) + float(values.std()) * STANDARD.pdf(z) / (1.0 - level)

    return one_day * math.sqrt(horizon)


def risk_of_loss(returns: pd.Series,
                 levels: Sequence[float] = LEVELS,
                 horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    """Every value at risk and expected shortfall of *returns*: a row a
    measure and method, a column a level and horizon."""
    rows = {}

    for name, figure, methods in (("var", value_at_risk, VAR_METHODS),
                                  ("es", expected_shortfall, ES_METHODS)):
        for method in methods:
            rows[(name, method)] = {(level, horizon): figure(returns, level, method, horizon)
                                    for level in levels for horizon in horizons}

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["measure", "method"])
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["level", "days"])

    return table


class RiskOfLoss:
    """A run's value at risk, expected shortfall and stress windows.

    Args:
        run: The run's data.
        settings: The window and conventions figures are measured with.
    """

    def __init__(self,
                 run: RunData,
                 settings: AnalysisSettings | None = None):
        self.run = run
        self.settings = settings
        self.returns = PerformanceAnalytics(run, settings).returns

    def books(self) -> list[str]:
        """The books the run can be measured on: the fund, each comparator,
        and the active return when it has an index."""
        found = [FUND, *self.run.comparators]

        return [*found, ACTIVE] if "index" in self.run.comparators else found

    def returns_of(self,
                   book: str = FUND) -> pd.Series:
        """A book's daily returns over the window: ``fund``, a comparator by
        name, or ``active`` (the fund's less its index's).

        Raises:
            KeyError: If the run has no such book.
        """
        if book == FUND:
            return self.returns

        name = "index" if book == ACTIVE else book

        if name not in self.run.comparators:
            raise KeyError(f"The run has no {book}. Known: {', '.join(self.books())}.")

        ours, theirs = aligned(self.returns, self.run.comparators[name])

        return ours - theirs if book == ACTIVE else theirs

    def table(self,
              book: str = FUND,
              levels: Sequence[float] = LEVELS,
              horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
        """The book's value at risk and expected shortfall, every method,
        level and horizon."""
        return risk_of_loss(self.returns_of(book), levels, horizons)

    def stress_windows(self,
                       market: str | pd.Series | None = None,
                       threshold: float = THRESHOLD,
                       named: Sequence[Episode] | None = None) -> pd.DataFrame:
        """Each crisis the run saw: its window, and each book's return and
        deepest fall over it, a row a crisis."""
        crises = CrisisAnalytics(self.run, market=market, threshold=threshold, named=named,
                                 settings=self.settings)
        books = {book: self.returns_of(book) for book in self.books() if book != ACTIVE}
        rows = []

        for crisis in crises.crises():
            row: dict[str, object] = {"crisis": crisis.name, "start": crisis.start,
                                      "end": crisis.end, "partial": crisis.partial}

            for book, returns in books.items():
                inside = returns[(returns.index > crisis.start) & (returns.index <= crisis.end)]
                path = formulas.path_of(inside)
                row[f"{book}_return"] = float(path.iloc[-1] - 1.0) if len(inside) else None
                row[f"{book}_drawdown"] = formulas.max_drawdown(path) if len(inside) else None

            rows.append(row)

        return pd.DataFrame(rows)


def _cornish_fisher(z: float,
                    values: pd.Series) -> float:
    """*z* moved by the returns' skew and excess kurtosis."""
    leaning = skew(values) or 0.0
    kurtosis = excess_kurtosis(values) or 0.0

    return (z + (z * z - 1.0) * leaning / 6.0 + (z ** 3 - 3.0 * z) * kurtosis / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * leaning * leaning / 36.0)


def _check(method: str,
           methods: tuple[str, ...],
           level: float) -> None:
    """Refuse an unknown method or a level outside 0 to 1."""
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}. Known: {', '.join(methods)}.")

    if not 0.0 < level < 1.0:
        raise ValueError(f"level is between 0 and 1, such as 0.95, got {level!r}.")
