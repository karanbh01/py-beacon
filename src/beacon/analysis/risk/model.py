# src/beacon/analysis/risk/model.py
"""
The risk model an analysis of a run uses: which kind, estimated over how long,
and estimated once at each rebalance for every figure that needs it.

**Kinds.** ``sample`` is the sample covariance of the names' daily returns,
shrunk toward a structured target (`beacon.risk.estimate_risk_model`).
``factor`` is the factor model fitted by cross-sectional regression on the
names' exposures (`beacon.risk.fit_factor_model`), which splits risk into
factor and specific parts; its exposures at each rebalance are given by the
caller.

**Each rebalance's model** is estimated on the run's daily name returns over
the *window* sessions before that day, never including it, for every name
the fund, its index or its benchmark held on it. A name whose returns do not
cover the whole window (one that listed during it, say) is left out of that
rebalance's model rather than shortening the window for every name, and is
reported by `uncovered`.
"""
# BN-386.
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ...data.base import as_of_position
from ...exceptions import CalculationError
from ...risk.model import (
    CONSTANT_CORRELATION,
    TARGETS,
    FactorRiskModel,
    RiskModel,
    estimate_risk_model,
    fit_factor_model,
)
from ..data import RunData
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE

SAMPLE = "sample"
FACTOR = "factor"
KINDS = (SAMPLE, FACTOR)

# A year of daily returns.
DEFAULT_WINDOW = 252

# The fewest sessions a model can be estimated over.
MINIMUM_WINDOW = 20

# Each rebalance's exposures, for the names asked about: names on the index,
# factors on the columns.
Exposures = Callable[[pd.Timestamp, list[str]], pd.DataFrame]


@dataclass(frozen=True)
class RiskModelSettings:
    """How an analysis estimates its risk model.

    Attributes:
        kind: ``sample`` or ``factor``.
        window: The sessions of daily returns before each rebalance it is
            estimated on.
        target: For a sample model, the structured target it is shrunk
            toward; see `beacon.risk.TARGETS`.
        intensity: For a sample model, the weight on the target, from 0 to
            1; None takes the heuristic from the panel's shape.

    Raises:
        ValueError: If the kind or target is unknown, the window is shorter
            than 20 sessions, or the intensity is outside 0 to 1.
    """
    kind: str = SAMPLE
    window: int = DEFAULT_WINDOW
    target: str = CONSTANT_CORRELATION
    intensity: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown risk model {self.kind!r}. Known: {', '.join(KINDS)}.")

        if self.window != int(self.window) or self.window < MINIMUM_WINDOW:
            raise ValueError(f"window is a whole number of sessions, at least "
                             f"{MINIMUM_WINDOW}, got {self.window!r}.")

        object.__setattr__(self, "window", int(self.window))

        if self.target not in TARGETS:
            raise ValueError(f"Unknown target {self.target!r}. Known: {', '.join(TARGETS)}.")

        if self.intensity is not None and not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity is from 0 to 1, got {self.intensity!r}.")


class RebalanceRiskModels:
    """A run's risk model at each rebalance, each estimated once when first
    asked for.

    Args:
        run: The run's data, with its name returns.
        settings: How the models are estimated.
        exposures: For a factor model, each rebalance's exposures of the
            names asked about.

    Raises:
        ValueError: If a factor model is asked for without exposures.
    """

    def __init__(self,
                 run: RunData,
                 settings: RiskModelSettings | None = None,
                 exposures: Exposures | None = None):
        self.run = run
        self.settings = settings or RiskModelSettings()

        if self.settings.kind == FACTOR and exposures is None:
            raise ValueError("A factor model needs each rebalance's exposures.")

        self.exposures = exposures
        self._models: dict[pd.Timestamp, RiskModel | FactorRiskModel] = {}
        self._uncovered: dict[pd.Timestamp, list[str]] = {}
        self._sorted: pd.DataFrame | None = None

    @property
    def dates(self) -> list[pd.Timestamp]:
        """The run's rebalances, in order."""
        reference = self.run.tables[REFERENCE]

        return sorted(pd.Timestamp(day) for day in reference.index.get_level_values(0).unique())

    def at(self,
           date: str | pd.Timestamp) -> RiskModel | FactorRiskModel:
        """The model estimated for the rebalance on *date*.

        Raises:
            KeyError: If *date* is not a rebalance.
            CalculationError: If fewer than the window's sessions come before
                it, or no name's returns cover them.
        """
        day = pd.Timestamp(date)

        if day not in self._models:
            if day not in self.dates:
                raise KeyError(f"{day.date()} is not one of the run's rebalances.")

            self._models[day] = self._estimate(day)

        return self._models[day]

    def uncovered(self,
                  date: str | pd.Timestamp) -> list[str]:
        """The names held on the rebalance on *date* whose returns did not
        cover the window, and so are not in its model."""
        day = pd.Timestamp(date)
        self.at(day)

        return self._uncovered[day]

    def held(self,
             date: pd.Timestamp) -> list[str]:
        """Every name the fund, its index or its benchmark held on *date*."""
        names: set[str] = set()

        for book in (FUND_WEIGHTS, INDEX_WEIGHTS, BENCHMARK_WEIGHTS):
            weights = self.run.tables.get(book)

            if weights is None or weights.empty:
                continue

            at = as_of_position(weights.index, date)

            if at is None:
                continue

            row = weights.iloc[at]
            names.update(str(name) for name in row.index[row.fillna(0.0) != 0.0])

        return sorted(names)

    def _estimate(self,
                  day: pd.Timestamp) -> RiskModel | FactorRiskModel:
        """The model for the rebalance on *day*."""
        window = self._window(day)
        names = self.held(day)
        panel = window.reindex(columns=names)
        # In one pass over the panel (BN-411): name by name it was 4 s of the
        # 20 fits at 6,000 names.
        complete = panel.notna().all(axis=0).to_numpy()
        covered = [name for name, whole in zip(names, complete, strict=True) if whole]
        self._uncovered[day] = [name for name in names if name not in covered]

        if not covered:
            raise CalculationError("Risk model", f"no name held on {day.date()} has returns "
                                                 f"over the {self.settings.window} sessions "
                                                 f"before it")

        panel = panel[covered]

        if self.settings.kind == SAMPLE:
            return estimate_risk_model(panel, target=self.settings.target,
                                       intensity=self.settings.intensity,
                                       periods_per_year=self.run.periods_per_year)

        assert self.exposures is not None
        loadings = self.exposures(day, covered).reindex(covered).fillna(0.0)

        return fit_factor_model(panel, loadings, periods_per_year=self.run.periods_per_year)

    def _window(self,
                day: pd.Timestamp) -> pd.DataFrame:
        """The window's daily name returns: the sessions before *day*."""
        if self._sorted is None:
            self._sorted = self.run.tables[NAME_RETURNS].sort_index()

        returns = self._sorted
        last = as_of_position(returns.index, day - pd.Timedelta(days=1))
        sessions = 0 if last is None else last + 1

        if sessions < self.settings.window:
            raise CalculationError("Risk model", f"{sessions} sessions of returns before "
                                                 f"{day.date()}, fewer than the "
                                                 f"{self.settings.window} it is estimated on")

        return returns.iloc[sessions - self.settings.window:sessions]
