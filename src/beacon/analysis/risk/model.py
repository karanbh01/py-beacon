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

**Structure.** A factor model is ``global`` or ``styles``. A global model
adds country, sector and currency factors to the styles and is fitted on
weekly returns (`beacon.analysis.risk.global_model`); a styles model is the
market and the styles alone, on daily returns. ``auto``, the default, takes
the global model at a rebalance whose names span more than one country and
the styles model otherwise.

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
from ...risk.model.global_fit import MINIMUM_NAMES
from ..data import RunData
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from .global_model import Caps, GlobalModels

SAMPLE = "sample"
FACTOR = "factor"
KINDS = (SAMPLE, FACTOR)

AUTO = "auto"
STYLES = "styles"
GLOBAL = "global"
STRUCTURES = (AUTO, STYLES, GLOBAL)

# A year of daily returns.
DEFAULT_WINDOW = 252

# Two years of weekly returns.
DEFAULT_WEEKS = 104

# The field a name's country is read from by default.
DOMICILE = "COUNTRY_DOMICILE"

# The fewest weeks a model can be estimated over.
MINIMUM_WEEKS = 8

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
        structure: For a factor model, ``global``, ``styles`` or ``auto``.
        country_field: For a global model, the reference field each name's
            country is read from: its domicile by default, its listing
            (``COUNTRY_LISTING``), or another such as ``REGION``; None for
            no country factors.
        minimum_names: For a global model, the fewest names a country needs
            for a factor of its own; one with fewer joins its region's.
        sectors: For a global model, whether sectors are factors.
        currencies: For a global model, whether currencies are factors.
        weekly: For a global model, whether it is fitted on weekly returns;
            on the window's daily returns when not.
        weeks: For a global model on weekly returns, the weeks before each
            rebalance it is estimated on.

    Raises:
        ValueError: If the kind, target or structure is unknown, the window
            is shorter than 20 sessions or the weeks than 8, the minimum
            names below 1, or the intensity outside 0 to 1.
    """
    kind: str = SAMPLE
    window: int = DEFAULT_WINDOW
    target: str = CONSTANT_CORRELATION
    intensity: float | None = None
    structure: str = AUTO
    country_field: str | None = DOMICILE
    minimum_names: int = MINIMUM_NAMES
    sectors: bool = True
    currencies: bool = True
    weekly: bool = True
    weeks: int = DEFAULT_WEEKS

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

        if self.structure not in STRUCTURES:
            raise ValueError(f"Unknown structure {self.structure!r}. "
                             f"Known: {', '.join(STRUCTURES)}.")

        if self.weeks != int(self.weeks) or self.weeks < MINIMUM_WEEKS:
            raise ValueError(f"weeks is a whole number, at least {MINIMUM_WEEKS}, "
                             f"got {self.weeks!r}.")

        if self.minimum_names != int(self.minimum_names) or self.minimum_names < 1:
            raise ValueError(f"minimum_names is a whole number, at least 1, "
                             f"got {self.minimum_names!r}.")

        object.__setattr__(self, "weeks", int(self.weeks))
        object.__setattr__(self, "minimum_names", int(self.minimum_names))
        object.__setattr__(self, "country_field", self.country_field or None)


class RebalanceRiskModels:
    """A run's risk model at each rebalance, each estimated once when first
    asked for.

    Args:
        run: The run's data, with its name returns.
        settings: How the models are estimated.
        exposures: For a factor model, each rebalance's style exposures of
            the names asked about.
        caps: For a global model, each rebalance's market caps of the names
            asked about, which weight its regressions; every name the same
            when None.

    Raises:
        ValueError: If a factor model is asked for without exposures.
    """

    def __init__(self,
                 run: RunData,
                 settings: RiskModelSettings | None = None,
                 exposures: Exposures | None = None,
                 caps: Caps | None = None):
        self.run = run
        self.settings = settings or RiskModelSettings()

        if self.settings.kind == FACTOR and exposures is None:
            raise ValueError("A factor model needs each rebalance's exposures.")

        self.exposures = exposures
        self.caps = caps
        chosen = self.settings
        self._global = GlobalModels(run, chosen.country_field, chosen.minimum_names,
                                    chosen.sectors, chosen.currencies, chosen.weekly)
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

    def is_global(self,
                  date: str | pd.Timestamp) -> bool:
        """Whether the model at the rebalance on *date* is the global one."""
        if self.settings.kind != FACTOR or self.settings.structure == STYLES:
            return False

        day = pd.Timestamp(date)

        return (self.settings.structure == GLOBAL
                or self._global.spans_countries(day, self.held(day)))

    def has_history(self,
                    date: str | pd.Timestamp) -> bool:
        """Whether enough returns come before the rebalance on *date* for its
        model: the weeks of a weekly global model, the window's sessions
        otherwise."""
        day = pd.Timestamp(date)

        if self.is_global(day) and self.settings.weekly:
            return len(self._global.periods_before(day)) >= self.settings.weeks

        return self._sessions_before(day) >= self.settings.window

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
        names = self.held(day)

        if self.is_global(day):
            periods = self.settings.weeks if self.settings.weekly else self.settings.window
            model, self._uncovered[day] = self._global.estimate(day, names, periods,
                                                                self.exposures, self.caps)

            return model

        window = self._window(day)
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
        returns = self._sorted_returns()
        sessions = self._sessions_before(day)

        if sessions < self.settings.window:
            raise CalculationError("Risk model", f"{sessions} sessions of returns before "
                                                 f"{day.date()}, fewer than the "
                                                 f"{self.settings.window} it is estimated on")

        return returns.iloc[sessions - self.settings.window:sessions]

    def _sorted_returns(self) -> pd.DataFrame:
        """The run's daily name returns, in date order."""
        if self._sorted is None:
            self._sorted = self.run.tables[NAME_RETURNS].sort_index()

        return self._sorted

    def _sessions_before(self,
                         day: pd.Timestamp) -> int:
        """The sessions of returns before *day*."""
        last = as_of_position(self._sorted_returns().index, day - pd.Timedelta(days=1))

        return 0 if last is None else last + 1
