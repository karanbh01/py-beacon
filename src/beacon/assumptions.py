# src/beacon/assumptions.py
"""
What a backtest takes as given about markets and data, in one object.

    from beacon import ModellingAssumptions, use_modelling_assumptions

    use_modelling_assumptions(ModellingAssumptions(fx_policy="EXACT_DAY"))

    Backtest(initial_capital=1e6,
             modelling_assumptions=ModellingAssumptions(cash_rate=0.02))

## Two kinds of assumption

**Data treatment** decides how data is read: the FX policy on a day a pair
printed no rate, how long a name may go without trading before it is stale,
and how far a free float carries over blank cells. The index calculation reads
data too, so these reach it as well, and an index and the backtest tracking it
never assume different things.

**Simulation conventions** decide how a backtest is valued and measured: what
cash earns, the risk-free rate the Sharpe ratio is measured against, how many
periods make a year when returns are annualised, the share of each
dividend withheld, and how far a day's traded volume carries over blank
cells when an execution limit needs it. They do not affect an index.

## Unset fields, and where their values come from

Every field defaults to None, meaning "not set here". A backtest's own
assumptions are laid over the process-wide default field by field, so passing
only `cash_rate` keeps the default's FX policy. A data-treatment field still
unset after that takes the data source's own setting, and a simulation field
takes the library default. `resolved` produces the fully specified result, and
that is what a run records.
"""
# BN-276. The fourth part of a backtest beside the strategy, the
# implementation and the vehicle (decisions/0006).
import threading
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from .data.fetcher import FX_POLICIES, KEEP, DataFetcher

# Library defaults for the simulation conventions, which are today's
# behaviour: cash earned nothing, Sharpe used a zero rate, and returns were
# annualised over 252 trading days.
DEFAULT_CASH_RATE = 0.0
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_PERIODS_PER_YEAR = 252
# BN-266: a missed volume print is carried this far before the execution
# limit falls back to the average daily volume.
DEFAULT_VOLUME_BACKFILL_DAYS = 5

# The fields that change how data is read, and so reach the index.
DATA_TREATMENT = ("fx_policy", "max_price_staleness_days",
                  "free_float_backfill_days")


@dataclass(frozen=True)
class ModellingAssumptions:
    """A backtest's modelling assumptions. Every field is optional.

    Args:
        fx_policy: How an FX rate is read on a day the pair printed none:
            ``"CARRY_FORWARD"`` uses the last rate published, ``"EXACT_DAY"``
            only that day's. Unset takes the data source's.
        max_price_staleness_days: How many calendar days a name may go without
            trading before it is dropped as stale. 0 keeps every name however
            long ago it traded. Unset takes the data source's.
        free_float_backfill_days: How many calendar days a reported free float
            carries forward over blank cells. 0 turns carrying off. Unset
            takes the data source's.
        cash_rate: The annual rate cash earns, accrued daily over calendar
            days (ACT/365). Unset is 0.
        risk_free_rate: The annual rate the Sharpe ratio is measured against.
            Unset is 0.
        periods_per_year: How many periods make a year when returns are
            annualised. Unset is 252.
        withholding_tax_rate: The share of each dividend withheld before the
            book receives it, as a decimal. Unset is 0.
        volume_backfill_days: How many calendar days a name's last reported
            volume stands in for a blank one when an execution limit reads
            the day's volume. Past that, its average daily volume is used. A
            volume reported as 0 is never replaced. Unset is 5.

    Raises:
        ValueError: If a field is set to a value it cannot take.
    """
    fx_policy: str | None = None
    max_price_staleness_days: int | None = None
    free_float_backfill_days: int | None = None
    cash_rate: float | None = None
    risk_free_rate: float | None = None
    periods_per_year: int | None = None
    withholding_tax_rate: float | None = None
    volume_backfill_days: int | None = None

    def __post_init__(self) -> None:
        if self.fx_policy is not None and self.fx_policy not in FX_POLICIES:
            raise ValueError(f"Unknown fx_policy: {self.fx_policy!r}. Supported "
                             f"values: {list(FX_POLICIES)}.")

        for name in ("max_price_staleness_days", "free_float_backfill_days",
                     "volume_backfill_days"):
            value = getattr(self, name)

            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, int)
                                      or value < 0):
                raise ValueError(f"{name} must be a whole number of days, 0 or "
                                 f"more, got {value!r}.")

        if self.periods_per_year is not None and self.periods_per_year < 1:
            raise ValueError(f"periods_per_year must be at least 1, got "
                             f"{self.periods_per_year!r}.")

        if (self.withholding_tax_rate is not None
                and not 0.0 <= self.withholding_tax_rate < 1.0):
            raise ValueError(f"withholding_tax_rate is a share of each "
                             f"dividend from 0 up to 1, got "
                             f"{self.withholding_tax_rate!r}.")

        for name in ("cash_rate", "risk_free_rate"):
            value = getattr(self, name)

            if value is not None and not -1.0 < value < 1.0:
                raise ValueError(f"{name} is an annual rate as a decimal, such "
                                 f"as 0.02 for 2%, got {value!r}.")

    def over(self,
             base: "ModellingAssumptions") -> "ModellingAssumptions":
        """These assumptions, with *base*'s filling every field unset here."""
        return replace(base, **{field.name: getattr(self, field.name)
                                for field in fields(self)
                                if getattr(self, field.name) is not None})

    def resolved(self,
                 fetcher: DataFetcher) -> "ModellingAssumptions":
        """Every field set: unset data treatment from *fetcher*, the rest from
        the library defaults."""
        staleness = fetcher.max_price_staleness_days

        return ModellingAssumptions(
            fx_policy=self.fx_policy or fetcher.fx_policy,
            max_price_staleness_days=(self.max_price_staleness_days
                                      if self.max_price_staleness_days is not None
                                      else staleness or 0),
            free_float_backfill_days=(self.free_float_backfill_days
                                      if self.free_float_backfill_days is not None
                                      else fetcher.free_float_backfill_days),
            cash_rate=_or(self.cash_rate, DEFAULT_CASH_RATE),
            risk_free_rate=_or(self.risk_free_rate, DEFAULT_RISK_FREE_RATE),
            periods_per_year=_or(self.periods_per_year,
                                 DEFAULT_PERIODS_PER_YEAR),
            withholding_tax_rate=_or(self.withholding_tax_rate, 0.0),
            volume_backfill_days=_or(self.volume_backfill_days,
                                     DEFAULT_VOLUME_BACKFILL_DAYS))

    def with_defaults(self) -> "ModellingAssumptions":
        """The simulation conventions filled from the library defaults where
        unset; the data treatment left as it is."""
        return replace(self,
                       cash_rate=_or(self.cash_rate, DEFAULT_CASH_RATE),
                       risk_free_rate=_or(self.risk_free_rate,
                                          DEFAULT_RISK_FREE_RATE),
                       periods_per_year=_or(self.periods_per_year,
                                            DEFAULT_PERIODS_PER_YEAR),
                       withholding_tax_rate=_or(self.withholding_tax_rate, 0.0),
                       volume_backfill_days=_or(self.volume_backfill_days,
                                                DEFAULT_VOLUME_BACKFILL_DAYS))

    def effective(self) -> "ModellingAssumptions":
        """These assumptions laid over the process-wide default."""
        return self.over(current_modelling_assumptions())

    def applied_to(self,
                   fetcher: DataFetcher) -> DataFetcher:
        """*fetcher*, reading data under these assumptions' data treatment.

        The same fetcher when nothing set here differs from its own settings,
        so a run with no assumptions shares its data source's caches.
        """
        staleness = self.max_price_staleness_days

        # The data source spells "no limit" as None, which here means unset;
        # 0 is this object's spelling of it.
        return fetcher.with_settings(
            fx_policy=_or(self.fx_policy, KEEP),
            max_price_staleness_days=(KEEP if staleness is None
                                      else staleness or None),
            free_float_backfill_days=_or(self.free_float_backfill_days, KEEP))

    def data_treatment(self) -> dict[str, Any]:
        """The fields that reach an index calculation, by name."""
        return {name: getattr(self, name) for name in DATA_TREATMENT}

    def as_dict(self) -> dict[str, Any]:
        """Every field, by name: what a result records."""
        return asdict(self)


def data_treatment_of(fetcher: Any) -> ModellingAssumptions | None:
    """The data treatment *fetcher* reads under, as assumptions with only
    those fields set. None for a data source that is not a `DataFetcher`."""
    if not isinstance(fetcher, DataFetcher):
        return None

    return ModellingAssumptions(
        fx_policy=fetcher.fx_policy,
        max_price_staleness_days=fetcher.max_price_staleness_days or 0,
        free_float_backfill_days=fetcher.free_float_backfill_days)


def apply(assumptions: ModellingAssumptions,
          fetcher: Any) -> Any:
    """*fetcher* under *assumptions*' data treatment, when it is a
    `DataFetcher`; any other data source (a test double) unchanged."""
    if not isinstance(fetcher, DataFetcher):
        return fetcher

    return assumptions.applied_to(fetcher)


def _or(value: Any,
        default: Any) -> Any:
    """*value*, or *default* when it is unset."""
    return default if value is None else value


class _ProcessAssumptions:
    """The process-wide default, behind a lock like the data source's."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = ModellingAssumptions()


_state = _ProcessAssumptions()


def use_modelling_assumptions(assumptions: ModellingAssumptions | None) -> None:
    """Set the process-wide modelling assumptions, or reset them with None.

    Every backtest and index calculation that is not given its own reads
    these. A backtest's own assumptions are laid over them field by field.
    """
    with _state.lock:
        _state.current = (assumptions if assumptions is not None
                          else ModellingAssumptions())


def current_modelling_assumptions() -> ModellingAssumptions:
    """The process-wide modelling assumptions."""
    with _state.lock:
        return _state.current
