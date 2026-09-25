"""
Environment configuration dataclasses and the central Environment class.

`beacon.data.loader.load_data` reads the market and reference data settings
and ``DATE_FORMAT`` from an Environment. The other fields are stored, and
reported by :meth:`Environment.summary`, but nothing else in Beacon reads them.
"""

from dataclasses import dataclass, fields
from typing import Any

import pandas as pd


@dataclass
class DataSourceConfig:
    """Where the input data comes from.

    Each dataset can be given either as an in-memory DataFrame or as a file
    path. When both are set for the same dataset, `load_data` uses the
    DataFrame.

    Attributes:
        MARKET_DATA_PATH: File to read market data (prices) from.
        REFERENCE_DATA_PATH: File to read reference data from.
        FUNDAMENTALS_DATA_PATH: File to read fundamentals data from.
        CORPORATE_ACTIONS_DATA_PATH: File to read corporate actions from.
        FX_RATES_DATA_PATH: File to read FX rates from.
        MARKET_DATA: Market data as a long-form DataFrame.
        REFERENCE_DATA: Reference data as a DataFrame.
        FUNDAMENTALS_DATA: Fundamentals data as a DataFrame.
        CORPORATE_ACTIONS_DATA: Corporate actions as a DataFrame.
        FX_RATES_DATA: FX rates as a DataFrame.
    """
    MARKET_DATA_PATH: str | None = None
    REFERENCE_DATA_PATH: str | None = None
    FUNDAMENTALS_DATA_PATH: str | None = None
    CORPORATE_ACTIONS_DATA_PATH: str | None = None
    FX_RATES_DATA_PATH: str | None = None
    MARKET_DATA: pd.DataFrame | None = None
    REFERENCE_DATA: pd.DataFrame | None = None
    FUNDAMENTALS_DATA: pd.DataFrame | None = None
    CORPORATE_ACTIONS_DATA: pd.DataFrame | None = None
    FX_RATES_DATA: pd.DataFrame | None = None


@dataclass
class DataConfig:
    """How input data is parsed.

    Attributes:
        DATE_FORMAT: The strftime format of the market data's dates.
    """
    DATE_FORMAT: str = "%Y-%m-%d"

@dataclass
class CalendarConfig:
    """Calendar conventions.

    Attributes:
        TRADING_DAYS: Trading days per year.
        DAY_COUNT_CONVENTION: Day-count convention name, such as ``"ACT/365"``.
    """
    TRADING_DAYS: int = 252
    DAY_COUNT_CONVENTION: str = 'ACT/365'

@dataclass
class SimulationConfig:
    """Simulation settings.

    Attributes:
        TRANSACTION_COST: Transaction cost.
        FLOAT_TOLERANCE: Tolerance for floating-point comparisons.
    """
    TRANSACTION_COST: float = 0.0
    FLOAT_TOLERANCE: float = 1e-9

@dataclass
class IndexConfig:
    """Reserved for index settings. It holds none and is not part of Environment."""

# A settings category: one of the dataclasses grouped on Environment.
CategoryConfig = DataSourceConfig | DataConfig | CalendarConfig | SimulationConfig

_CATEGORIES: dict[str, type[CategoryConfig]] = {
    "data_source": DataSourceConfig,
    "data": DataConfig,
    "calendar": CalendarConfig,
    "simulation": SimulationConfig,
}


class Environment:
    """Centralized configuration for a Beacon session.

    Settings are grouped into category dataclasses and can be set via
    ``set_environment(**kwargs)`` using flat parameter names.

    Attributes:
        data_source: Where the input data comes from.
        data: How input data is parsed.
        calendar: Calendar conventions.
        simulation: Simulation settings.
    """

    def __init__(self) -> None:
        self.data_source = DataSourceConfig()
        self.data = DataConfig()
        self.calendar = CalendarConfig()
        self.simulation = SimulationConfig()

    def _build_lookup(self) -> dict[str, tuple[CategoryConfig, str]]:
        """Map each field name to its (category_instance, field_name) pair."""
        lookup: dict[str, tuple[CategoryConfig, str]] = {}
        for attr, cls in _CATEGORIES.items():
            instance = getattr(self, attr)
            for f in fields(cls):
                lookup[f.name] = (instance, f.name)
        return lookup

    def set_environment(self,
                        **kwargs: Any) -> None:
        """Set one or more parameters by their flat field names.

        Names are checked against every category's fields before anything is
        set, so an unknown name changes nothing. Values are not checked.

        Args:
            **kwargs: Field name to new value, e.g. ``MARKET_DATA_PATH="prices.csv"``.

        Raises:
            ValueError: If any name is not a field of a settings category.
        """
        lookup = self._build_lookup()

        unknown = [k for k in kwargs if k not in lookup]
        if unknown:
            raise ValueError(
                f"Unknown parameter(s): {', '.join(unknown)}. "
                f"Valid parameters: {', '.join(sorted(lookup))}"
            )

        for name, value in kwargs.items():
            instance, field_name = lookup[name]
            setattr(instance, field_name, value)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return all current settings as a nested dict.

        Returns:
            dict: Category name to ``{field name: value}``. DataFrame values
            are left out.
        """
        result: dict[str, dict[str, Any]] = {}
        for attr in _CATEGORIES:
            instance = getattr(self, attr)
            result[attr] = {
                f.name: getattr(instance, f.name)
                for f in fields(type(instance))
                if not isinstance(getattr(instance, f.name), pd.DataFrame)
            }
        return result
