"""
Environment configuration dataclasses and the central Environment class.
"""

from dataclasses import dataclass, fields
from typing import Any

import pandas as pd


@dataclass
class DataSourceConfig:
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
    DATE_FORMAT: str = "%Y-%m-%d"

@dataclass
class CalendarConfig:
    TRADING_DAYS: int = 252
    DAY_COUNT_CONVENTION: str = 'ACT/365'

@dataclass
class SimulationConfig:
    TRANSACTION_COST: float = 0.0
    FLOAT_TOLERANCE: float = 1e-9

@dataclass
class IndexConfig:
    pass

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
        """Set one or more validated parameters.

        Raises ValueError on unknown parameter names.
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
        """Return all current settings as a nested dict."""
        result: dict[str, dict[str, Any]] = {}
        for attr in _CATEGORIES:
            instance = getattr(self, attr)
            result[attr] = {
                f.name: getattr(instance, f.name)
                for f in fields(type(instance))
                if not isinstance(getattr(instance, f.name), pd.DataFrame)
            }
        return result
