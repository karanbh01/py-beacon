"""
Session configuration.

An :class:`Environment` groups settings into categories (data sources, data
formats, calendar, simulation) and sets them by flat name through
``set_environment(**kwargs)``. Its main use is to tell
`beacon.data.loader.load_data` where the market and reference data are,
either as in-memory DataFrames or as file paths.
"""
from .config import Environment

__all__ = ["Environment"]
