# src/beacon/analysis/analytics/vehicle.py
"""
A run's vehicle analytics: the fund as an ETF and its futures.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class VehicleCategory(Category):
    """A run's vehicle analytics: the fund as an ETF and its futures."""

    name = "vehicle"
