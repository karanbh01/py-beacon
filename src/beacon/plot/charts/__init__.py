# src/beacon/plot/charts/__init__.py
"""
A chart class for each kind of chart, each taking a DataFrame or a Series and
drawing it inside the Beacon frame.

Building a chart only describes it; it is drawn when it is shown, saved, or
its figure is asked for. So these import without matplotlib.
"""
from .base import Chart, Plan
from .lines import Area, Line

__all__ = ["Area", "Chart", "Line", "Plan"]
