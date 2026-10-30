# src/beacon/portfolio/valuation.py
"""
A book's weights from what it holds: each name's units at its price over
the book's net asset value.
"""
# BN-417: moved from the server's holdings view, which now calls this.
import pandas as pd


def held_weights(units: dict[str, float],
                 prices: pd.Series,
                 nav: float) -> dict[str, tuple[float, float]]:
    """Each name's units and weight: its units at its price in *prices* over
    *nav*. A name with no price is left out."""
    return {name: (count, float(count * prices[name] / nav))
            for name, count in units.items() if name in prices and pd.notna(prices[name])}
