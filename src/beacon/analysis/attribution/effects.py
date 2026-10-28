# src/beacon/analysis/attribution/effects.py
"""
A return split into named effects each day (local, currency and cross term;
market, factors and specific), linked over the days so the effects add up to
the compounded return, with what they leave unexplained as the residual.

A book's own return links by Carino's factor ``ln(1 + R) / R``; a return over
a comparator by the active factor ``(ln(1 + R) - ln(1 + R^b)) / (R - R^b)``,
as Brinson-Fachler links (`beacon.analysis.attribution.brinson`).
"""
# BN-395.
from dataclasses import dataclass

import pandas as pd

from .brinson import active_carino
from .contribution import carino_factor


@dataclass(frozen=True)
class EffectsResult:
    """A return by effect, linked over the days.

    Attributes:
        start: The first day attributed.
        end: The last.
        periods: The days.
        total_return: The return explained: a book's compounded return, or
            the fund's less a comparator's.
        effects: Each effect, linked, largest first.
        explained: What the effects add up to.
        residual: The return less what the effects explain.
        table: By group where one was asked for (each currency, say), each
            effect linked; else empty.
    """
    start: str
    end: str
    periods: int
    total_return: float
    effects: pd.Series
    explained: float
    residual: float
    table: pd.DataFrame

    def reconciles(self,
                   tolerance: float = 1e-9) -> bool:
        """Whether the effects account for the return."""
        return abs(self.residual) <= tolerance


def linked_effects(daily: pd.DataFrame,
                   ours: pd.Series,
                   theirs: pd.Series | None = None,
                   table: dict[str, pd.DataFrame] | None = None) -> EffectsResult:
    """Each of *daily*'s effects (days by effect) linked over its days, as a
    share of *ours*' compounded return or, with *theirs*, of the return over
    it; *table*'s groups (effect -> days by group) linked the same way."""
    days = daily.index.intersection(ours.index)
    daily, ours = daily.loc[days], ours.loc[days].astype(float)
    total_ours = float((1.0 + ours).prod() - 1.0)

    if theirs is None:
        whole = carino_factor(total_ours)
        factors = pd.Series([carino_factor(float(value)) for value in ours], index=days)
        total = total_ours
    else:
        theirs = theirs.reindex(days).astype(float)
        total_theirs = float((1.0 + theirs).prod() - 1.0)
        whole = active_carino(total_ours, total_theirs)
        factors = pd.Series([active_carino(float(a), float(b))
                             for a, b in zip(ours, theirs, strict=True)], index=days)
        total = total_ours - total_theirs

    scale = factors / whole
    effects = daily.mul(scale, axis=0).sum()
    grouped = (pd.DataFrame({name: frame.loc[days].mul(scale, axis=0).sum()
                             for name, frame in table.items()}).fillna(0.0)
               if table else pd.DataFrame())
    explained = float(effects.sum())

    return EffectsResult(start=days[0].isoformat() if len(days) else "",
                         end=days[-1].isoformat() if len(days) else "", periods=len(days),
                         total_return=total, effects=effects.sort_values(ascending=False),
                         explained=explained, residual=total - explained, table=grouped)
