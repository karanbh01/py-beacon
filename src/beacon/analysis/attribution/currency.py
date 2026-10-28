# src/beacon/analysis/attribution/currency.py
"""
Currency attribution: how much of a book's return came from its names' own
markets and how much from the exchange rates into the book's currency.

A name's daily return in the book's currency ``r`` is its return in its own
``l`` and the move of its currency ``x``: ``1 + r = (1 + l)(1 + x)``, so

    r = l + x + l x

and each name contributes its weight at the close before times each part:
the local return, the currency effect, and the cross term on a line of its
own. Local returns are worked out from the run's own closes and corporate
actions in each name's currency, as the book's are.
"""
# BN-395. Agreed with Karan on 2026-10-28: the cross term on its own line.
import pandas as pd

from ...data.returns import name_returns
from ..data import RunData
from ..packing import StoredSource
from ..panels import ACTIONS, CLOSES, CURRENCIES, NAME_RETURNS, RATES

LOCAL = "local"
CURRENCY = "currency"
CROSS = "cross"
PARTS = (LOCAL, CURRENCY, CROSS)


def local_returns(run: RunData) -> pd.DataFrame:
    """Each name's daily total return in its own currency, on the days of
    the run's name returns."""
    tables = run.tables
    stored = StoredSource(tables[CURRENCIES], tables[RATES], tables[ACTIONS], run.currency)
    closes = tables[CLOSES]
    found = name_returns(stored, list(closes.columns), currency=None, closes=closes)

    return found.rename(columns=str).reindex(tables[NAME_RETURNS].index)


def currency_parts(lagged: pd.DataFrame,
                   book: pd.DataFrame,
                   local: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each name's daily contribution from each part: its weight at the close
    before times its local return, its currency effect and the cross term."""
    names = [name for name in lagged.columns if name in book.columns and name in local.columns]
    weights = lagged[names]
    own = local.reindex(lagged.index)[names]
    moved = ((1.0 + book.reindex(lagged.index)[names]) / (1.0 + own) - 1.0)

    return {LOCAL: (weights * own).fillna(0.0), CURRENCY: (weights * moved).fillna(0.0),
            CROSS: (weights * own * moved).fillna(0.0)}


def by_currency(parts: dict[str, pd.DataFrame],
                currencies: pd.Series) -> dict[str, pd.DataFrame]:
    """Each part's daily contributions added up by the names' currencies."""
    labels = currencies.reindex(next(iter(parts.values())).columns).fillna("Unknown")

    return {name: frame.T.groupby(labels.to_numpy()).sum().T for name, frame in parts.items()}
