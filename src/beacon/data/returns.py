# src/beacon/data/returns.py
"""
Each name's daily total return, in one currency.

    returns = name_returns(fetcher, ["AAA", "BBB"], "2024-01-02", "2024-06-28",
                           currency="USD")

A day's return is what a holder of one share made, as a share of the previous
close, in *currency*:

    (close × split ratio + cash paid per share) ÷ previous close − 1

- **Splits** (and reverse splits and stock dividends) multiply the shares
  held, so the close on the day is scaled by the ratio rather than read as a
  fall.
- **Cash** (dividends, special dividends and returns of capital) counts on its
  ex-date, gross of any tax withheld, converted at that day's rate. That is
  when a holder is owed it, and how a backtest's NAV counts it.
- **Mergers, spin-offs and rights issues** are not a scaling of one price, so
  no return accounts for them, as an adjusted close leaves them out.
- A **cancelled** action never happened and is left out.
- A day with no close is empty; the next close's return runs from the last
  close before the gap, so nothing is lost across it.
"""
# BN-362. Built from whole tables: on 6,000 names over seven years a name at a
# time took far longer than the backtest it was recorded for.
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .corporate_actions import CASH, RATIO, kind_of, without_cancelled
from .prices import PRICE_COLUMN, _Source, converted, prices_in


def name_returns(source: _Source,
                 identifiers: Iterable[str],
                 start: str | None = None,
                 end: str | None = None,
                 currency: str | None = None,
                 actions: pd.DataFrame | None = None,
                 closes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily total returns by date and name, in *currency*.

    Args:
        source: The data, usually a `DataFetcher`.
        identifiers: The names. One without prices in the window is left out.
        start: First date, inclusive. Its returns are empty, having no close
            before them in the window.
        end: Last date, inclusive.
        currency: The currency returns are measured in; each name's own when
            None.
        actions: The corporate actions, with `IDENTIFIER`, `TYPE`, `VALUE` and
            `EX_DATE`; read from *source*'s `corporate_actions` when None.
        closes: The names' closes in their own currencies, dates by name, for
            a caller that has read them already; read from *source* when
            None.

    Returns:
        pd.DataFrame: Dates by name, NaN where a name has no close or no
        close before it.
    """
    names = list(dict.fromkeys(identifiers))
    local = (prices_in(source, names, start, end, None, PRICE_COLUMN) if closes is None
             else closes)

    if local.empty:
        return local

    book = local if currency is None else converted(source, local, currency)

    ratios, cash = _events(_actions_of(source) if actions is None else actions,
                           local.index, list(local.columns))

    # The rate into *currency* each day, as the converted close over the
    # name's own; what converts the cash it pays on the day.
    rate = book.to_numpy() / local.to_numpy()
    close = book.to_numpy()
    previous = book.ffill().shift(1).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        values = (close * ratios + cash * rate) / previous - 1.0

    values[~(np.isfinite(previous) & (previous > 0) & np.isfinite(close))] = np.nan

    return pd.DataFrame(values, index=local.index, columns=local.columns)


def _actions_of(source: _Source) -> pd.DataFrame:
    """The source's corporate actions, or none."""
    actions = getattr(source, "corporate_actions", None)

    if actions is None or actions.is_empty:
        return pd.DataFrame()

    return actions.data.reset_index(drop=True)


def _events(actions: pd.DataFrame,
            dates: pd.DatetimeIndex,
            names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Each day's split ratio (1 where none) and cash per share (0 where
    none), by date and name, in the names' own currencies.

    An action dated between sessions counts on the next session, as a holder
    first sees it there.
    """
    ratios = np.ones((len(dates), len(names)))
    cash = np.zeros((len(dates), len(names)))

    if actions.empty:
        return ratios, cash

    frame = without_cancelled(actions)
    column = {name: position for position, name in enumerate(names)}
    frame = frame[frame["IDENTIFIER"].astype(str).isin(column)]

    if frame.empty:
        return ratios, cash

    rows = dates.searchsorted(pd.to_datetime(frame["EX_DATE"]).to_numpy())
    columns = frame["IDENTIFIER"].astype(str).map(column).to_numpy()
    kinds = frame["TYPE"].map(kind_of).to_numpy()
    values = pd.to_numeric(frame["VALUE"], errors="coerce").to_numpy(dtype=float)
    inside = (rows < len(dates)) & np.isfinite(values)

    split = inside & (kinds == RATIO) & (values > 0)
    np.multiply.at(ratios, (rows[split], columns[split]), values[split])

    paid = inside & (kinds == CASH) & (values > 0)
    np.add.at(cash, (rows[paid], columns[paid]), values[paid])

    return ratios, cash
