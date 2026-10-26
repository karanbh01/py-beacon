# src/beacon/analysis/packing.py
"""
A run's market inputs, kept small, and the data source they stand in for.

A run keeps the closes it read, in each name's own currency, rather than the
tables worked out from them: a close is quoted to a few decimals, so it packs
into a small whole number, where a return or a weight is a full-precision
float that barely compresses. The tables are worked out again from these when
first asked for, by the same code a finished run used, on the same numbers,
so they come back exactly.

- Closes are kept per name as whole numbers at the fewest decimals that hold
  every one of that name's closes exactly (whole cents for most), with the
  scale beside them. A name whose closes need more than six decimals is kept
  as it is.
- `StoredSource` serves the kept closes' currencies, exchange rates and
  corporate actions to the code that reads them from a data store.
"""
# BN-403. At 6,000 names over six years, closes in whole cents, by name then
# date and delta-packed, are 12 MB where the returns worked out from them are
# 62 MB.
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..data.corporate_actions import CorporateActions
from ..data.prices import currencies_of

# The tables a run's packed closes are kept in.
CLOSES_PACKED = "closes_packed"
CLOSES_FLOAT = "closes_float"
CLOSES_SCALE = "closes_scale"

# How many decimals a close may be packed at; a name needing more is kept as
# it is.
MAX_DECIMALS = 6

# The columns of the currencies and corporate actions kept with a run.
CURRENCY = "CURRENCY"
ACTION_COLUMNS = ("IDENTIFIER", "EX_DATE", "TYPE", "VALUE", "STATUS")


def decimals_of(values: np.ndarray) -> int | None:
    """The fewest decimals that hold every one of *values* exactly, or None
    when more than `MAX_DECIMALS` would be needed."""
    found = _decimals(values.reshape(-1, 1))[0]

    return None if found < 0 else int(found)


def _decimals(grid: np.ndarray) -> np.ndarray:
    """`decimals_of` for each column of *grid*, -1 where none fits."""
    answer = np.full(grid.shape[1], -1)
    blank = np.isnan(grid)

    for decimals in range(MAX_DECIMALS + 1):
        scale = 10.0 ** decimals
        whole = np.rint(grid * scale)
        exact = ((whole / scale == grid) & (np.abs(whole) < 2.0 ** 53)) | blank
        answer[(answer < 0) & exact.all(axis=0)] = decimals

        if (answer >= 0).all():
            break

    return answer


def pack_closes(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """*closes* (dates by name) as the tables that keep them.

    Returns:
        dict: ``closes_packed``, each packable name as whole numbers at its
        scale; ``closes_scale``, each of those names' decimals; and
        ``closes_float``, the names that are not packable, as they are.
    """
    names = [str(name) for name in closes.columns]
    grid = closes.to_numpy(dtype=float)
    decimals = _decimals(grid)
    packable = decimals >= 0
    blank = np.isnan(grid)
    whole = np.rint(grid * 10.0 ** np.where(packable, decimals, 0))
    whole[blank] = 0
    packed = {names[i]: pd.arrays.IntegerArray(whole[:, i].astype("int64"), blank[:, i])
              for i in np.flatnonzero(packable)}

    return {CLOSES_PACKED: pd.DataFrame(packed, index=closes.index),
            CLOSES_SCALE: pd.DataFrame({"decimals": pd.Series(
                {names[i]: int(decimals[i]) for i in np.flatnonzero(packable)}, dtype="int64")}),
            CLOSES_FLOAT: pd.DataFrame({names[i]: grid[:, i] for i in np.flatnonzero(~packable)},
                                       index=closes.index)}


def unpack_closes(tables: Mapping[str, pd.DataFrame],
                  order: list[str] | None = None) -> pd.DataFrame:
    """The closes `pack_closes` kept, exactly as they were, names in *order*
    (or packed names first)."""
    packed = tables[CLOSES_PACKED]
    scales = tables[CLOSES_SCALE]["decimals"]
    columns: dict[str, np.ndarray] = {}

    for name in packed.columns:
        whole = packed[name].to_numpy(dtype=float, na_value=np.nan)
        columns[name] = whole / 10.0 ** int(scales[name])

    floats = tables.get(CLOSES_FLOAT, pd.DataFrame())
    frame = pd.DataFrame(columns, index=packed.index)

    for name in floats.columns:
        frame[name] = floats[name].to_numpy(dtype=float)

    return frame[order] if order is not None else frame


class _Actions:
    """The face of `CorporateActions` the return calculation reads."""

    def __init__(self,
                 data: pd.DataFrame):
        self.data = data
        self.is_empty = data.empty


class StoredSource:
    """The parts of a data store a run's kept market inputs stand in for:
    each name's currency, the exchange rates on the kept days and the
    corporate actions, for the code that reads them from a `DataFetcher`.

    Args:
        currencies: Each name's currency, indexed by name, in a ``CURRENCY``
            column.
        rates: Each currency's rate into the book's currency on the kept
            days, a column per currency.
        actions: The kept names' corporate actions.
        book: The book's currency.
    """

    def __init__(self,
                 currencies: pd.DataFrame,
                 rates: pd.DataFrame,
                 actions: pd.DataFrame,
                 book: str):
        self._currencies = currencies
        self._rates = rates
        self.book = book.upper()
        self.corporate_actions: Any = _Actions(actions)

    def fetch_market_data(self,
                          identifier: str | list[str],
                          start_date: str | None = None,
                          end_date: str | None = None,
                          columns: list[str] | None = None) -> pd.DataFrame:
        """No market data: the kept closes are handed to the calculation
        directly, so this is never what it reads."""
        return pd.DataFrame()

    def fetch_reference_data(self,
                             identifier: str | list[str],
                             date: str | None = None,
                             columns: list[str] | None = None) -> pd.DataFrame:
        """Each kept name's currency, as reference data gives it."""
        names = [identifier] if isinstance(identifier, str) else list(identifier)

        return self._currencies.loc[[name for name in names if name in self._currencies.index]]

    def fx_rates_on(self,
                    from_currency: str,
                    to_currency: str,
                    days: pd.Index) -> pd.Series | None:
        """The kept rate from *from_currency* into the book's currency on
        *days*, or None when none was kept."""
        if to_currency.upper() != self.book or from_currency not in self._rates.columns:
            return None

        rates = self._rates[from_currency].reindex(days)
        rates.name = None

        return rates


def market_inputs(source: Any,
                  closes: pd.DataFrame,
                  currency: str) -> dict[str, pd.DataFrame]:
    """What a run keeps beside its closes so its returns can be worked out
    again: each name's currency, the rates the conversion read, and the
    names' corporate actions.

    Args:
        source: The data the run used.
        closes: The closes kept, dates by name.
        currency: The book's currency.
    """
    names = [str(name) for name in closes.columns]
    target = currency.upper()
    quoted = currencies_of(source, names, default=target)
    rates: dict[str, pd.Series] = {}

    for code in sorted(set(quoted.values()) - {target}):
        found = source.fx_rates_on(code, target, closes.index)

        if found is not None:
            rates[code] = found.reindex(closes.index)

    actions = getattr(source, "corporate_actions", None)
    kept = pd.DataFrame(columns=list(ACTION_COLUMNS))

    if isinstance(actions, CorporateActions) and not actions.is_empty:
        frame = actions.data.reset_index(drop=True)
        frame = frame[frame["IDENTIFIER"].astype(str).isin(set(names))]
        columns = [column for column in ACTION_COLUMNS if column in frame.columns]
        kept = frame[columns].reset_index(drop=True)

    return {"currencies": pd.DataFrame({CURRENCY: pd.Series(quoted)}),
            "rates": pd.DataFrame(rates, index=closes.index),
            "actions": kept}
