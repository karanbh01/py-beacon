# src/beacon/data/prices.py
"""Prices for several instruments at once, in one currency.

    prices = fetcher.fetch_prices(["AAA", "BBB"], "2024-01-02", "2024-06-28",
                                  currency="USD")

Comparing or adding up prices across instruments only means something once
they are in the same money. This is the one read that says which money: each
instrument's prices are converted from the currency it is quoted in (its
reference data's `CURRENCY`) into *currency*, day by day, under the dataset's
FX policy and through the same rate lookup every other conversion uses
(`DataFetcher.fx_rates_on`).

- An instrument whose currency has **no rate at all** into *currency* is
  refused, naming it, rather than left in its own currency.
- A day the policy has no rate for is left empty (NaN) rather than guessed.
- An instrument with no currency on record is read as quoted in *currency*: a
  dataset whose reference data does not model currency is single-currency.
- With `currency=None` nothing is converted, and each column stays in its
  instrument's own currency.
"""
# BN-253: before this read, each caller decided for itself whether to
# convert, and the server's risk, optimisation, weights and attribution views
# and LiquidityRule did not. See the audit on #266.
from collections.abc import Iterable
from typing import Protocol

import pandas as pd

from ..exceptions import CalculationError

PRICE_COLUMN = "CLOSE"
CURRENCY_COLUMN = "CURRENCY"


class _Source(Protocol):
    """What this module reads from a `DataFetcher`."""

    def fetch_market_data(self,
                          identifier: str | list[str],
                          start_date: str | None = None,
                          end_date: str | None = None,
                          columns: list[str] | None = None) -> pd.DataFrame: ...

    def fetch_reference_data(self,
                             identifier: str | list[str],
                             date: str | None = None,
                             columns: list[str] | None = None) -> pd.DataFrame: ...

    def fx_rates_on(self,
                    from_currency: str,
                    to_currency: str,
                    days: pd.Index) -> pd.Series | None: ...


def prices_in(source: _Source,
              identifiers: Iterable[str],
              start: str | None = None,
              end: str | None = None,
              currency: str | None = None,
              column: str = PRICE_COLUMN) -> pd.DataFrame:
    """Prices by date and instrument, converted into *currency*.

    Args:
        source: The data, usually a `DataFetcher`.
        identifiers: The instruments. One without prices in the window is
            left out.
        start: First date, inclusive.
        end: Last date, inclusive.
        currency: The currency to convert into, or None to leave each
            instrument in its own.
        column: The market-data column to read.

    Returns:
        pd.DataFrame: Dates as rows, one column per instrument priced, in the
        order given.

    Raises:
        CalculationError: If an instrument's currency has no rate into
            *currency*.
    """
    wanted = list(dict.fromkeys(identifiers))
    wide = _wide(source, wanted, start, end, column)

    if wide.empty or currency is None:
        return wide

    target = currency.upper()
    quoted = currencies_of(source, list(wide.columns), default=target)

    for from_currency in sorted(set(quoted.values()) - {target}):
        names = [name for name in wide.columns if quoted[name] == from_currency]
        rates = source.fx_rates_on(from_currency, target, wide.index)

        if rates is None:
            raise CalculationError(
                calculation_name="Prices",
                details=(f"no {from_currency}/{target} rate, so "
                         f"{', '.join(names)} cannot be priced in {target}. "
                         f"Leaving them in {from_currency} would compare them "
                         f"with the rest on magnitude alone. Load the pair, or "
                         f"leave these names out."))

        wide[names] = wide[names].mul(rates, axis=0)

    return wide


def currencies_of(source: _Source,
                  identifiers: list[str],
                  default: str) -> dict[str, str]:
    """The currency each instrument is quoted in, from its latest record.

    The latest record rather than one on a date, so an instrument that has
    since been delisted still has its currency. *default* stands in where no
    currency is on record.
    """
    frame = source.fetch_reference_data(list(identifiers))

    if frame.empty or CURRENCY_COLUMN not in frame.columns:
        return dict.fromkeys(identifiers, default)

    known = frame[CURRENCY_COLUMN].dropna()
    latest = known.groupby(level=0).last() if not known.empty else known

    return {name: str(latest[name]).upper() if name in latest.index else default
            for name in identifiers}


def _wide(source: _Source,
          identifiers: list[str],
          start: str | None,
          end: str | None,
          column: str) -> pd.DataFrame:
    """One batched read, as dates by instrument."""
    if not identifiers:
        return pd.DataFrame()

    market = source.fetch_market_data(identifiers, start, end)

    if market.empty or column not in market.columns:
        return pd.DataFrame()

    if isinstance(market.index, pd.MultiIndex):
        wide = market[column].unstack(level="IDENTIFIER")
    else:
        # One instrument comes back indexed by date alone.
        wide = market[[column]].rename(columns={column: identifiers[0]})

    wide = wide.sort_index().astype(float)

    return wide[[name for name in identifiers if name in wide.columns]]
