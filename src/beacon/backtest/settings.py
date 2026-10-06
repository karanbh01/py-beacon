# src/beacon/backtest/settings.py
"""
How a backtest engine's settings are resolved and checked before it runs:
the book's currency, whether the index was calculated under the same
modelling assumptions, and that no screen was passed as a modifier.
"""
# Moved out of engine.py when it was split (BN-268).
import logging

from ..assumptions import data_treatment_of
from ..data.fetcher import DataFetcher
from ..index.result import IndexResult
from .rules import BacktestModifier
from .screens import Screen

logger = logging.getLogger(__name__)


def book_currency(currency: str | None,
                  index_result: IndexResult) -> str:
    """The book's currency: the one asked for, else the index's, else USD.

    A book in a different currency from its index is a real case (a dollar
    investor tracking a euro index) and is allowed, but said, because its
    tracking figures then include exchange-rate moves the index does not see.
    """
    # BN-228: the default was USD whatever the index's currency, so a euro
    # index's backtest kept its books in dollars and nothing said so.
    index_currency = (index_result.currency.upper()
                      if index_result.currency else None)

    if currency is None:
        if index_currency is None:
            logger.info("The index does not record its currency; the book is "
                        "kept in USD.")

        return index_currency or "USD"

    book = currency.upper()

    if index_currency is not None and book != index_currency:
        logger.info("The book is kept in %s and the index is in %s, so the "
                    "tracking figures include exchange-rate moves.", book,
                    index_currency)

    return book


def warn_if_assumptions_differ(index_result: IndexResult,
                               data_provider: DataFetcher) -> None:
    """Log each data-treatment setting the index was calculated under that
    differs from the one this run reads data under."""
    calculated = index_result.modelling_assumptions
    simulated = data_treatment_of(data_provider)

    if calculated is None or simulated is None:
        return

    differ = [f"{name} {calculated_value!r} for the index, "
              f"{simulated.data_treatment()[name]!r} for the backtest"
              for name, calculated_value in calculated.data_treatment().items()
              if calculated_value != simulated.data_treatment()[name]]

    if differ:
        logger.warning("The index was calculated under different modelling "
                       "assumptions from this backtest: %s.", "; ".join(differ))


def refuse_screens_as_modifiers(modifiers: list[BacktestModifier]) -> None:
    """Say where a screen goes, if one was passed as a modifier."""
    screens = [type(modifier).__name__ for modifier in modifiers
               if isinstance(modifier, Screen)]

    if screens:
        raise TypeError(f"{', '.join(screens)} is a screen, not a modifier: "
                        f"pass it as Implementation(screens=[...]).")
