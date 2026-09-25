# src/beacon/data/free_float.py
"""The free float in force on a date, and the one refusal when there is none.

Every float-adjusted read (the weighting, the market values and the
special-dividend divisor) resolves a missing value the same way. Free float
moves on corporate events and index-provider reviews, not daily, so a value a
few weeks old is still the right number, and the last known value carries
forward within a window:

- **Global**, set as `free_float_backfill_days` on the `DataFetcher` beside
  `fx_policy` and `max_price_staleness_days`. It changes numbers, so it is one
  setting for the installation and published on `/health`, rather than a
  default buried at each read.
- **Default 90 days.** A quarterly review cycle fits inside it with room to
  spare. 0 turns carrying off, so only a value dated that day is used.
- **Forward only.** A value dated after the date is never used.
- **Beyond the window, refuse.** Everywhere, including the dividend path. A
  float last seen a year ago is not the float.
"""
# BN-219. A float-adjusted index read the free float in three places, and
# they disagreed about a blank cell. The weighting and the market values
# refused, because weighting one name by its full market cap puts it on a
# different basis from the rest. The special-dividend path used the value if
# it was there and otherwise skipped the adjustment, so the dividend came off
# at full size. Refusing on a blank cell was also stricter than the data
# warrants, hence the window. Before it, every read behaved as 0 does now.
# The lookup goes through `as_of_position`, so BN-208's look-ahead guard
# covers it.
from typing import Any

import pandas as pd

from ..exceptions import CalculationError
from .base import as_of_position

# How many calendar days a free float carries forward over missing values
# when nobody chooses. Calendar rather than trading days, like the price
# staleness window: it measures how old the observation is, and a holiday
# does not make a number fresher.
DEFAULT_FREE_FLOAT_BACKFILL_DAYS = 90


def validated_window(days: int) -> int:
    """Check a backfill window, returning it unchanged.

    Raises:
        ValueError: If it is negative or not an integer. There is no
            "unlimited": a float last seen a year ago is not the float.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days < 0:
        raise ValueError(
            f"free_float_backfill_days must be a whole number of days, 0 or "
            f"more, got {days!r}. 0 uses only a value dated that day.")

    return days


def carried_forward(history: pd.Series,
                    date: pd.Timestamp,
                    window_days: int) -> float | None:
    """The last value on or before *date*, if it is no older than the window.

    Args:
        history: One name's observed free floats, indexed by date, sorted,
            with the missing cells already dropped.
        date: The date the free float is wanted on.
        window_days: How many calendar days old the value may be.

    Returns:
        float | None: The value in force, or None when there is none on or
        before *date*, or the latest one is older than the window.
    """
    position = as_of_position(history.index, date)

    if position is None:
        return None

    if (date - history.index[position]).days > window_days:
        return None

    return float(history.iloc[position])


def require_free_float(provider: Any,
                       identifier: str,
                       date: str,
                       calculation_name: str) -> float:
    """The free float to scale *identifier*'s market cap by, or a refusal.

    Every float-adjusted read goes through here (the weighting, the market
    values and the special-dividend divisor), so the three give one answer
    to a missing value and say the same thing when they refuse.

    Takes the provider as an argument rather than living on the fetcher, so a
    test double that answers `fetch_free_float_factor` gets the same rule.

    Args:
        provider: Anything with `fetch_free_float_factor`, which applies the
            backfill window.
        identifier: The instrument.
        date: The date wanted, YYYY-MM-DD.
        calculation_name: What is asking, for the error.

    Raises:
        CalculationError: If there is no value within the window, or the value
            is outside 0-1. Using the full market cap instead would weight
            the name as though every share were freely traded, which is a
            different index from the one specified.
    """
    factor = provider.fetch_free_float_factor(identifier, date)

    if factor is not None and 0.0 <= factor <= 1.0:
        return float(factor)

    window = getattr(provider, "free_float_backfill_days", None)
    looked = (f"on or within {window} days before {date}"
              if isinstance(window, int) and window > 0 else f"on {date}")
    found = ("none was found" if factor is None
             else f"the value found, {factor!r}, is not between 0 and 1")

    raise CalculationError(
        calculation_name=calculation_name,
        details=(f"{identifier} needs a free float {looked}, because this "
                 f"index is free-float adjusted, and {found}. Using its full "
                 f"market cap instead would weight it as though every share "
                 f"were freely traded, which is a different index from the "
                 f"one specified. Load the value, or widen "
                 f"free_float_backfill_days if an older one should count."))
