# src/beacon/data/adjustment.py
"""
Back-adjusting a price series for corporate actions.

    adjusted = adjust_closes(closes, actions.get("CMPA"))

A raw close series has a step in it wherever a split happened, and it
understates what a holder made wherever a dividend was paid. `ADJ_CLOSE`
removes both.

## What `ADJ_CLOSE` means here

**Splits and dividends**, which is the vendor convention — Yahoo and Bloomberg
both ship a column of this name that reinvests cash. It is what most people
mean when they say "adjusted", and shipping a splits-only series under the
same name would give a number that looks right and is not.

The consequence is worth stating: **an adjusted series is no longer a price**.
It answers "what would a holder have made", so its level is not what anything
traded at on that day. A chart of it is a total-return chart.

## Adjusted backwards from the latest date

The most recent close equals the raw close, and only history moves. Every
vendor does it this way and the reason is practical: the right-hand edge of a
chart matches the quote a user can see elsewhere, so the series is checkable.
Adjusting forwards from the start would make the *last* number unrecognisable,
which is the one people look at.

It also means the series changes when a new action lands — every historical
value shifts. That is inherent to the convention rather than a defect, and a
client caching an adjusted series needs to know it is not immutable.

## Missing prices, and why a dividend needs one

A cash adjustment is a *fraction* of the price the day before the ex-date:
paying 0.75 out of a 220 stock is a different event from paying it out of a 5
stock. So a dividend whose preceding close is missing cannot be applied, and
is skipped with a warning rather than guessed at — a guessed factor silently
misstates every earlier value in the series.
"""
import logging

import numpy as np
import pandas as pd

from .corporate_actions import CASH_ACTIONS, RATIO_ACTIONS

logger = logging.getLogger(__name__)

ADJUSTED_COLUMN = "ADJ_CLOSE"


def adjust_closes(closes: pd.Series,
                  actions: pd.DataFrame) -> pd.Series:
    """Back-adjust a close series for splits and cash.

    Args:
        closes: Date-indexed closes, ascending.
        actions: This instrument's actions, with `EX_DATE`, `TYPE`, `VALUE`,
            as `CorporateActions.get` returns.

    Returns:
        pd.Series: Adjusted closes on the same index. The last value equals
        the last raw close; earlier ones are scaled by every action after
        them.
    """
    if closes.empty:
        return closes

    factors = _factors(closes, actions)

    # The factor for a date is the product of every action *after* it, so the
    # last date is unadjusted. Reversed cumulative product, shifted by one so
    # an action on date t does not adjust t itself -- the ex-date price has
    # already dropped.
    cumulative = factors[::-1].cumprod()[::-1].shift(-1).fillna(1.0)

    return closes * cumulative


def _factors(closes: pd.Series,
             actions: pd.DataFrame) -> pd.Series:
    """One multiplier per date, 1.0 where nothing happened."""
    factors = pd.Series(1.0, index=closes.index)

    if actions is None or actions.empty:
        return factors

    for _, action in actions.iterrows():
        ex_date = pd.Timestamp(action["EX_DATE"])
        action_type = str(action["TYPE"])

        if action_type in RATIO_ACTIONS:
            factor = _ratio_factor(action)
        elif action_type in CASH_ACTIONS:
            factor = _cash_factor(closes, ex_date, action)
        else:
            # A structural action -- a merger, a spin-off -- is not a scaling
            # of the same instrument's price, so there is no factor that makes
            # the series continuous. Left alone rather than approximated.
            continue

        if factor is None:
            continue

        position = closes.index.searchsorted(ex_date)

        if position >= len(closes):
            # After the series ends: nothing before it to adjust.
            continue

        factors.iloc[position] *= factor

    return factors


def _ratio_factor(action: pd.Series) -> float | None:
    """A split's multiplier: prices before it are divided by the ratio."""
    ratio = float(action["VALUE"])

    if ratio <= 0:
        logger.warning("Ignoring a non-positive split ratio (%s); it would "
                       "erase or invert the series.", ratio)

        return None

    return 1.0 / ratio


def _cash_factor(closes: pd.Series,
                 ex_date: pd.Timestamp,
                 action: pd.Series) -> float | None:
    """A dividend's multiplier: what fraction of the price was retained.

    Measured against the close *before* the ex-date, which is the price the
    cash came out of. Using the ex-date close instead would divide by a number
    that has already fallen by the dividend, overstating the adjustment.
    """
    amount = float(action["VALUE"])

    if not np.isfinite(amount) or amount <= 0:
        return None

    before = closes.loc[closes.index < ex_date]

    if before.empty:
        return None

    previous = before.iloc[-1]

    if not np.isfinite(previous) or previous <= 0:
        logger.warning("No usable close before %s, so a cash action of %s "
                       "cannot be applied as a fraction of price; the series "
                       "is left unadjusted for it.",
                       ex_date.date(), amount)

        return None

    if amount >= previous:
        logger.warning("A cash action of %s on %s exceeds the preceding close "
                       "of %s; skipped rather than driving the series to zero "
                       "or negative.", amount, ex_date.date(), previous)

        return None

    return float((previous - amount) / previous)
