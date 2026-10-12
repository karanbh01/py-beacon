# src/beacon/synthetic/fx.py
"""
Exchange rates, generated as market data so the FX paths are actually exercised.

A pair is an ordinary market series, its rate in ``CLOSE``, and a reference
row says what it is: ``ASSET_CLASS`` ``Currency``, ``SECURITY_TYPE`` ``Spot``
and its two currencies (see `beacon.data.kinds`). `EURUSD` converts euros
into dollars. Nothing else is special about it, which is why the calculator
and the corporate-action handler can convert without knowing where the rate
came from.

## Why a rate is not a price

Two differences, and both matter to what the data is for.

**Volatility is far lower.** A major pair realises 7-11% annualised against
25-35% for a single equity. A generator that gave currencies equity-like
volatility would make every unhedged exposure look like the dominant risk in a
global portfolio, which is the opposite of true.

**Not everything floats.** The Hong Kong dollar runs inside a band the
monetary authority defends, and realises well under 1%. Modelling it like the
others would manufacture a diversification benefit that does not exist, and an
optimiser told it can hedge HKD risk would allocate to a trade nobody makes.

## Flight to quality

Crises are not currency-neutral. Money moves into dollars when volatility
spikes, so every pair here drifts *down* against USD in proportion to how far
the regime has lifted correlations. It is the same intensity series the return
process uses, so the currency move lines up with the equity drawdown rather
than wandering off on its own, which is what makes a hedged-versus-unhedged
comparison over a crisis show anything at all.
"""
import logging

import numpy as np
import pandas as pd

from . import regions
from .regimes import CRISES, Regime, market_multipliers
from .returns import BURN_IN, TRADING_DAYS, simulate_gjr, standardised_t

logger = logging.getLogger(__name__)

# How far a currency falls against the dollar at full crisis intensity,
# annualised. Modest on purpose: the dollar gained about 20% through the
# second half of 2008 against a basket, over a window where equities fell 40%,
# so FX is a real effect and a second-order one.
FLIGHT_TO_QUALITY = 0.18

# A pegged currency does not participate. The peg is what the authority is
# defending, and it holds through exactly the episodes this would otherwise
# move it in.
PEG_VOLATILITY = 0.01

# A pair's rate is its close, like any other series (BN-308).
CLOSE_COLUMN = "CLOSE"


def build(dates: pd.DatetimeIndex,
          rng: np.random.Generator,
          regimes: tuple[Regime, ...] = CRISES,
          start_from: dict[str, float] | None = None) -> pd.DataFrame:
    """Generate one rate series per non-base currency.

    Args:
        dates: The panel's business days.
        rng: Seeded generator.
        regimes: Dated episodes, for the flight-to-quality drift.
        start_from: The rate each pair starts from, where known. An
            extension passes the last stored rates, so each path carries on
            from where it stopped rather than from the pair's usual level.

    Returns:
        pd.DataFrame: Long-form, with IDENTIFIER/DATE and the rate in CLOSE,
        ready to be concatenated onto the equity market data.
    """
    pairs = regions.pairs()

    if not pairs:
        return pd.DataFrame(columns=["DATE", "IDENTIFIER", CLOSE_COLUMN])

    _scale, _drift, lift = market_multipliers(dates, regimes)
    steps = len(dates)

    frames = []
    for identifier, initial, volatility in pairs:
        rate = _path(rng, steps, volatility, lift,
                     (start_from or {}).get(identifier, initial))

        frames.append(pd.DataFrame({"DATE": dates,
                                    "IDENTIFIER": identifier,
                                    CLOSE_COLUMN: rate}))

    logger.info("Generated %d FX pair(s): %s.",
                len(frames), ", ".join(pair for pair, _, _ in pairs))

    return pd.concat(frames, ignore_index=True)


def reference(valid_from: str) -> pd.DataFrame:
    """A reference row for each pair `build` generates, valid from
    *valid_from*: what makes the series a currency pair."""
    rows = [{"IDENTIFIER": identifier,
             "DATE_FROM": pd.Timestamp(valid_from),
             "NAME": f"{base}/{quote}",
             "CURRENCY": quote,
             "ASSET_CLASS": "Currency",
             "SECURITY_TYPE": "Spot",
             "BASE_CURRENCY": base,
             "QUOTE_CURRENCY": quote}
            for identifier, base, quote in regions.pair_currencies()]

    return pd.DataFrame(rows)


def _path(rng: np.random.Generator,
          steps: int,
          volatility: float,
          lift: np.ndarray,
          initial: float) -> np.ndarray:
    """One rate path: GARCH shocks, a crisis drift, compounded from `initial`.

    The same GJR recursion the equities use, so exchange rates cluster their
    volatility too -- which they do, and which a plain random walk would miss.
    """
    daily_variance = np.array([volatility ** 2 / TRADING_DAYS])
    persistence, degrees = np.array([0.96]), np.array([6.0])

    innovations = standardised_t(rng, degrees, (steps + BURN_IN, 1))
    shocks = simulate_gjr(steps, daily_variance, persistence, innovations)[:, 0]

    # A peg is defended, so it neither drifts in a crisis nor compounds a
    # trend; only the small band movement survives.
    if volatility <= PEG_VOLATILITY:
        return np.asarray(initial * np.exp(np.cumsum(shocks)))

    drift = -lift * FLIGHT_TO_QUALITY / TRADING_DAYS

    return np.asarray(initial * np.exp(np.cumsum(shocks + drift)))
