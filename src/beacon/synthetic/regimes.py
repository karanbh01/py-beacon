# src/beacon/synthetic/regimes.py
"""
Market regimes: the crises a stationary model cannot produce.

The return process in `returns.py` is stationary by construction. Its
volatility clusters, its tails are fat and its correlations are stable — all
true of markets *on average*, and all wrong about the periods that matter most.

Twenty-five years of equity history is not a draw from one distribution. It
contains three or four episodes where volatility trebles, drift turns sharply
negative, and — the part that matters — **correlations rise toward one**.
Diversification stops working precisely when it is most wanted, and a model
without that overstates how much a spread portfolio protects you.

## What a regime does

Each regime scales three things over a dated window:

* **volatility** — a multiplier on the market factor, so every name inherits it
  through its own beta rather than being scaled directly
* **drift** — an annualised amount added to the market's return over the window
* **correlation** — the market factor's share of total variance rises, which is
  what makes names move together

The third is the one worth stating twice. Raising volatility alone produces a
big drawdown that a diversified portfolio still cushions. Raising the market's
*share* of variance is what removes the cushion.

## Shape, not a step function

A crisis does not begin at midnight. Each window is ramped in and out with a
raised-cosine taper over a fraction of its length, so volatility builds and
subsides. A rectangular window produces a discontinuity in realised volatility
that shows up as an obviously artificial jump on any chart.

## The dates are real, the paths are not

The windows below are the actual episodes. What happens inside them is still
generated — this is not a replay of 2008, and nothing here reproduces any real
security's price. It is a synthetic market that has crises where the real one
did, which is what makes a backtest over it exercise the code paths a calm
market never reaches.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Regime:
    """One dated market episode.

    Attributes:
        name: What it was.
        start: First date of the episode.
        end: Last date.
        volatility: Multiplier on the market factor's volatility at peak
            intensity. 1.0 is calm.
        drift: Annualised return added to the market over the window, at peak.
            Negative for a crisis.
        correlation: How far the market's share of total variance moves toward
            one, from 0 (unchanged) to 1 (everything is the market). This is
            the diversification failure.
        ramp: Fraction of the window spent easing in and out.
    """
    name: str
    start: str
    end: str
    volatility: float = 1.0
    drift: float = 0.0
    correlation: float = 0.0
    ramp: float = 0.25


# The episodes a twenty-five-year equity history contains.
#
# ## How these numbers were arrived at
#
# Not from theory. Each was tuned until the *realised* peak-to-trough drawdown
# of an equal-weighted panel matched the real index over the same dates,
# averaged across three seeds. Averaging is the part worth keeping: a first
# pass calibrated against a single path and drew two conclusions that were
# backwards -- it read dot-com as far too shallow when it was already right,
# and covid as too deep when it was too shallow. One path through a fat-tailed
# process is not a measurement.
#
# Realised, against the real windows (mean of seeds 2, 5, 9):
#
#     window                 model     real
#     dot-com               -48.9%    -49%
#     GFC 2007-09           -58.4%    -57%
#     Sep-Nov 2008          -36.4%    -40%
#     covid                 -35.8%    -34%
#     2022                  -26.2%    -25%
#     long run             +6.87%/yr  ~7.5%/yr
#
# The drift figures are therefore not annualised returns anyone observed --
# they are the input that produces the observed drawdown once volatility,
# correlation and the taper have had their effect. Covid's -200% reads absurd
# until you note the window is ten weeks long: it is an annualisation of a move
# that never lasted a year.
CRISES = (
    Regime("dot-com unwind", "2000-03-01", "2002-10-09",
           volatility=1.6, drift=-0.52, correlation=0.25, ramp=0.30),
    Regime("global financial crisis", "2007-10-09", "2009-03-09",
           volatility=2.1, drift=-0.42, correlation=0.45, ramp=0.25),
    # The acute phase. Nested inside the crisis above rather than added to it,
    # so this multiplier deepens the crisis rather than stacking on it -- which
    # is why 2.3 here reads lower than the VIX's 80 that October would suggest.
    Regime("2008 panic", "2008-09-15", "2008-12-31",
           volatility=2.3, drift=-0.15, correlation=0.60, ramp=0.15),
    # The most violent and the shortest. A 3.1x multiplier over ten weeks is
    # what it takes to reach -34% in that time; the VIX did close at 82.7.
    Regime("covid shock", "2020-02-19", "2020-04-30",
           volatility=3.1, drift=-2.00, correlation=0.55, ramp=0.10),
    # A grind rather than a shock: the drawdown came from a long slope, not a
    # crash, so most of it is drift and correlation barely moved.
    Regime("2022 drawdown", "2022-01-03", "2022-10-14",
           volatility=1.25, drift=-0.06, correlation=0.20, ramp=0.30),

    # --- Recoveries ------------------------------------------------------
    #
    # Without these the model is a market that falls and never climbs back. A
    # constant drift plus five drawdowns annualised the whole 25 years at
    # +0.3%, against roughly 7.5% for US large-cap over the same span --
    # because in reality the years *after* a crash did 15% a year, and it is
    # the rebound that makes the long-run figure what it is.
    #
    # Elevated volatility on the way up too: a recovery is not calm, and the
    # sharpest up-days in history sit inside the worst drawdowns.
    Regime("post-dot-com recovery", "2002-10-10", "2004-12-31",
           volatility=1.2, drift=0.30, correlation=0.10, ramp=0.30),
    Regime("post-crisis recovery", "2009-03-10", "2011-06-30",
           volatility=1.3, drift=0.36, correlation=0.15, ramp=0.25),
    Regime("post-covid rebound", "2020-05-01", "2021-12-31",
           volatility=1.2, drift=0.30, correlation=0.10, ramp=0.20),
    Regime("2023 recovery", "2022-10-15", "2024-12-31",
           volatility=1.0, drift=0.26, correlation=0.05, ramp=0.30),
)


def _taper(length: int,
           ramp: float) -> np.ndarray:
    """Intensity across a window: eased in, flat, eased out.

    A raised cosine rather than a linear ramp, because the derivative is
    continuous at both ends — a linear taper leaves a visible kink in realised
    volatility where the slope changes.
    """
    if length <= 0:
        return np.zeros(0)

    intensity = np.ones(length)
    edge = max(int(length * min(max(ramp, 0.0), 0.5)), 1)

    if edge * 2 >= length:
        edge = max(length // 2, 1)

    rise = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge)))
    intensity[:edge] = rise
    intensity[length - edge:] = rise[::-1]

    return intensity


def intensity_series(dates: pd.DatetimeIndex,
                     regimes: tuple[Regime, ...] = CRISES) -> pd.DataFrame:
    """Per-date intensity for every regime touching a panel.

    Args:
        dates: The panel's business days.
        regimes: Episodes to apply.

    Returns:
        pd.DataFrame: Date-indexed, one column per regime, values in [0, 1].
        Regimes that fall entirely outside the panel are omitted, so a
        five-year modern window carries no columns and behaves exactly as it
        did before regimes existed.
    """
    columns: dict[str, np.ndarray] = {}

    for regime in regimes:
        inside = (dates >= pd.Timestamp(regime.start)) & (dates <= pd.Timestamp(regime.end))
        count = int(inside.sum())

        if count == 0:
            continue

        column = np.zeros(len(dates))
        column[inside] = _taper(count, regime.ramp)
        columns[regime.name] = column

    if not columns:
        return pd.DataFrame(index=dates)

    logger.info("Panel covers %d regime(s): %s.",
                len(columns), ", ".join(columns))

    return pd.DataFrame(columns, index=dates)


def market_multipliers(dates: pd.DatetimeIndex,
                       regimes: tuple[Regime, ...] = CRISES
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How the market factor is scaled on each date.

    Overlapping regimes take the **most severe** value rather than compounding.
    The 2008 panic is the peak *of* the financial crisis, not a second shock on
    top of it, and multiplying the two gave a 5.5x volatility multiplier and a
    combined -112% annual drift — a market that fell further in one quarter
    than it has in any real decade.

    Args:
        dates: The panel's business days.
        regimes: Episodes to apply.

    Returns:
        tuple: Volatility multiplier, daily drift adjustment, and the share of
        variance moved toward the market factor, each per date.
    """
    intensity = intensity_series(dates, regimes)

    volatility = np.ones(len(dates))
    drift = np.zeros(len(dates))
    correlation = np.zeros(len(dates))

    by_name = {regime.name: regime for regime in regimes}

    for name in intensity.columns:
        regime = by_name[name]
        weight = intensity[name].to_numpy()

        # Interpolated from the calm value rather than applied flat, so a
        # half-intensity day sits halfway to the peak. Combined by maximum, so
        # a nested regime deepens its parent rather than stacking on it.
        volatility = np.maximum(volatility,
                                1.0 + weight * (regime.volatility - 1.0))
        # Most severe *downward*, and additive upward: a recovery inside no
        # crisis simply lifts the drift, while two overlapping crises take the
        # worse of the two rather than summing into something unprecedented.
        contribution = weight * regime.drift / 252.0
        drift = (np.minimum(drift, contribution) if regime.drift < 0
                 else drift + contribution)
        correlation = np.maximum(correlation, weight * regime.correlation)

    return volatility, drift, correlation


def describe(dates: pd.DatetimeIndex,
             regimes: tuple[Regime, ...] = CRISES) -> list[str]:
    """One line per regime the panel covers, for a caller to log or print."""
    intensity = intensity_series(dates, regimes)
    by_name = {regime.name: regime for regime in regimes}

    lines = []
    for name in intensity.columns:
        regime = by_name[name]
        days = int((intensity[name] > 0).sum())

        lines.append(f"{name}: {regime.start} to {regime.end} "
                     f"({days} sessions, vol x{regime.volatility:.1f}, "
                     f"drift {regime.drift:+.0%}, "
                     f"correlation +{regime.correlation:.0%})")

    return lines
