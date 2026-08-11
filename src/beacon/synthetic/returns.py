# src/beacon/synthetic/returns.py
"""
The return process: a factor model with GJR-GARCH volatility and fat tails.

Gaussian random walks are the obvious way to make fake prices and they are
wrong in every way that matters to what this data is for. They have no
volatility clustering, so a drawdown chart shows nothing recognisable; no fat
tails, so a risk model estimated from them is never stressed; and no
cross-correlation, so an optimiser sees a diversification opportunity that no
market offers and every constraint binds strangely. Each stylized fact below
is here because leaving it out breaks a view somebody has to look at.

## The model

For name *i* on day *t*:

    r[i,t] = mu[i] + b[i]·f_market[t] + g[i]·f_sector(i)[t]
                   + h[i]·f_region(i)[t] + e[i,t]

Three independent sources, each a GJR-GARCH(1,1) process with standardised
Student-t innovations:

* **The market factor** — one series everything loads on. This is what makes
  names co-move, and giving it its own GARCH is what makes them co-move
  *more in a crisis*, which is when correlation matters.
* **Sector factors** — one per GICS sector, so two banks resemble each other
  more than a bank resembles a utility.
* **Region factors** — one per listing venue, so two names listed in Tokyo
  move together for reasons that have nothing to do with their industry.
* **Idiosyncratic noise** — per name.

Loadings are set from a variance budget rather than drawn directly, so a name's
total volatility is a target that is hit rather than an outcome to be
discovered. With the market at ~34% of variance, the sector at ~16% and the
region at ~8%, same-sector pairs correlate near 0.50 and cross-sector pairs
near 0.35, and the average across the universe lands near 0.39.

## Why GJR rather than plain GARCH

    sigma2[t] = omega + (alpha + lam·1[e<0])·e[t-1]^2 + beta·sigma2[t-1]

The `lam` term makes a fall raise tomorrow's volatility more than a rise of the
same size does. That asymmetry is the leverage effect, and it is what produces
negative skew — without it, simulated returns are symmetric and a drawdown
looks like an upswing turned upside down.

Persistence is `alpha + lam/2 + beta`, drawn in 0.94-0.99: high enough that
quiet and turbulent periods last for months rather than days, below one so the
process is stationary and the unconditional variance is defined.

## What is deterministic here, and what is not

Given a seed, this module produces the same numbers on the same machine and
the same numpy. It does *not* guarantee bit-identical output across operating
systems: `standard_t` and the exponentials behind it run through the platform's
libm, which is free to differ in the last bit. That is a deliberate limit —
avoiding transcendentals entirely would mean abandoning Student-t innovations
and log-normal volume, which is most of what makes this data worth generating.
Reproducibility is tested per-platform; the statistical acceptance checks are
tolerance-based and hold anywhere.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regimes import CRISES, Regime, market_multipliers

TRADING_DAYS = 252

# How many names are simulated at a time.
#
# The unblocked version held roughly ten full ``(days x names)`` arrays alive
# at once -- innovations, the variance and shock paths, the lifted shares, the
# market component -- and so peaked at about seven times the size of the panel
# it returned. Blocking bounds that at the block rather than the universe, and
# the returned panel becomes the floor rather than a tenth of the total.
#
# There is no speed/memory trade-off to balance here, which was not the
# expectation. Blocking narrows the arrays the GARCH recursion vectorises
# over, so it looked like it should cost time; measured over 1,600 names and
# ten years it *saves* it, presumably on cache locality:
#
#     block   100    250    500   1000   5000 (unblocked)
#     time   0.75s  0.63s  0.66s  0.80s  0.89s
#
# So this is chosen on memory alone. The intermediates run about 145 KB per
# name per decade, putting a 250-name block near 36 MB.
BLOCK_SIZE = 250

# Annualised volatility of the market factor itself. Loadings are scaled
# against this, so a name's `b` is its CAPM beta rather than an arbitrary
# coefficient — which matters because the drift below is a CAPM expectation.
MARKET_VOLATILITY = 0.16

# GJR persistence, drawn per series. The issue's band: below 0.94 the clustering
# is too short-lived to see on a chart, at 1.0 the variance is undefined.
MIN_PERSISTENCE = 0.94
MAX_PERSISTENCE = 0.98

# How persistence splits between the ARCH terms and the GARCH term. Beta takes
# most of it — that is what makes volatility decay slowly rather than spike and
# reset — and the asymmetric term is about half the symmetric one, which is the
# usual finding on equity indices. The ARCH share is what drives the
# autocorrelation of squared returns, so it is the dial to turn if the
# clustering signature comes out too faint to detect.
ARCH_FRACTION = 0.08
ASYMMETRY_FRACTION = 0.11

# Two-piece scaling applied to the innovations: negative draws are stretched
# and positive ones compressed. GJR alone does *not* deliver negative skew —
# with symmetric innovations the variance path is independent of the sign of
# today's draw, so E[e^3] stays zero however asymmetric the volatility
# response is. GJR gives the leverage effect proper (a fall raises tomorrow's
# volatility more than a rise does); this gives the negative skew that usually
# accompanies it, and the two are separate facts.
SKEW_ASYMMETRY = 0.28

# Student-t degrees of freedom. Four to six is the range fitted to daily equity
# returns; below four the variance of the innovation is undefined.
MIN_DEGREES_OF_FREEDOM = 4.5
MAX_DEGREES_OF_FREEDOM = 6.0

# Discarded so the process starts from its own stationary distribution rather
# than from the unconditional variance it was seeded with.
BURN_IN = 260


@dataclass(frozen=True)
class _Shared:
    """Everything a block needs that does not depend on which names it holds.

    All of it is per-date rather than per-name, so it costs a few kilobytes
    and is computed once for the whole universe. It has to be: the factors are
    what make names co-move, and a block that drew its own would produce a
    universe correlated only within blocks.
    """
    market: np.ndarray
    sector_factors: np.ndarray
    sector_names: list[str]
    region_factors: np.ndarray
    region_names: list[str]
    market_variance: float
    scale: np.ndarray
    extra_drift: np.ndarray
    lift: np.ndarray
    risk_free_rate: float
    equity_premium: float


def standardised_t(rng: np.random.Generator,
                    degrees: np.ndarray | float,
                    shape: tuple[int, ...]) -> np.ndarray:
    """Negatively skewed Student-t draws, rescaled to unit variance.

    A raw t has variance ``nu/(nu-2)``, so feeding it into a GARCH recursion
    unscaled would inflate every variance by that factor and make the target
    volatility wrong by 40% at four degrees of freedom.

    The two-piece step stretches the downside and compresses the upside, then
    the whole thing is re-standardised — the stretch changes both the mean and
    the variance, and leaving either uncorrected would show up as a spurious
    drift and a missed volatility target.
    """
    raw = rng.standard_t(degrees, size=shape) / np.sqrt(degrees / (degrees - 2.0))

    skewed = np.where(raw < 0.0,
                      raw * (1.0 + SKEW_ASYMMETRY),
                      raw * (1.0 - SKEW_ASYMMETRY))

    # `np.asarray` is for mypy, not for numpy: on the numpy that CI resolves,
    # ndarray arithmetic types as Any, and a bare return trips --strict's
    # no-any-return. It is a no-op at runtime on an array that already is one.
    return np.asarray((skewed - skewed.mean(axis=0)) / skewed.std(axis=0))


def simulate_gjr(steps: int,
                 target_variance: np.ndarray,
                 persistence: np.ndarray,
                 innovations: np.ndarray) -> np.ndarray:
    """Run the GJR-GARCH(1,1) recursion over pre-drawn innovations.

    Vectorised across series and looped over time, which is the only way round:
    each day's variance depends on the day before, but every series can take
    its step together.

    The innovations are an argument rather than drawn here, and that is what
    makes block generation possible. Every series in the recursion is
    independent of every other — the arithmetic is elementwise throughout — so
    running it over a slice of the universe gives bit-identical results to
    running it over all of it, provided each series sees the same innovations.

    Args:
        steps: Days to return, after burn-in.
        target_variance: Unconditional daily variance per series.
        persistence: ``alpha + lam/2 + beta`` per series, strictly below 1.
        innovations: Shape ``(steps + BURN_IN, len(target_variance))``,
            standardised to unit variance.

    Returns:
        np.ndarray: Shape ``(steps, len(target_variance))``, mean zero, with
        unconditional variance equal to ``target_variance``.
    """
    count = len(target_variance)
    total = steps + BURN_IN

    alpha = persistence * ARCH_FRACTION
    asymmetry = persistence * ASYMMETRY_FRACTION
    beta = persistence - alpha - asymmetry / 2.0

    # From E[sigma2] = omega / (1 - persistence), which is what makes the
    # target a target rather than a hope.
    omega = target_variance * (1.0 - persistence)

    variance = np.empty((total, count))
    shocks = np.empty((total, count))

    variance[0] = target_variance
    shocks[0] = np.sqrt(variance[0]) * innovations[0]

    for step in range(1, total):
        previous = shocks[step - 1]
        leverage = alpha + asymmetry * (previous < 0.0)

        variance[step] = (omega + leverage * previous ** 2
                          + beta * variance[step - 1])
        shocks[step] = np.sqrt(variance[step]) * innovations[step]

    return shocks[BURN_IN:]


def pin_realised_variance(series: np.ndarray,
                          target_variance: np.ndarray) -> np.ndarray:
    """Rescale a factor so its *realised* variance equals its target.

    Applied to the shared factors only, and it is not cosmetic. A GARCH series
    at persistence 0.98 with t(4.5) innovations has enormous dispersion in its
    sample variance: over five years one draw can realise nearly twice its
    unconditional level. For an idiosyncratic series that is a fact about one
    name. For the market factor — which every name loads on — it rescales the
    entire universe, and a generated dataset comes out with every volatility
    at 43% and an average correlation of 0.73 instead of 29% and 0.40. That
    happened on seed 7 and would have shipped as "the seed you get".

    Only the overall level is pinned. The clustering, the fat tails and the
    leverage asymmetry are properties of the *shape* of the path and survive a
    single multiplicative rescaling untouched.

    Idiosyncratic series are deliberately left alone, so a name's realised
    volatility still varies around its target the way a real one does.
    """
    return np.asarray(series * np.sqrt(target_variance / series.var(axis=0)))


def _draw_parameters(rng: np.random.Generator,
                     count: int) -> tuple[np.ndarray, np.ndarray]:
    """Persistence and degrees of freedom for `count` series."""
    persistence = rng.uniform(MIN_PERSISTENCE, MAX_PERSISTENCE, size=count)
    degrees = rng.uniform(MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM,
                          size=count)

    return persistence, degrees


def _shared_factor(rng: np.random.Generator,
                   steps: int,
                   target_variance: np.ndarray) -> np.ndarray:
    """A factor every name loads on, pinned to its realised variance.

    Drawn whole rather than in blocks: these are one column per *sector* at
    most, so they cost nothing to hold, and every block has to see the same
    ones or the correlation structure would differ across the universe.
    """
    persistence, degrees = _draw_parameters(rng, len(target_variance))
    innovations = standardised_t(rng, degrees,
                                  (steps + BURN_IN, len(target_variance)))

    return pin_realised_variance(
        simulate_gjr(steps, target_variance, persistence, innovations),
        target_variance)


def _idiosyncratic(children: list[np.random.Generator],
                   target_variance: np.ndarray,
                   steps: int) -> np.ndarray:
    """The per-name GARCH series for one block of names.

    Every draw belonging to a name comes from that name's **own** generator —
    its persistence, its degrees of freedom and its innovations. That is what
    makes the block size an honest memory knob: name *i* gets the same path
    whichever block it lands in, and whatever the block size, so tuning it for
    memory cannot silently produce a different market.

    Drawing a column at a time costs nothing measurable. At 2,000 names over
    ten years, per-name draws plus the spawn plus stacking ran 0.24s against
    0.20s for a single bulk draw.
    """
    persistence = np.array([child.uniform(MIN_PERSISTENCE, MAX_PERSISTENCE)
                            for child in children])
    degrees = np.array([child.uniform(MIN_DEGREES_OF_FREEDOM,
                                      MAX_DEGREES_OF_FREEDOM)
                        for child in children])

    innovations = np.stack(
        [standardised_t(child, degree, (steps + BURN_IN,))
         for child, degree in zip(children, degrees, strict=True)], axis=1)

    return simulate_gjr(steps, target_variance, persistence, innovations)


def simulate(universe: pd.DataFrame,
             dates: pd.DatetimeIndex,
             rng: np.random.Generator,
             risk_free_rate: float,
             equity_premium: float,
             regimes: tuple[Regime, ...] = CRISES,
             block_size: int = BLOCK_SIZE) -> pd.DataFrame:
    """Simulate the total-return panel.

    Args:
        universe: Output of `universe.build`, carrying the volatility target
            and variance shares each loading is derived from.
        dates: Business days to simulate.
        rng: Seeded generator.
        risk_free_rate: Annualised, the base of the CAPM expectation.
        equity_premium: Annualised excess return on a beta-one name.
        regimes: Dated crisis episodes to overlay. Empty for a stationary
            market, which is what every panel produced before regimes existed.
        block_size: How many names to simulate at a time. Affects peak memory
            and nothing else — the panel is identical at any block size, and a
            test holds that.

    Returns:
        pd.DataFrame: Date-indexed daily total returns, one column per name.
    """
    steps = len(dates)
    count = len(universe)

    sector_names = sorted(set(universe["SECTOR"].to_numpy()))
    region_names = sorted(set(universe["REGION"].to_numpy()))
    market_variance = (MARKET_VOLATILITY ** 2) / TRADING_DAYS

    market = _shared_factor(rng, steps, np.array([market_variance]))
    sector_factors = _shared_factor(
        rng, steps, np.full(len(sector_names), market_variance))

    # A third shared factor, on the same footing as the sector one. Two names
    # listed in Tokyo move together for reasons that have nothing to do with
    # being in the same industry, and without this a "global" index is eleven
    # sectors of names that happen to be labelled with different countries.
    region_factors = _shared_factor(
        rng, steps, np.full(len(region_names), market_variance))

    # --- Regime overlay --------------------------------------------------
    #
    # Two separate effects, and conflating them is the trap. The volatility
    # multiplier scales the market factor, so a crisis reaches a name through
    # its own beta and a high-beta name suffers more. The correlation lift
    # *redistributes* variance from the idiosyncratic and sector terms into the
    # market term, leaving the total alone.
    #
    # Redistribution, not addition. The first attempt kept each name's full
    # market exposure and added more on top, which inflated total variance
    # instead of moving it: realised volatility in the 2008 window came out at
    # 109% against a real figure nearer 40%, and the whole 25-year path
    # annualised at -5%. Diversification must fail *without* the market itself
    # becoming impossible.
    scale, extra_drift, lift = market_multipliers(pd.DatetimeIndex(dates),
                                                  regimes)

    shared = _Shared(market=market,
                     sector_factors=sector_factors,
                     sector_names=sector_names,
                     region_factors=region_factors,
                     region_names=region_names,
                     market_variance=market_variance,
                     scale=scale,
                     extra_drift=extra_drift,
                     lift=lift,
                     risk_free_rate=risk_free_rate,
                     equity_premium=equity_premium)

    # One generator per name, so a name's path is a function of the seed and
    # its own position -- never of how the work happened to be divided up.
    children = rng.spawn(count)

    returns = np.empty((steps, count))

    for start in range(0, count, max(block_size, 1)):
        stop = min(start + max(block_size, 1), count)

        returns[:, start:stop] = _block(universe.iloc[start:stop],
                                        children[start:stop],
                                        shared,
                                        steps)

    return pd.DataFrame(returns,
                        index=pd.DatetimeIndex(dates, name="DATE"),
                        columns=universe.index)


def _block(names: pd.DataFrame,
           children: list[np.random.Generator],
           shared: "_Shared",
           steps: int) -> np.ndarray:
    """Simulate the returns of one block of names.

    Holds ``(steps x len(names))`` intermediates rather than
    ``(steps x universe)`` ones, which is the whole point: the unblocked
    version kept about ten full-universe arrays alive at once and peaked at
    seven times the size of the panel it returned.
    """
    daily_variance = (names["volatility"].to_numpy() ** 2) / TRADING_DAYS
    market_share = names["market_share"].to_numpy()
    sector_share = names["sector_share"].to_numpy()
    region_share = names["region_share"].to_numpy()

    sector_index = np.array([shared.sector_names.index(sector)
                             for sector in names["SECTOR"].to_numpy()])
    region_index = np.array([shared.region_names.index(region)
                             for region in names["REGION"].to_numpy()])

    # Loadings from the variance budget: b^2 * market_variance is the share of
    # this name's variance the market is meant to explain, so b follows.
    market_beta = np.sqrt(market_share * daily_variance / shared.market_variance)
    sector_beta = np.sqrt(sector_share * daily_variance / shared.market_variance)
    region_beta = np.sqrt(region_share * daily_variance / shared.market_variance)

    idiosyncratic = _idiosyncratic(
        children,
        daily_variance * (1.0 - market_share - sector_share - region_share),
        steps)

    # CAPM, so a high-beta name earns more in expectation and the cross-section
    # of long-run returns is not flat.
    drift = (shared.risk_free_rate + market_beta * shared.equity_premium
             + names["alpha"].to_numpy()) / TRADING_DAYS

    # Per date and name: the market's share of variance, lifted toward one.
    lifted = (market_share[None, :]
              + shared.lift[:, None] * (1.0 - market_share[None, :]))

    # What is left over, as a fraction of what the sector and idiosyncratic
    # terms had before. At lift 0 this is exactly 1 and nothing changes.
    remaining = np.sqrt((1.0 - lifted) / (1.0 - market_share[None, :]))

    stressed_market = shared.market * shared.scale[:, None]

    # sqrt(share x variance / market_variance) is the beta that realises that
    # share, factored so the per-name part is computed once.
    per_name = np.sqrt(daily_variance / shared.market_variance)
    market_component = stressed_market * np.sqrt(lifted) * per_name[None, :]

    return np.asarray(
        drift
        + shared.extra_drift[:, None]
        + market_component
        + remaining * (sector_beta * shared.sector_factors[:, sector_index]
                       + region_beta * shared.region_factors[:, region_index]
                       + idiosyncratic))
