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

    r[i,t] = mu[i] + b[i]·f_market[t] + g[i]·f_sector(i)[t] + e[i,t]

Three independent sources, each a GJR-GARCH(1,1) process with standardised
Student-t innovations:

* **The market factor** — one series everything loads on. This is what makes
  names co-move, and giving it its own GARCH is what makes them co-move
  *more in a crisis*, which is when correlation matters.
* **Sector factors** — one per GICS sector, so two banks resemble each other
  more than a bank resembles a utility.
* **Idiosyncratic noise** — per name.

Loadings are set from a variance budget rather than drawn directly, so a name's
total volatility is a target that is hit rather than an outcome to be
discovered. With the market at ~35% of variance and the sector at ~15%,
same-sector pairs correlate near 0.50 and cross-sector pairs near 0.35.

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
import numpy as np
import pandas as pd

TRADING_DAYS = 252

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


def _standardised_t(rng: np.random.Generator,
                    degrees: np.ndarray,
                    shape: tuple[int, int]) -> np.ndarray:
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

    return (skewed - skewed.mean(axis=0)) / skewed.std(axis=0)


def simulate_gjr(rng: np.random.Generator,
                 steps: int,
                 target_variance: np.ndarray,
                 persistence: np.ndarray,
                 degrees: np.ndarray) -> np.ndarray:
    """Simulate GJR-GARCH(1,1) series with Student-t innovations.

    Vectorised across series and looped over time, which is the only way round:
    each day's variance depends on the day before, but every series can take
    its step together.

    Args:
        rng: Seeded generator.
        steps: Days to return, after burn-in.
        target_variance: Unconditional daily variance per series.
        persistence: ``alpha + lam/2 + beta`` per series, strictly below 1.
        degrees: Student-t degrees of freedom per series.

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

    innovations = _standardised_t(rng, degrees, (total, count))

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
    return series * np.sqrt(target_variance / series.var(axis=0))


def _draw_parameters(rng: np.random.Generator,
                     count: int) -> tuple[np.ndarray, np.ndarray]:
    """Persistence and degrees of freedom for `count` series."""
    persistence = rng.uniform(MIN_PERSISTENCE, MAX_PERSISTENCE, size=count)
    degrees = rng.uniform(MIN_DEGREES_OF_FREEDOM, MAX_DEGREES_OF_FREEDOM,
                          size=count)

    return persistence, degrees


def simulate(universe: pd.DataFrame,
             dates: pd.DatetimeIndex,
             rng: np.random.Generator,
             risk_free_rate: float,
             equity_premium: float) -> pd.DataFrame:
    """Simulate the total-return panel.

    Args:
        universe: Output of `universe.build`, carrying the volatility target
            and variance shares each loading is derived from.
        dates: Business days to simulate.
        rng: Seeded generator.
        risk_free_rate: Annualised, the base of the CAPM expectation.
        equity_premium: Annualised excess return on a beta-one name.

    Returns:
        pd.DataFrame: Date-indexed daily total returns, one column per name.
    """
    steps = len(dates)
    count = len(universe)

    sectors = universe["SECTOR"].to_numpy()
    sector_names = sorted(set(sectors))
    sector_index = np.array([sector_names.index(sector) for sector in sectors])

    daily_variance = (universe["volatility"].to_numpy() ** 2) / TRADING_DAYS
    market_variance = (MARKET_VOLATILITY ** 2) / TRADING_DAYS

    market_target = np.array([market_variance])
    market = pin_realised_variance(
        simulate_gjr(rng, steps, market_target, *_draw_parameters(rng, 1)),
        market_target)

    sector_variance = np.full(len(sector_names), market_variance)
    sector_factors = pin_realised_variance(
        simulate_gjr(rng, steps, sector_variance,
                     *_draw_parameters(rng, len(sector_names))),
        sector_variance)

    # Loadings from the variance budget: b^2 * market_variance is the share of
    # this name's variance the market is meant to explain, so b follows.
    market_beta = np.sqrt(universe["market_share"].to_numpy() * daily_variance
                          / market_variance)
    sector_beta = np.sqrt(universe["sector_share"].to_numpy() * daily_variance
                          / market_variance)

    idiosyncratic_variance = daily_variance * (
        1.0 - universe["market_share"].to_numpy()
        - universe["sector_share"].to_numpy())
    idiosyncratic = simulate_gjr(rng, steps, idiosyncratic_variance,
                                 *_draw_parameters(rng, count))

    # CAPM, so a high-beta name earns more in expectation and the cross-section
    # of long-run returns is not flat.
    drift = (risk_free_rate + market_beta * equity_premium
             + universe["alpha"].to_numpy()) / TRADING_DAYS

    returns = (drift
               + market_beta * market
               + sector_beta * sector_factors[:, sector_index]
               + idiosyncratic)

    return pd.DataFrame(returns,
                        index=pd.DatetimeIndex(dates, name="DATE"),
                        columns=universe.index)
