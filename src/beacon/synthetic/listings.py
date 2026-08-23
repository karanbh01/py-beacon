# src/beacon/synthetic/listings.py
"""
When a company joins the universe and when it leaves it.

Every generated name used to list on day one and never leave. That is the
definition of a survivorship-biased dataset, and it makes a whole class of
behaviour unreachable: additions and deletions, the divisor adjustment each
one needs, a backtest that has to dispose of a holding that stopped trading,
and point-in-time universe resolution — "who was in the index *then*",
answered from history rather than from today's list.

It also makes the bias itself unmeasurable, which is the point. An index built
only from the names that survived to the end of the panel outperforms one
built as it went along, because the losers were quietly removed from the
sample before the question was asked. A dataset where that difference is zero
cannot demonstrate the single most common error in backtesting.

## The rates are real, the events are not

Roughly 3% of a large-cap index leaves per year and roughly 3% joins —
acquisitions, failures, and index-committee decisions all together. Over ten
years that is about a quarter of the universe turning over, which is enough to
matter and not so much that the panel becomes unrecognisable.

## Failures cluster, and that is the whole point

A constant hazard would put delistings evenly across the panel, and the bias a
point-in-time index suffers would be a slow drip. Real companies fail
*together*, in exactly the windows where the index is already falling, which
is what makes survivorship bias large rather than merely present. The hazard
here is scaled by the same regime intensity the return process uses, so
delistings pile into 2001, 2008 and 2020.

Listings do the opposite: nobody floats a company into a collapsing market, so
new issuance is suppressed when the intensity is high. That asymmetry is why a
crisis shrinks a universe rather than churning it.
"""
import logging

import numpy as np
import pandas as pd

from .regimes import CRISES, Regime, market_multipliers

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Annualised, and roughly the turnover of a large-cap index. Both are hazard
# rates rather than shares of the universe: over a ten-year panel a 3% hazard
# retires about a quarter of the names.
#
# Kept at the realistic figure rather than tuned down to enlarge the live
# universe. The union is what a client sees in the universe pane and the live
# subset is what an index can select, so the two have to be sized together --
# but the honest lever is the *count*, not the hazard. Halving the rate to
# 1.5% was tried and reverted: it would have made the panel less like a market
# and roughly halved the survivorship bias the dataset can demonstrate, to buy
# something an extra thousand names buys for nothing.
#
# Measured over ten years, at 3%:
#
#     universe   live at start .. end   share of the union
#     5,000      3,646 .. 3,880         73%
#     6,000      4,375 .. 4,656         73%
#
# The CLI default is 6,000 for that reason -- see `synthetic/__main__.py`.
ANNUAL_DELISTING_RATE = 0.030
ANNUAL_LISTING_RATE = 0.030

# How much a full-intensity crisis multiplies the delisting hazard. Company
# failures in 2008-09 ran several times their normal rate, and the whole
# reason to model this is that the failures land inside the drawdown.
CRISIS_FAILURE_MULTIPLIER = 4.0

# ...and how far it suppresses new listings. IPO issuance essentially stopped
# in the second half of 2008, so a crisis shrinks the universe rather than
# churning it.
CRISIS_LISTING_MULTIPLIER = 0.15

# How strongly a weak company is more likely to leave than a strong one.
#
# This is the whole mechanism of survivorship bias and it was missing from the
# first version. Delisting names at a hazard that ignored their prospects
# produced a survivors-only index that matched a point-in-time one to a tenth
# of a percent -- because a *random* subset of the universe is not a biased
# sample of it. Real delistings are not random: a company that fails has
# usually fallen a long way first, so the names that leave are the ones that
# were going to drag, and removing them with hindsight is what flatters a
# backtest.
#
# Applied to each name's alpha, which is its drift relative to CAPM, so the
# weakest names are several times likelier to go than the strongest.
FAILURE_ALPHA_SENSITIVITY = 12.0

# No name is delisted in the first month or listed in the last: an index whose
# constituent joins and leaves inside a single rebalance period is a test of
# nothing, and it makes the generated actions hard to read.
EDGE_BUFFER_DAYS = 21

# The floor on the per-day shape. A crisis suppresses new listings but does
# not make them impossible, and a zero would also make the weights
# unnormalisable if a panel were entirely inside one episode.
MINIMUM_SHAPE = 0.05


def _hazard(dates: pd.DatetimeIndex,
            annual_rate: float,
            multiplier: float,
            regimes: tuple[Regime, ...]) -> np.ndarray:
    """Per-day event probability, scaled by how intense the regime is.

    Args:
        dates: The panel's business days.
        annual_rate: Hazard per year in calm conditions.
        multiplier: What full crisis intensity multiplies it by. Above one for
            failures, below one for new listings.
        regimes: Dated episodes.

    Returns:
        np.ndarray: One probability per date, normalised so the *expected*
        number of events over the panel matches `annual_rate` regardless of
        how many crises the window happens to contain. Without that, a
        25-year panel would retire far more of its universe than a calm
        10-year one purely because it covers more episodes -- the crises would
        be changing the overall rate rather than only its timing.
    """
    # The *volatility* multiplier, not the correlation lift. Both rise in a
    # crisis, but correlation moves over a narrow range (0.25 to 0.60 at peak)
    # while volatility runs 1.0 calm to 3.1 through covid -- and it is the
    # wider one that produces clustering strong enough to see. Using the
    # correlation lift put crisis years only 1.26x above calm ones, which is
    # not the phenomenon this is here to model.
    scale, _drift, _lift = market_multipliers(dates, regimes)

    # Scaled by how far the volatility multiplier has risen above calm, and
    # floored: the listing multiplier is *below* one (a crisis suppresses new
    # issuance), so at covid's 3.1x the raw shape would go negative -- which
    # is not "no IPOs" but an invalid probability.
    stress = np.clip(scale - 1.0, 0.0, None)
    shape = np.clip(1.0 + stress * (multiplier - 1.0), MINIMUM_SHAPE, None)
    daily = annual_rate / TRADING_DAYS

    if shape.mean() > 0:
        shape = shape / shape.mean()

    return np.asarray(daily * shape)


def draw(count: int,
         dates: pd.DatetimeIndex,
         rng: np.random.Generator,
         delisting_rate: float = ANNUAL_DELISTING_RATE,
         listing_rate: float = ANNUAL_LISTING_RATE,
         regimes: tuple[Regime, ...] = CRISES,
         alpha: np.ndarray | None = None) -> pd.DataFrame:
    """Draw a listed life for every name.

    Args:
        count: How many names.
        dates: The panel's business days.
        rng: Seeded generator.
        delisting_rate: Annualised hazard of leaving. Zero keeps every name
            for the whole panel, which is what every dataset did before this
            module existed.
        listing_rate: Annualised hazard of having joined partway through.
        regimes: Dated episodes, for the clustering.

    Returns:
        pd.DataFrame: ``listed_from`` and ``listed_to`` per name, both
        Timestamps; ``listed_to`` is NaT for a name still listed at the end.
    """
    first, last = dates[0], dates[-1]

    listed_from = np.full(count, first)
    listed_to = np.full(count, pd.NaT)

    if len(dates) > 2 * EDGE_BUFFER_DAYS:
        inner = dates[EDGE_BUFFER_DAYS:-EDGE_BUFFER_DAYS]

        if listing_rate > 0.0:
            listed_from = _sample(inner, count, rng, listing_rate,
                                  CRISIS_LISTING_MULTIPLIER, regimes, first)

        if delisting_rate > 0.0:
            listed_to = _sample(inner, count, rng, delisting_rate,
                                CRISIS_FAILURE_MULTIPLIER, regimes, pd.NaT,
                                propensity=_weakness(alpha))

    frame = pd.DataFrame({"listed_from": pd.to_datetime(listed_from),
                          "listed_to": pd.to_datetime(listed_to)})

    # A name cannot leave before it arrives. Rather than redraw, the later
    # event simply does not happen -- redrawing would bias the timing of
    # whichever event was kept.
    conflicting = frame["listed_to"].notna() & (frame["listed_to"]
                                                <= frame["listed_from"])
    frame.loc[conflicting, "listed_to"] = pd.NaT

    logger.info("Listings: %d of %d names join after %s, %d leave before %s.",
                int((frame["listed_from"] > first).sum()), count,
                first.date(),
                int(frame["listed_to"].notna().sum()), last.date())

    return frame


def _weakness(alpha: np.ndarray | None) -> np.ndarray | None:
    """Per-name relative likelihood of failing, from its drift.

    Centred on the universe mean and exponentiated, so the scale is a
    *ratio* rather than an offset and the result is positive whatever the
    alphas are. Normalised to mean one, so it redistributes which names leave
    without changing how many do.
    """
    if alpha is None or len(alpha) == 0:
        return None

    weight = np.exp(-FAILURE_ALPHA_SENSITIVITY
                    * (np.asarray(alpha) - np.mean(alpha)))

    return np.asarray(weight / weight.mean())


def _sample(inner: pd.DatetimeIndex,
            count: int,
            rng: np.random.Generator,
            annual_rate: float,
            multiplier: float,
            regimes: tuple[Regime, ...],
            default: object,
            propensity: np.ndarray | None = None) -> np.ndarray:
    """Draw an event date per name, or `default` where the event never happens.

    The event probability over the whole panel is ``1 - exp(-rate * years)``,
    which is the survival form rather than ``rate * years`` -- the latter
    exceeds one on a long enough panel and would retire every name in a
    25-year window.
    """
    hazard = _hazard(inner, annual_rate, multiplier, regimes)
    years = len(inner) / TRADING_DAYS

    probability = 1.0 - np.exp(-annual_rate * years)

    if propensity is not None:
        # Clipped at one: a name cannot leave more than certainly, and a very
        # weak name would otherwise push the implied probability above it.
        probability = np.clip(probability * propensity, 0.0, 1.0)

    happens = rng.uniform(size=count) < probability

    if not happens.any():
        return np.full(count, default)

    weights = hazard / hazard.sum()
    positions = rng.choice(len(inner), size=count, p=weights)

    drawn = np.full(count, default, dtype=object)
    drawn[happens] = inner[positions[happens]]

    return drawn
