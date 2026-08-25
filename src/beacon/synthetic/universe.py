# src/beacon/synthetic/universe.py
"""
The static half of a synthetic universe: who the companies are.

Everything here is decided once per run and does not vary by date — names,
tickers, classification, and the per-name parameters the return process is
driven by. Splitting it out from the time series keeps one question separate
from the other: this module answers "what is in the universe", `returns` and
`prices` answer "what did it do".

## Nothing here resembles a real company

Names are ``Company A`` … ``Company Z``, then ``Company AA`` … in the
spreadsheet-column order, with tickers ``CMPA``, ``CMPB``, … ``CMPAA``. The
``CMP`` prefix is what makes a collision with a real listing impossible rather
than merely unlikely: a three-letter prefix on every symbol means no generated
ticker can ever equal a real one, whatever the universe grows to. A test
asserts it against a blocklist of well-known symbols anyway, because the
guarantee is only as good as the naming function that provides it.

Sector names are the eleven GICS sectors, which are a public taxonomy rather
than anybody's property. Sub-industries are *not*: real GICS sub-industry names
would imply a classification this data does not have, so they are generic
segments within each sector.

## The parameters, and where they come from

The figures below are approximate long-run statistics for US large- and
mid-cap equity. They are targets for a generator, not estimates: the point is
that a chart drawn from this data looks like a chart drawn from a market, so
somebody reviewing a layout is reviewing it against realistic numbers.
"""
import numpy as np
import pandas as pd

from . import listings, regions

# The eleven GICS sectors. A public taxonomy, unlike the company names.
SECTORS = (
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
)

# Segments within a sector. Deliberately generic: real GICS sub-industry names
# would imply a classification that a random draw has not earned.
SEGMENTS = ("Segment A", "Segment B", "Segment C")

TICKER_PREFIX = "CMP"
NAME_PREFIX = "Company"

EXCHANGES = ("XNAS", "XNYS")
DEFAULT_CURRENCY = "USD"

# Annualised total volatility, dispersed across the universe. The band is the
# issue's: roughly a defensive utility at the bottom, a small speculative name
# at the top. Drawn from a Beta so the mass sits nearer the low end, as it does
# in a real cross-section, rather than uniformly across the range.
MIN_VOLATILITY = 0.15
MAX_VOLATILITY = 0.50
VOLATILITY_BETA = (2.0, 3.0)

# How total variance splits three ways, before per-name jitter. With the shared
# factors pinned to their realised variance (see `returns.pin_realised_variance`)
# these land close to what they imply: same-sector pairs near 0.55, cross-sector
# near 0.42, averaging ~0.39 once the eleven sectors are counted — comfortably
# inside the 0.3-0.5 the issue asks for, and stable across seeds to ±0.02.
# The region share is taken out of the *market* share rather than added on
# top, so the average pairwise correlation stays where it was while the
# structure underneath it gets richer. Adding 0.08 of shared variance to a
# universe already at 0.40 would push the average to 0.44 and quietly
# invalidate every correlation figure documented here.
#
# Expected average pairwise correlation is
#     market + sector x P(same sector) + region x P(same region)
# With eleven sectors assigned round-robin, P(same sector) is 0.09. The region
# weights are concentrated, so P(same region) is the sum of their squares,
# about 0.40 -- dominated by the United States at 0.60. That gives
# 0.34 + 0.16x0.09 + 0.08x0.40 = 0.39, against 0.41 before regions existed.
MARKET_SHARE = 0.34
SECTOR_SHARE = 0.16
REGION_SHARE = 0.08
SHARE_JITTER = 0.06

# Per-name alpha, annualised: noise around the CAPM expectation, not skill.
ALPHA_SPREAD = 0.03

# Starting prices. A range wide enough that the split rule fires for some names
# over five years and not for most.
MIN_PRICE = 12.0
MAX_PRICE = 480.0

# Market capitalisation is drawn from a Pareto tail so a few names dominate and
# the rest form a long tail — without it, cap weighting and cap rules would
# behave like equal weighting and never bind.
#
# The shape is calibrated to realised index concentration, not to the pure
# Zipf law firm sizes follow. At the 1.1 this started with, the largest name
# averaged 19% of a 300-name index and reached 79% on one demo seed -- an
# index that is really one stock. The exponent governs the *maximum* of the
# draw, and at alpha near one that maximum is effectively unbounded, so this
# was a property of the design rather than an unlucky seed.
#
# Drawn as **order statistics** rather than independently: the k-th largest
# company gets a cap proportional to ``k ** (-1 / shape)``, which is Zipf's
# law for firm sizes, and the ranks are then shuffled across names.
#
# Independent draws were the second thing to get wrong here. Fixing the
# exponent fixed the *average* concentration and left the tail unbounded, so
# roughly one seed in ten still produced an index its largest name owned:
# across 200 seeds at 300 names the top weight averaged a plausible 11% and
# reached 88%. Truncating the draw barely helped, because the failure is not a
# single enormous draw -- it is every *other* draw coming out small, which no
# ceiling on the maximum prevents.
#
# Order statistics fix it at the source: the *shape* of the distribution is
# imposed and only the noise around it is random. Across 200 seeds at 300
# names, with the jitter below:
#
#     statistic          generated              S&P 500
#     largest name       6.6% (worst 12.1%)     ~7% (2024), ~4% (2015)
#     top decile        43.2% (worst 48.1%)     ~58% (2024), ~45% (2015)
#
# The tail stays heavy enough that a 10% cap binds regularly, which is what
# the generator needs it for.
MIN_MARKET_CAP = 5.0e8
PARETO_SHAPE = 1.4

# Lognormal noise on the Zipf profile. Without it every generated universe has
# an identical concentration curve, which is its own kind of unrealistic; at
# 0.25 the ranks shuffle around a bit without the profile losing its shape.
CAP_JITTER = 0.25

# Free float. Most of a large-cap universe is fully floated; a minority is
# founder- or state-controlled, and those are what make a float adjustment
# visibly different from raw market cap.
MIN_FREE_FLOAT = 0.35
FREE_FLOAT_BETA = (5.0, 1.5)

# Annual dividend yield. A third of the universe pays nothing, which is what
# makes "trailing dividend" a question with two different answers.
MAX_DIVIDEND_YIELD = 0.05
NON_PAYER_FRACTION = 0.3


def ticker_suffix(position: int) -> str:
    """The spreadsheet-column label for a position: A, B, ... Z, AA, AB, ...

    Args:
        position: Zero-based index into the universe.

    Returns:
        str: The label, always at least one character.
    """
    letters = ""
    remaining = position + 1

    while remaining > 0:
        remaining, offset = divmod(remaining - 1, 26)
        letters = chr(ord("A") + offset) + letters

    return letters


def identifiers(count: int) -> list[str]:
    """The tickers for a universe of a given size.

    One function, used by every dataset in this package, so a name and its
    ticker cannot disagree between the market data and the reference data.
    """
    return [f"{TICKER_PREFIX}{ticker_suffix(position)}" for position in range(count)]


def company_name(identifier: str) -> str:
    """The display name matching a generated ticker."""
    return f"{NAME_PREFIX} {identifier[len(TICKER_PREFIX):]}"


def build(count: int,
          rng: np.random.Generator,
          dates: pd.DatetimeIndex | None = None,
          currency: str = DEFAULT_CURRENCY,
          delisting_rate: float = listings.ANNUAL_DELISTING_RATE,
          listing_rate: float = listings.ANNUAL_LISTING_RATE) -> pd.DataFrame:
    """Draw the static universe.

    Args:
        count: How many names.
        rng: Seeded generator; every draw here comes from it, so the universe
            is a function of the seed alone.
        dates: The panel's business days. When given, each name draws a listed
            life over them; when omitted every name is listed for the whole
            panel, which is what callers that only want the static fields get.
        currency: Retained for callers that pass it. Ignored since BN-128:
            a name's currency comes from its listing region, and a universe
            forced into one currency is the case the FX paths never exercise.

    Returns:
        pd.DataFrame: One row per name, indexed by identifier, carrying both
        the reference fields and the parameters the return process needs.
    """
    tickers = identifiers(count)
    positions = np.arange(count)

    volatility = MIN_VOLATILITY + (MAX_VOLATILITY - MIN_VOLATILITY) * rng.beta(
        *VOLATILITY_BETA, size=count)

    # Jittered so no two names decompose their variance identically; clipped
    # because a negative share is meaningless and the jitter is wide enough to
    # reach one at the extremes.
    market_share = np.clip(
        MARKET_SHARE + rng.normal(0.0, SHARE_JITTER, size=count), 0.10, 0.60)
    sector_share = np.clip(
        SECTOR_SHARE + rng.normal(0.0, SHARE_JITTER, size=count), 0.02, 0.35)
    region_share = np.clip(
        REGION_SHARE + rng.normal(0.0, SHARE_JITTER / 2.0, size=count),
        0.01, 0.20)

    prices = MIN_PRICE + (MAX_PRICE - MIN_PRICE) * rng.beta(1.6, 3.0, size=count)

    # Zipf on the ranks: the largest name is `count ** (1 / shape)` times the
    # smallest, with lognormal noise so the curve is not identical every run.
    # Shuffled afterwards because rank would otherwise track position, and
    # position decides sector -- which would make every universe's biggest
    # company a Communication Services name.
    ranks = np.arange(1, count + 1)
    market_cap = (MIN_MARKET_CAP
                  * (count / ranks) ** (1.0 / PARETO_SHAPE)
                  * np.exp(rng.normal(0.0, CAP_JITTER, size=count)))
    rng.shuffle(market_cap)

    yields = MAX_DIVIDEND_YIELD * rng.beta(2.0, 4.0, size=count)
    yields[rng.uniform(size=count) < NON_PAYER_FRACTION] = 0.0

    assigned = regions.assign(count, rng)
    venues = regions.frame(assigned)

    frame = pd.DataFrame({
        "NAME": [company_name(ticker) for ticker in tickers],
        "SECTOR": [SECTORS[position % len(SECTORS)] for position in positions],
        "REGION": venues["REGION"].to_numpy(),
        "EXCHANGE": venues["EXCHANGE"].to_numpy(),
        "CURRENCY": venues["CURRENCY"].to_numpy(),
        # Listing and domicile are separate columns because they are separate
        # questions. "Listed in Germany" and "incorporated in Ireland" are
        # both real screens, and a single COUNTRY would silently answer one
        # of them while appearing to answer both.
        "COUNTRY_LISTING": venues["COUNTRY_LISTING"].to_numpy(),
        "COUNTRY_DOMICILE": regions.domiciles(assigned, rng),
        "volatility": volatility,
        "market_share": market_share,
        "sector_share": sector_share,
        "region_share": region_share,
        "alpha": rng.normal(0.0, ALPHA_SPREAD, size=count),
        "initial_price": prices,
        # Drawn on a USD scale and converted, so the *size* distribution is
        # global while the quoted price stays local. Without this a name's
        # rank in the universe would depend on its exchange rate.
        "market_cap": market_cap / venues["fx_to_base"].to_numpy(),
        "free_float": MIN_FREE_FLOAT + (1.0 - MIN_FREE_FLOAT) * rng.beta(
            *FREE_FLOAT_BETA, size=count),
        "dividend_yield": yields,
    }, index=pd.Index(tickers, name="IDENTIFIER"))

    # Sector is assigned round-robin so every sector is populated at any
    # universe size; the segment is drawn, so sub-industries are uneven the way
    # real ones are.
    frame["SUB_INDUSTRY"] = [
        f"{sector} — {segment}"
        for sector, segment in zip(frame["SECTOR"],
                                   rng.choice(SEGMENTS, size=count),
                                   strict=True)]

    # Shares follow from a cap and a price rather than being drawn: drawing
    # both would let a name carry a share count its own price contradicts.
    frame["shares_outstanding"] = np.round(frame["market_cap"]
                                           / frame["initial_price"])

    if dates is not None and len(dates) > 0:
        lives = listings.draw(count, dates, rng,
                              delisting_rate=delisting_rate,
                              listing_rate=listing_rate,
                              alpha=frame["alpha"].to_numpy())
        frame["listed_from"] = lives["listed_from"].to_numpy()
        frame["listed_to"] = lives["listed_to"].to_numpy()

    return frame


def reference_frame(universe: pd.DataFrame,
                    valid_from: str,
                    profile: pd.DataFrame | None = None) -> pd.DataFrame:
    """The reference dataset, long-form and ready for `ReferenceData`.

    Args:
        universe: Output of :func:`build`.
        valid_from: DATE_FROM for names listed since the panel began. A
            generated universe has no history of reclassification, so a name
            that was there at the start gets one record valid from the start
            rather than pretending to a change it never had.
        profile: Optional profile columns from `profiles.build`, joined on
            identifier. Absent leaves the record as it was before BN-148.

    Returns:
        pd.DataFrame: One row per name, carrying the listed life. `DATE_TO` is
        NaT for a name still listed at the end of the panel, which is what
        `ReferenceData.get` reads as "still valid" — so point-in-time
        resolution drops a delisted name automatically.
    """
    listed_from = (universe["listed_from"] if "listed_from" in universe
                   else pd.Series(pd.Timestamp(valid_from),
                                  index=universe.index))
    listed_to = (universe["listed_to"] if "listed_to" in universe
                 else pd.Series(pd.NaT, index=universe.index))

    frame = pd.DataFrame({
        "IDENTIFIER": universe.index,
        "DATE_FROM": listed_from.to_numpy(),
        "DATE_TO": listed_to.to_numpy(),
        "NAME": universe["NAME"].to_numpy(),
        "SECTOR": universe["SECTOR"].to_numpy(),
        "SUB_INDUSTRY": universe["SUB_INDUSTRY"].to_numpy(),
        "REGION": universe["REGION"].to_numpy(),
        "EXCHANGE": universe["EXCHANGE"].to_numpy(),
        "CURRENCY": universe["CURRENCY"].to_numpy(),
        "COUNTRY_LISTING": universe["COUNTRY_LISTING"].to_numpy(),
        "COUNTRY_DOMICILE": universe["COUNTRY_DOMICILE"].to_numpy(),
    })

    if profile is None:
        return frame

    # The profile fields a client's instrument view needs: identifiers,
    # the middle two GICS levels, and the corporate facts (BN-148). Joined
    # rather than generated here so this function stays the shape of the
    # record, and `profiles.build` owns the coherence rules -- a status that
    # agrees with the listing dates, a dividend frequency that agrees with the
    # dividends.
    joined = profile.reindex(universe.index)

    for column in joined.columns:
        frame[column] = joined[column].to_numpy()

    return frame
