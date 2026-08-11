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
MARKET_SHARE = 0.40
SECTOR_SHARE = 0.16
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
# No single exponent fits the real market at both ends: the very top is
# *flatter* than a Pareto (index rules, antitrust and mean reversion all bite
# hardest on the largest company) while the body is heavier. 1.4 is the best
# joint fit, over 300 names and 60 seeds:
#
#     statistic          generated    S&P 500
#     largest name            9.4%    ~7% (2024), ~4% (2015)
#     top ten                30.4%    ~35% (2024), ~20% (2015)
#     top decile             46.2%    ~58% (2024), ~45% (2015)
#
# Slightly too concentrated at the very top, slightly too flat below it. The
# tail stays heavy enough that a 10% cap binds regularly, which is what the
# generator needs it for.
MIN_MARKET_CAP = 5.0e8
PARETO_SHAPE = 1.4

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
          currency: str = DEFAULT_CURRENCY) -> pd.DataFrame:
    """Draw the static universe.

    Args:
        count: How many names.
        rng: Seeded generator; every draw here comes from it, so the universe
            is a function of the seed alone.
        currency: Reporting currency for every name.

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

    prices = MIN_PRICE + (MAX_PRICE - MIN_PRICE) * rng.beta(1.6, 3.0, size=count)

    # Pareto by inverse transform: a few names orders of magnitude above the
    # floor, most near it.
    market_cap = MIN_MARKET_CAP * (1.0 - rng.uniform(size=count)) ** (-1.0 / PARETO_SHAPE)

    yields = MAX_DIVIDEND_YIELD * rng.beta(2.0, 4.0, size=count)
    yields[rng.uniform(size=count) < NON_PAYER_FRACTION] = 0.0

    frame = pd.DataFrame({
        "NAME": [company_name(ticker) for ticker in tickers],
        "SECTOR": [SECTORS[position % len(SECTORS)] for position in positions],
        "EXCHANGE": rng.choice(EXCHANGES, size=count),
        "CURRENCY": currency,
        "volatility": volatility,
        "market_share": market_share,
        "sector_share": sector_share,
        "alpha": rng.normal(0.0, ALPHA_SPREAD, size=count),
        "initial_price": prices,
        "market_cap": market_cap,
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

    return frame


def reference_frame(universe: pd.DataFrame,
                    valid_from: str) -> pd.DataFrame:
    """The reference dataset, long-form and ready for `ReferenceData`.

    Args:
        universe: Output of :func:`build`.
        valid_from: DATE_FROM stamped on every record. A generated universe has
            no history of reclassification, so every record is valid from the
            start of the panel rather than pretending to a change it never had.

    Returns:
        pd.DataFrame: One row per name.
    """
    return pd.DataFrame({
        "IDENTIFIER": universe.index,
        "DATE_FROM": valid_from,
        "DATE_TO": pd.NaT,
        "NAME": universe["NAME"].to_numpy(),
        "SECTOR": universe["SECTOR"].to_numpy(),
        "SUB_INDUSTRY": universe["SUB_INDUSTRY"].to_numpy(),
        "EXCHANGE": universe["EXCHANGE"].to_numpy(),
        "CURRENCY": universe["CURRENCY"].to_numpy(),
    })
