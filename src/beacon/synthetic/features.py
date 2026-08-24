# src/beacon/synthetic/features.py
"""
Synthetic features: fundamental ratios and a little alternative data.

A small set, deliberately. The point is to have something to screen on and
something to exercise the point-in-time path with — not to simulate a data
vendor. Four fundamental ratios and two alternative series is enough for a
universe filter, an index rule and a look-ahead test, and cheap enough that
every generated store can carry it.

## Coherent with the price series, not drawn beside it

The trap this module exists to avoid.

A price-to-earnings ratio drawn independently of the prices in the same
dataset contradicts them: screen on `pe_ratio < 15` and screen on price, and
the two disagree about the same company. Worse, the disagreement is invisible
until somebody checks, and by then it has been believed.

So the ratios are *derived from* the generated prices:

    eps        = close / pe_ratio          (so pe x eps == price, exactly)
    book_value = close / pb_ratio
    d/e        = drawn per name, by sector — the one genuinely independent of
                 price, because leverage is a balance-sheet fact

The multiple is what gets drawn, sector by sector, and the per-share figure
follows from it. That ordering is what makes them consistent by construction
rather than by luck.

Sentiment and page views are anchored the same way: sentiment tracks recent
returns, because it does, and page views scale with size and spike on large
moves. A name nobody has heard of does not trend on a quiet day.

## The announcement lag varies, and that is not decoration

`DATE` holds when a value became knowable (`beacon.data.features`), so a
fundamental for the quarter ending 31 March is published somewhere in the
following weeks. **A constant lag would make every look-ahead test pass
whether or not the accessor was correct** — with all values 45 days late, any
off-by-one still lands in the same gap. The lag is drawn per name per quarter,
so a test standing on a given date sees a genuinely ragged edge.

## Coverage is deliberately incomplete

Real fundamentals are missing for some names and some quarters, and
alternative datasets cover a fraction of a universe — mostly the large,
visible names. Generating a complete grid would make the missing-coverage
behaviour in `FeatureRule` untestable against this data, and would overstate
what an alternative vendor sells.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FUNDAMENTALS = "fundamentals"
ALTERNATIVE = "alternative"

# Quarter ends, and how long after one a figure is published. Real reporting
# runs four to ten weeks behind the period; the spread is what matters more
# than the centre, for the reason in the module docstring.
MIN_LAG_DAYS = 28
MAX_LAG_DAYS = 75

# Typical multiples by sector. Approximate long-run US figures rather than
# anybody's estimate: technology trades richer than utilities, and a generator
# that gave every sector the same multiple would make a sector screen and a
# valuation screen the same screen.
SECTOR_PE = {
    "Information Technology": 28.0, "Health Care": 22.0,
    "Consumer Discretionary": 22.0, "Communication Services": 19.0,
    "Industrials": 18.0, "Consumer Staples": 20.0, "Materials": 15.0,
    "Real Estate": 17.0, "Financials": 13.0, "Energy": 12.0,
    "Utilities": 17.0,
}
SECTOR_PB = {
    "Information Technology": 6.0, "Health Care": 4.0,
    "Consumer Discretionary": 3.5, "Communication Services": 2.8,
    "Industrials": 3.0, "Consumer Staples": 4.0, "Materials": 2.0,
    "Real Estate": 2.2, "Financials": 1.3, "Energy": 1.6,
    "Utilities": 1.9,
}
# Debt to equity: the one ratio that is not a function of price. Utilities and
# real estate carry leverage that technology does not.
SECTOR_DEBT_EQUITY = {
    "Utilities": 1.4, "Real Estate": 1.3, "Energy": 0.6, "Industrials": 0.7,
    "Consumer Staples": 0.7, "Materials": 0.6, "Communication Services": 0.6,
    "Health Care": 0.4, "Consumer Discretionary": 0.6, "Financials": 1.1,
    "Information Technology": 0.3,
}

# How widely a name's multiple sits around its sector's, as a lognormal sigma.
MULTIPLE_DISPERSION = 0.35

# What share of names a dataset covers.
#
# Fundamentals are near-universal for listed equity and still not complete.
# Alternative data covers a fraction, skewed to the large and visible -- which
# is both realistic and what keeps this cheap: monthly series over the whole
# universe would outweigh the ratios several times over.
FUNDAMENTAL_COVERAGE = 0.85
ALTERNATIVE_COVERAGE = 0.25

# How much that coverage tilts towards the large. A name at the top of the
# universe by size is covered `FLOOR + SLOPE` times the base rate and one at
# the bottom `FLOOR` times it; the two are chosen so the mean over a uniform
# rank is exactly 1, which keeps the tilt from changing the volume.
SIZE_TILT_FLOOR = 0.4
SIZE_TILT_SLOPE = 1.2

# Alternative data is near-real-time, so it is knowable within days rather
# than weeks.
ALTERNATIVE_LAG_DAYS = 2

# Sentiment runs on [-1, 1] and follows returns rather than leading them.
SENTIMENT_RETURN_SENSITIVITY = 6.0
SENTIMENT_NOISE = 0.25


def build(universe: pd.DataFrame,
          prices: pd.DataFrame,
          returns: pd.DataFrame,
          rng: np.random.Generator,
          fundamentals: bool = True,
          alternative: bool = True) -> pd.DataFrame:
    """Generate the feature table for a panel.

    Args:
        universe: Output of `universe.build`, for sector and size.
        prices: Wide close prices, dates by identifier.
        returns: Wide daily returns, for the sentiment anchor.
        rng: Seeded generator.
        fundamentals: Include the ratio set.
        alternative: Include sentiment and page views.

    Returns:
        pd.DataFrame: Long-form feature rows, ready for `FeatureData`.
    """
    frames = []

    if fundamentals:
        frames.append(_fundamentals(universe, prices, rng))

    if alternative:
        frames.append(_alternative(universe, prices, returns, rng))

    if not frames:
        return pd.DataFrame(columns=["IDENTIFIER", "DATE", "TYPE", "FIELD",
                                     "VALUE", "DETAIL"])

    table = pd.concat(frames, ignore_index=True)

    logger.info("Generated %s feature row(s) across %d field(s).",
                f"{len(table):,}", table["FIELD"].nunique())

    return table


def _quarter_ends(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """The quarter ends inside a panel."""
    quarters = pd.date_range(dates[0], dates[-1], freq="QE")

    return [pd.Timestamp(date) for date in quarters]


def _multiple(sector: str,
              table: dict[str, float],
              rng: np.random.Generator) -> float:
    """One name's multiple, drawn around its sector's."""
    centre = table.get(sector, 18.0)

    return float(centre * rng.lognormal(0.0, MULTIPLE_DISPERSION))


def _visible_names(universe: pd.DataFrame,
                   rng: np.random.Generator) -> pd.Index:
    """Which names an alternative dataset covers.

    Skewed to the large, because that is how these datasets are built: a
    sentiment vendor scrapes what people post about, and people post about
    companies they have heard of. Drawing coverage uniformly would put a
    micro-cap and a mega-cap on the same footing and quietly make the coverage
    gap independent of size -- so a screen on `x_sentiment` would drop names
    at random rather than dropping the obscure ones, which is the wrong shape
    of hole to test against.

    The tilt preserves the average. Measured over 4,000 names and 12 seeds:
    the top decile is covered 38% of the time and the bottom decile 12%, a
    ratio of 3.3, while the mean over the whole universe comes out at 0.250
    against a target of `ALTERNATIVE_COVERAGE` = 0.25.
    """
    rank = universe["market_cap"].rank(pct=True).to_numpy()
    probability = ALTERNATIVE_COVERAGE * (SIZE_TILT_FLOOR
                                          + SIZE_TILT_SLOPE * rank)

    return universe.index[rng.uniform(size=len(universe)) < probability]


def _fundamentals(universe: pd.DataFrame,
                  prices: pd.DataFrame,
                  rng: np.random.Generator) -> pd.DataFrame:
    """Four ratios per name per quarter, derived from the price path."""
    dates = pd.DatetimeIndex(prices.index)
    quarters = _quarter_ends(dates)
    covered = universe.index[rng.uniform(size=len(universe))
                             < FUNDAMENTAL_COVERAGE]

    # Row positions resolved once, not once per name per quarter. Scanning
    # `dates <= quarter` inside the loop is O(days) each time, which at the
    # 6,000-name default came to six hundred million comparisons and took ten
    # times longer than generating the prices these are derived from.
    positions = [(quarter, int(dates.searchsorted(quarter, side="right")) - 1)
                 for quarter in quarters]
    positions = [(quarter, row) for quarter, row in positions if row >= 0]

    closes = prices.to_numpy()
    columns = {name: index for index, name in enumerate(prices.columns)}
    last_date = dates[-1]

    records = []
    for identifier in covered:
        sector = str(universe.loc[identifier, "SECTOR"])

        # Drawn once per name and held: a company's rating drifts, it does not
        # resample every quarter. The *price* moving is what makes the ratios
        # move between reports.
        pe = _multiple(sector, SECTOR_PE, rng)
        pb = _multiple(sector, SECTOR_PB, rng)
        debt_equity = _multiple(sector, SECTOR_DEBT_EQUITY, rng)

        column = columns.get(identifier)

        if column is None:
            continue

        for quarter, row in positions:
            close = closes[row, column]

            if not np.isfinite(close) or close <= 0:
                continue

            lag = int(rng.integers(MIN_LAG_DAYS, MAX_LAG_DAYS))
            known = quarter + pd.Timedelta(days=lag)

            if known > last_date:
                # Published after the panel ends, so nobody ever knew it.
                continue

            period = f"{quarter.year}Q{quarter.quarter}"
            detail = f"period ending {quarter.date()}, reported {period}"

            records += [
                (identifier, known, "pe_ratio", pe, detail),
                (identifier, known, "eps", float(close) / pe, detail),
                (identifier, known, "pb_ratio", pb, detail),
                (identifier, known, "debt_to_equity", debt_equity, detail),
            ]

    return _frame(records, FUNDAMENTALS)


def _alternative(universe: pd.DataFrame,
                 prices: pd.DataFrame,
                 returns: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Monthly sentiment and page views, for a fraction of the universe."""
    dates = pd.DatetimeIndex(prices.index)
    months = [pd.Timestamp(date) for date
              in pd.date_range(dates[0], dates[-1], freq="ME")]

    covered = _visible_names(universe, rng)

    # The trailing-month mean for every name and month at once. Slicing the
    # return panel per name per month was the same O(days) scan the ratios
    # had, for the same reason and at the same cost.
    monthly = returns.rolling(21, min_periods=1).mean()
    drifts = monthly.reindex(monthly.index.union(months)).ffill().loc[months]

    # Attention scales with size, so the page-view level is anchored to market
    # cap rather than drawn flat: a mega-cap and a small-cap trending equally
    # would make the series useless as a proxy for anything.
    caps = universe.loc[covered, "market_cap"]
    scale = np.log10(caps / caps.min() + 10.0)

    records = []
    for position, identifier in enumerate(covered):
        base_views = float(1_000 * scale.iloc[position]
                           * rng.lognormal(0.0, 0.5))

        if identifier not in drifts.columns:
            continue

        per_month = drifts[identifier]

        for month in months:
            drift = float(per_month.get(month, np.nan))

            if not np.isfinite(drift):
                continue

            known = month + pd.Timedelta(days=ALTERNATIVE_LAG_DAYS)

            if known > dates[-1]:
                continue

            # Sentiment follows returns rather than leading them, which is
            # what the research finds and what keeps this from being a
            # free signal nobody could have traded.
            sentiment = float(np.tanh(
                SENTIMENT_RETURN_SENSITIVITY * drift * 21
                + rng.normal(0.0, SENTIMENT_NOISE)))

            # Attention spikes on movement in either direction: a collapse
            # draws readers as readily as a rally.
            views = base_views * float(np.exp(abs(drift) * 40.0)
                                       * rng.lognormal(0.0, 0.3))

            records += [
                (identifier, known, "x_sentiment", sentiment,
                 f"month ending {month.date()}"),
                (identifier, known, "wikipedia_views", round(views),
                 f"month ending {month.date()}"),
            ]

    return _frame(records, ALTERNATIVE)


def _frame(records: list[tuple[str, pd.Timestamp, str, float, str]],
           feature_type: str) -> pd.DataFrame:
    """Assemble long-form rows."""
    if not records:
        return pd.DataFrame(columns=["IDENTIFIER", "DATE", "TYPE", "FIELD",
                                     "VALUE", "DETAIL"])

    return pd.DataFrame({
        "IDENTIFIER": [record[0] for record in records],
        "DATE": [record[1] for record in records],
        "TYPE": feature_type,
        "FIELD": [record[2] for record in records],
        "VALUE": [record[3] for record in records],
        "DETAIL": [record[4] for record in records],
    })
