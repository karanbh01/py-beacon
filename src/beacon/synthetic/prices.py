# src/beacon/synthetic/prices.py
"""
Turning a return panel into the market data a client actually reads: OHLCV,
shares outstanding and free float, with dividends and splits folded into the
price path rather than bolted on beside it.

## The price is not the return path

A return series compounds into a *total-return* index. A stored `CLOSE` is
neither that nor a clean geometric path: it drops on an ex-dividend date and it
halves on a split, and every downstream calculation — trailing yield, the
divisor adjustment, a split-adjusted chart — exists to undo one of those two
things. Generating prices that never do either would produce a dataset on which
none of that code could be exercised, which is the opposite of the point.

So the stored close is built as

    close = initial · Π (1 + r) · (1 - q) / cumulative_split

where `q` is the ex-date drop as a fraction of price, and the corporate-action
history records exactly the dividends and splits that appear in it. The two are
generated together, so they cannot disagree.

## Reconstructing the economic path

Multiplying `CLOSE` by the cumulative split ratio and adding dividends back
recovers the return panel this module was handed. A test asserts it, because
"coherent" is otherwise a claim rather than a property.

## OHLC

Open comes off the previous close through an overnight gap; high and low are
pushed out from whichever of open and close is the extreme. Constructing them
that way makes ``H >= max(O, C)`` and ``L <= min(O, C)`` true by arithmetic
rather than by a repair pass afterwards — a clamp that fixes violations after
the fact is a clamp somebody eventually has to trust.

Splits are applied to all four prices at once, so the intraday relationships
survive a split day intact.

## Volume

Log-normal, scaled by market capitalisation so a mega-cap trades more than a
small-cap, and pushed up on days when the move was large — the well-documented
volume/volatility relationship. Without the second part, ADV would be a
constant with noise on it and a liquidity screen built on it would never bind.
"""
import numpy as np
import pandas as pd

from ..data.corporate_actions import ANNOUNCED, PAID

# Ex-dividend months, and the day within the month. Quarterly, deliberately
# off the quarter ends where index rebalances land: an action falling on a
# rebalance date every single time would hide any bug in ordering the two.
DIVIDEND_MONTHS = (2, 5, 8, 11)
DIVIDEND_DAY = 15

# A price above this at an annual review splits. Measured rather than guessed:
# at 500 some seeds produce no split at all over five years, which leaves the
# corporate-actions pane empty on a dataset generated to fill it. At 300 a
# third of the universe splits, which no market does. At 400 every seed tried
# produced at least three, and the frequency stays plausible.
SPLIT_THRESHOLD = 400.0
HIGH_SPLIT_THRESHOLD = 800.0
SPLIT_RATIO = 2.0
HIGH_SPLIT_RATIO = 4.0

# Overnight gap and intraday range, as fractions of the day's own volatility.
# Both are proportional rather than absolute, so a quiet name has quiet bars.
GAP_SCALE = 0.35
RANGE_SCALE = 0.55

# Daily turnover as a fraction of shares outstanding, which is what sets the
# level of volume before the day's activity multiplies it.
MIN_TURNOVER = 0.002
MAX_TURNOVER = 0.012

# How strongly an unusually large move lifts volume, and the noise around it.
VOLUME_ACTIVITY = 0.45
VOLUME_NOISE = 0.35

# Days between an ex-date and the cash actually landing. Real dividends settle
# a few weeks later, and the gap is what makes "announced" a state the pane can
# show: an action whose pay date has not arrived by the end of the panel is
# announced rather than paid. A split settles on its ex-date, so its pay date
# is the same day.
DIVIDEND_SETTLEMENT_DAYS = 21

# Prices are stored at the precision a real feed quotes them, not at the
# seventeen significant figures a float carries. It is more realistic and it is
# a third off the store, and rounding cannot break the OHLC ordering because
# rounding is monotonic: if HIGH >= CLOSE then round(HIGH) >= round(CLOSE).
PRICE_DECIMALS = 4

# Single precision, on the panel that dominates memory. At 800 names over ten
# years the market data holds 119 MB as float64 and 64 MB as float32, and the
# whole point of the larger universes is to fit on an ordinary machine.
#
# The precision this costs is real but bounded. float32 carries about 7.2
# decimal digits, and the widest thing stored here is a price near the 480 top
# of the range quoted to four decimals -- 480.1234, which needs 7. So the
# rounding above is preserved to roughly a tenth of a tick rather than exactly,
# and `test_prices_reconstruct_from_returns` compares against a tolerance
# derived from the tick size for that reason.
#
# SHARES_OUTSTANDING is the one to watch: it runs to 1e9 and above, where a
# float32 ulp is over 64. That is harmless for a market value (a relative error
# of 1e-7) and would not be for a share count anybody had to reconcile, which
# this is not -- it is generated.
STORAGE_DTYPE = "float32"


def dividend_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The ex-dividend dates inside a panel's date range.

    The first business day on or after the 15th of each dividend month, so
    every ex-date is a day the panel actually has a price for.
    """
    candidates = [date for date in dates
                  if date.month in DIVIDEND_MONTHS and date.day >= DIVIDEND_DAY]

    seen: dict[tuple[int, int], pd.Timestamp] = {}
    for date in candidates:
        seen.setdefault((date.year, date.month), date)

    return pd.DatetimeIndex(sorted(seen.values()))


def review_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The annual split-review dates: the first business day of each year.

    The first year is skipped — a name cannot split before the panel starts,
    and reviewing on day one would split names purely for having been drawn an
    expensive opening price.
    """
    seen: dict[int, pd.Timestamp] = {}
    for date in dates:
        seen.setdefault(date.year, date)

    return pd.DatetimeIndex(sorted(seen.values())[1:])


def _split_paths(pre_split: pd.DataFrame,
                 dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Walk the annual reviews, splitting names whose price has run away.

    Args:
        pre_split: The price path before any split is applied.
        dates: The panel's dates.

    Returns:
        tuple: A cumulative-split-factor frame aligned to ``pre_split``, and
        the split actions recorded along the way.
    """
    factor = pd.DataFrame(1.0, index=pre_split.index, columns=pre_split.columns)
    actions: list[dict[str, object]] = []

    for review in review_dates(dates):
        # The price as it would be *quoted* that day: the raw path divided by
        # the splits already applied. Reviewing the raw path instead would keep
        # splitting a name that had already been brought back down.
        quoted = pre_split.loc[review] / factor.loc[review]

        ratios = pd.Series(1.0, index=quoted.index)
        ratios[quoted > SPLIT_THRESHOLD] = SPLIT_RATIO
        ratios[quoted > HIGH_SPLIT_THRESHOLD] = HIGH_SPLIT_RATIO

        splitting = ratios[ratios > 1.0]
        if splitting.empty:
            continue

        factor.loc[review:, splitting.index] *= splitting.to_numpy()

        actions += [{"IDENTIFIER": identifier, "EX_DATE": review,
                     "TYPE": "SPLIT", "VALUE": float(ratio)}
                    for identifier, ratio in splitting.items()]

    return factor, actions


def _dividend_fractions(universe: pd.DataFrame,
                        dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The ex-date price drop, as a fraction of price, per name and date."""
    fractions = pd.DataFrame(0.0, index=dates, columns=universe.index)
    quarterly = universe["dividend_yield"].to_numpy() / 4.0

    for date in dividend_dates(dates):
        fractions.loc[date] = quarterly

    return fractions


def build(universe: pd.DataFrame,
          returns: pd.DataFrame,
          rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the market-data panel and the corporate-action history.

    Args:
        universe: Output of `universe.build`.
        returns: Date-indexed total returns, one column per name.
        rng: Seeded generator.

    Returns:
        tuple: Long-form market data with IDENTIFIER/DATE and OHLCV plus
        SHARES_OUTSTANDING and FREE_FLOAT, and a long-form action history.
    """
    dates = pd.DatetimeIndex(returns.index)

    drops = _dividend_fractions(universe, dates)

    # One multiplicative path carrying both the economic return and the ex-date
    # drop, so the close and the dividend cannot drift apart.
    multipliers = (1.0 + returns) * (1.0 - drops)
    pre_split = multipliers.cumprod() * universe["initial_price"].to_numpy()

    factor, split_actions = _split_paths(pre_split, dates)

    # The dividend is the drop, measured against the price before it — and then
    # divided by the split factor, because the recorded amount has to be quoted
    # in the same currency as the stored CLOSE. Without that division a dividend
    # paid after a two-for-one split is reported at twice the cash the holder
    # of one (post-split) share actually received, and every trailing yield
    # computed from it is double.
    before_drop = pre_split / (1.0 - drops)
    dividends = before_drop * drops / factor

    close = pre_split / factor
    shares = factor * universe["shares_outstanding"].to_numpy()

    frames = _bars(universe, returns, pre_split, factor, rng)
    frames["CLOSE"] = close
    frames["VOLUME"] = _volume(universe, returns, factor, rng)

    market = _long_form(frames, shares, universe)
    actions = _action_frame(split_actions, dividends, drops, dates[-1])

    return market, actions


def _bars(universe: pd.DataFrame,
          returns: pd.DataFrame,
          pre_split: pd.DataFrame,
          factor: pd.DataFrame,
          rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Open, high and low, split-adjusted alongside the close."""
    shape = returns.shape
    daily_volatility = universe["volatility"].to_numpy() / np.sqrt(252.0)

    previous = pre_split.shift(1)
    previous.iloc[0] = universe["initial_price"].to_numpy()

    gap = rng.normal(0.0, GAP_SCALE * daily_volatility, size=shape)
    open_ = previous * (1.0 + gap)

    upper = np.abs(rng.normal(0.0, RANGE_SCALE * daily_volatility, size=shape))
    lower = np.abs(rng.normal(0.0, RANGE_SCALE * daily_volatility, size=shape))

    # Pushed out from the extremes rather than drawn independently, so the
    # ordering holds by construction and never needs repairing.
    top = np.maximum(open_, pre_split) * (1.0 + upper)
    bottom = np.minimum(open_, pre_split) * (1.0 - lower)

    return {"OPEN": open_ / factor,
            "HIGH": top / factor,
            "LOW": bottom / factor}


def _volume(universe: pd.DataFrame,
            returns: pd.DataFrame,
            factor: pd.DataFrame,
            rng: np.random.Generator) -> pd.DataFrame:
    """Log-normal volume, lifted on days with a large move."""
    count = len(universe)
    turnover = rng.uniform(MIN_TURNOVER, MAX_TURNOVER, size=count)
    base = universe["shares_outstanding"].to_numpy() * turnover

    daily_volatility = universe["volatility"].to_numpy() / np.sqrt(252.0)
    standardised = (returns.abs() / daily_volatility).to_numpy()

    noise = rng.normal(0.0, VOLUME_NOISE, size=returns.shape)

    # Centred on the mean absolute standardised move so the activity term
    # shifts volume around its base rather than scaling it up wholesale.
    lifted = np.exp(VOLUME_ACTIVITY * (standardised - np.sqrt(2.0 / np.pi))
                    + noise - VOLUME_NOISE ** 2 / 2.0)

    # Split-adjusted like the prices: twice the shares trade at half the price.
    volume = pd.DataFrame(np.round(base * lifted),
                          index=returns.index, columns=returns.columns)

    return volume * factor


def _long_form(frames: dict[str, pd.DataFrame],
               shares: pd.DataFrame,
               universe: pd.DataFrame) -> pd.DataFrame:
    """Stack the per-field panels into the IDENTIFIER/DATE frame MarketData reads."""
    ordered = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]

    stacked = {name: frames[name].round(PRICE_DECIMALS).stack()
               for name in ordered if name != "VOLUME"}
    stacked["VOLUME"] = frames["VOLUME"].stack()
    stacked["SHARES_OUTSTANDING"] = shares.stack()

    market = pd.DataFrame(stacked)[[*ordered, "SHARES_OUTSTANDING"]]
    market.index.names = ["DATE", "IDENTIFIER"]
    market = market.reset_index()

    # Constant per name, but it lives in the market data because that is where
    # `fetch_free_float_factor` reads it from.
    market["FREE_FLOAT"] = market["IDENTIFIER"].map(
        universe["free_float"].round(PRICE_DECIMALS))

    market = market.astype({name: STORAGE_DTYPE for name in market.columns
                            if name not in ("DATE", "IDENTIFIER")})

    return market.sort_values(["IDENTIFIER", "DATE"], ignore_index=True)


def _action_frame(splits: list[dict[str, object]],
                  dividends: pd.DataFrame,
                  drops: pd.DataFrame,
                  last_date: pd.Timestamp) -> pd.DataFrame:
    """Assemble the action history from the splits and the dividend amounts.

    Args:
        splits: Split actions recorded during the review walk.
        dividends: Per-name, per-date cash amounts.
        drops: Which (name, date) pairs are ex-dates.
        last_date: End of the panel, which decides whether a pay date has
            arrived — and so whether an action reads as paid or announced.
    """
    paying = drops > 0.0

    # `.dropna()` is load-bearing, not tidying: pandas 2 stopped dropping NaN in
    # `stack`, so without it every non-ex-date and every non-payer becomes a
    # dividend action and the history is the size of the price panel.
    cash = (dividends.where(paying)
            .stack()
            .dropna()
            .rename("VALUE")
            .reset_index())
    cash.columns = ["EX_DATE", "IDENTIFIER", "VALUE"]
    cash["TYPE"] = "DIVIDEND"

    frame = pd.concat([cash, pd.DataFrame(splits)], ignore_index=True)
    frame["VALUE"] = frame["VALUE"].round(PRICE_DECIMALS)

    # A split is effective on its ex-date; a dividend settles weeks later.
    settlement = pd.to_timedelta(
        np.where(frame["TYPE"] == "DIVIDEND", DIVIDEND_SETTLEMENT_DAYS, 0), unit="D")
    frame["PAY_DATE"] = frame["EX_DATE"] + settlement

    frame["STATUS"] = np.where(frame["PAY_DATE"] <= last_date, PAID, ANNOUNCED)

    return frame[["IDENTIFIER", "EX_DATE", "TYPE", "VALUE",
                  "PAY_DATE", "STATUS"]].sort_values(
        ["IDENTIFIER", "EX_DATE"], ignore_index=True)
