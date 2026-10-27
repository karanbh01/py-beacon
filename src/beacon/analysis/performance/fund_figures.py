# src/beacon/analysis/performance/fund_figures.py
"""
A fund's own figures: what it held looked like (its characteristics against
its index's), what it earned (its income), how big it was and how money moved
in and out (its AUM and flows), and, for an ETF, how its price on the
exchange stood against its NAV.

Characteristics are weighted over the names with a value, the weights scaled
to them: market cap and dividend yield as weighted means, P/E and P/B as
weighted harmonic means, through earnings and book yields, which a loss or a
tiny book does not blow up.
"""
# BN-385. As agreed with Karan on 2026-10-27.
import numpy as np
import pandas as pd

# The characteristics weighted, and how: a mean, or a harmonic mean.
MEANS = ("MARKET_CAP", "DIVIDEND_YIELD")
HARMONIC = ("PE_RATIO", "PB_RATIO")

# A year back, for the trailing yields.
YEAR = pd.DateOffset(years=1)


def weighted_characteristics(characteristics: pd.DataFrame,
                             weights: pd.Series) -> dict[str, float | None]:
    """A book's characteristics on one day, from its weights by name and
    each name's characteristics (a row a name, ``IDENTIFIER`` and a column
    each)."""
    by_name = characteristics.set_index("IDENTIFIER")
    held = weights[weights > 0]
    found: dict[str, float | None] = {}

    for column in (*MEANS, *HARMONIC):
        if column not in by_name.columns:
            found[column] = None
            continue

        values = by_name[column].reindex(held.index).astype(float)
        usable = values.notna() & np.isfinite(values) & ((values != 0) if column in HARMONIC
                                                         else True)
        share = held[usable]

        if share.sum() <= 0:
            found[column] = None
        elif column in HARMONIC:
            found[column] = float(share.sum() / (share / values[usable]).sum())
        else:
            found[column] = float((share * values[usable]).sum() / share.sum())

    return found


def characteristics_over_time(characteristics: pd.DataFrame,
                              books: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each book's characteristics at each date the characteristics were
    taken (the rebalances), from its weights at the end of that day: a row a
    date, columns by book and characteristic."""
    rows: dict[pd.Timestamp, dict[tuple[str, str], float | None]] = {}

    for date, taken in characteristics.groupby(level=0):
        row: dict[tuple[str, str], float | None] = {}

        for name, weights in books.items():
            on = weights.index[weights.index <= date]

            if not len(on):
                continue

            figures = weighted_characteristics(taken, weights.loc[on[-1]].fillna(0.0))
            row.update({(name, figure): value for figure, value in figures.items()})

        rows[pd.Timestamp(date)] = row

    table = pd.DataFrame.from_dict(rows, orient="index")

    if not table.empty:
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=["book", "figure"])

    return table


def trailing_yields(cash: pd.DataFrame,
                    nav: pd.Series) -> pd.DataFrame:
    """Each day's distribution yield (what was paid out over the year to
    it, over that day's NAV) and trailing dividend yield (the dividends the
    fund became owed over the year to it, gross, over its mean NAV over the
    year), from a year after the run's start."""
    days = nav.index
    # Paid out is negative in the cash; from zero, so nothing paid is 0, not -0.
    paid = 0.0 - cash["distributions"].reindex(days, fill_value=0.0).astype(float)
    owed = cash["dividends_gross"].reindex(days, fill_value=0.0).astype(float)
    window = "365D"
    paid_year = paid.rolling(window).sum()
    owed_year = owed.rolling(window).sum()
    mean_nav = nav.rolling(window).mean()
    full = pd.Series(days >= days[0] + YEAR, index=days) if len(days) else pd.Series(dtype=bool)

    return pd.DataFrame({"distribution_yield": (paid_year / nav).where(full),
                         "dividend_yield": (owed_year / mean_nav).where(full)})


def dividend_growth(cash: pd.DataFrame) -> pd.DataFrame:
    """Each calendar year's dividends (gross, as the fund became owed them)
    and their growth on the year before's."""
    owed = cash["dividends_gross"].astype(float)
    yearly = owed.groupby(pd.DatetimeIndex(owed.index).year.astype(str)).sum()
    table = pd.DataFrame({"dividends": yearly,
                          "growth": yearly / yearly.shift(1).where(yearly.shift(1) > 0) - 1})
    table.index.name = "year"

    return table


def net_flows(cash: pd.DataFrame,
              nav: pd.Series,
              frequency: str = "M") -> pd.DataFrame:
    """Each month's (``M``) or year's (``Y``) net flows (money in less money
    out, as dealt) and their share of the AUM at the close before it began."""
    flows = cash["flows_dealt"].reindex(nav.index, fill_value=0.0).astype(float)
    periods = pd.DatetimeIndex(nav.index).to_period(frequency)
    table = pd.DataFrame({"net_flows": flows.groupby(periods).sum()})
    starts = nav.shift(1).groupby(periods).first()
    table["share_of_aum"] = table["net_flows"] / starts.where(starts > 0)
    table.index = table.index.astype(str)
    table.index.name = "period"

    return table


def flow_costs(trades: pd.DataFrame,
               cash: pd.DataFrame,
               costs: pd.DataFrame) -> pd.DataFrame:
    """The cost of trading each day's flows: on a day with flows and no
    rebalance, every trading cost that day; on a rebalance day, the share of
    the day's trading costs the flow's size is of what was traded; with the
    creation fees dealing investors paid set against it. A row a day with
    flows."""
    flows = cash["flows_dealt"].astype(float)
    flowed = flows[flows != 0]
    days = pd.DatetimeIndex(trades.index).normalize()
    traded = trades["VALUE"].astype(float).groupby(days).sum()
    cost = trades["COST"].astype(float).groupby(days).sum()
    rebalanced = set(days[(trades["REASON"] == "rebalance").to_numpy()])
    rows = {}

    for dealt, amount in flowed.items():
        day = pd.Timestamp(dealt).normalize()
        on, value = float(cost.get(day, 0.0)), float(traded.get(day, 0.0))
        share = 1.0 if day not in rebalanced else (min(1.0, abs(amount) / value) if value else 0.0)
        fees = float(costs["creation_fees"].get(day, 0.0)) if "creation_fees" in costs else 0.0
        rows[day] = {"flow": float(amount), "trading_cost": on * share,
                     "fees_received": fees, "net_cost": on * share - fees}

    table = pd.DataFrame.from_dict(rows, orient="index",
                                   columns=["flow", "trading_cost", "fees_received", "net_cost"])
    table.index.name = "date"

    return table


def premium_figures(market: pd.DataFrame) -> dict[str, float | None]:
    """An ETF's premium (its price over its NAV, less one) on the exchange:
    its mean, its standard deviation, and its widest above and below; and
    the mean bid-ask spread."""
    if market.empty or "premium" not in market:
        return dict.fromkeys(("mean_premium", "premium_volatility", "widest_premium",
                              "widest_discount", "mean_spread"))

    premium = market["premium"].astype(float)

    return {"mean_premium": float(premium.mean()),
            "premium_volatility": float(premium.std()) if len(premium) > 1 else None,
            "widest_premium": float(premium.max()),
            "widest_discount": float(premium.min()),
            "mean_spread": float(market["spread"].astype(float).mean())
            if "spread" in market else None}
