# src/beacon/analysis/performance/dealing.py
"""
What dealing in a fund cost: each subscription or redemption (each creation
or redemption, for an ETF), and each share class's costs.

**A flow's cost to the fund** is the trading it caused, as `fund_figures`
attributes it (every trading cost of a day with flows and no rebalance; on a
rebalance day, the flow's share of the day's trading), less what the dealing
investors paid into the fund toward it through the price or a levy. For an
ETF, **its cost to the AP** is the flow times the quote's cost of creating or
redeeming that day: the AP's fee, and the basket's trading or the creation
fee.

**A class's costs** are its fees; its hedge's trading costs, less the carry
its forwards earned (each forward's locked-in carry, for the share of its term
it was held); the trading its own flows caused, less what its dealing
investors paid; and its share of the rest of the fund's trading (rebalances,
dividends, the cash band, futures rolls), by its share of the fund's assets
at the close before. All are in the book's currency, so the classes' trading
adds up to the fund's.
"""
# BN-408. As agreed with Karan on 2026-10-27.
import numpy as np
import pandas as pd

from ...portfolio.reasons import REDEMPTION, SUBSCRIPTION
from ..class_panels import CLASS_DAYS, CLASS_FLOWS, CLASS_HEDGES
from .fund_figures import flow_costs

BASIS_POINTS = 10_000.0

# A class's costs, in the order a table gives them.
CLASS_FIGURES = ("fees", "hedge_costs", "hedge_carry", "flow_trading", "other_trading",
                 "dealing_received", "net_cost")


def dealing_costs(trades: pd.DataFrame,
                  cash: pd.DataFrame,
                  costs: pd.DataFrame,
                  market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each day's flow: whether money came in or went out, its size, the
    trading it caused, what the dealing investors paid, the net cost to the
    fund and in basis points of the flow, and, for an ETF, its cost to the
    AP."""
    table = flow_costs(trades, cash, costs)
    table.insert(0, "kind", np.where(table["flow"] > 0, SUBSCRIPTION, REDEMPTION))
    size = table["flow"].abs()
    table["net_bps"] = table["net_cost"] / size.where(size > 0) * BASIS_POINTS
    table["ap_cost"] = np.nan

    if market is not None and not market.empty:
        quotes = market.reindex(pd.DatetimeIndex(table.index).normalize())
        rate = np.where(table["flow"] > 0, quotes["create_cost"], quotes["redeem_cost"])
        table["ap_cost"] = size.to_numpy() * rate.astype(float)

    return table


def dealing_summary(table: pd.DataFrame,
                    by: str = "run") -> pd.DataFrame:
    """The flows' costs added up by kind, over the run (``run``) or each
    year too (``year``): how many, their size, the trading they caused, what
    dealing investors paid, the net cost and in basis points of the flows,
    and the AP's cost."""
    keys: list[pd.Series | pd.Index] = [table["kind"]]

    if by == "year":
        keys.insert(0, pd.Index(pd.DatetimeIndex(table.index).year.astype(str), name="year"))
    elif by != "run":
        raise ValueError(f"Unknown summary by {by!r}. Known: run, year.")

    size = table["flow"].abs()
    frame = table.assign(flows=1, size=size)
    summed = frame.groupby(keys)[["flows", "size", "trading_cost", "fees_received",
                                  "net_cost", "ap_cost"]].sum(min_count=1)
    summed["flows"] = summed["flows"].astype(int)
    summed["net_bps"] = summed["net_cost"] / summed["size"].where(summed["size"] > 0) * BASIS_POINTS

    return summed


def class_costs(tables: dict[str, pd.DataFrame],
                trades: pd.DataFrame,
                cash: pd.DataFrame,
                costs: pd.DataFrame,
                periods_per_year: int,
                by: str = "run") -> pd.DataFrame:
    """Each class's costs in the book's currency, over the run (``run``,
    with the net in basis points of its mean assets a year) or each year
    (``year``, with the net in basis points of the year's mean assets)."""
    days = tables[CLASS_DAYS]
    assets = days.pivot_table(index=days.index, columns="CLASS", values="BOOK_NET_ASSETS")
    rate = (days.pivot_table(index=days.index, columns="CLASS", values="NET_ASSETS")
            / assets.where(assets > 0)).ffill()
    fees = days.pivot_table(index=days.index, columns="CLASS", values="FEES") / rate
    flow_trading, other_trading = _trading_by_class(tables[CLASS_FLOWS], rate, assets,
                                                    trades, cash, costs)
    dealt = _flows_in_book(tables[CLASS_FLOWS], rate, "ADJUSTMENT")
    hedges = tables[CLASS_HEDGES]
    carry = _carry_earned(hedges, days.index.max())
    hedge_costs = hedges.pivot_table(index=hedges.index, columns="CLASS", values="COST",
                                     aggfunc="sum")

    figures = {"fees": fees, "hedge_costs": hedge_costs, "hedge_carry": carry,
               "flow_trading": flow_trading, "other_trading": other_trading,
               "dealing_received": dealt}
    frame = pd.concat({name: _daily(found, assets) for name, found in figures.items()}, axis=1)
    frame = frame.stack(level=1, future_stack=True).fillna(0.0)
    frame["net_cost"] = (frame["fees"] + frame["hedge_costs"] - frame["hedge_carry"]
                         + frame["flow_trading"] + frame["other_trading"]
                         - frame["dealing_received"])
    held = assets.iloc[1:].stack().rename("assets")

    return _by_class(frame[list(CLASS_FIGURES)], held, periods_per_year, by)


def _by_class(frame: pd.DataFrame,
              held: pd.Series,
              periods_per_year: int,
              by: str) -> pd.DataFrame:
    """*frame*'s daily figures by class and day, added up by class (and
    year), with the net in basis points of the mean assets."""
    names = frame.index.get_level_values(1)

    if by == "run":
        table = frame.groupby(names).sum()
        mean = held.groupby(level=1).mean()
        years = held.groupby(level=1).size() / float(periods_per_year)
        table["net_bps_a_year"] = (table["net_cost"] / (mean * years).where(mean > 0)
                                   * BASIS_POINTS)
    elif by == "year":
        years_of = pd.DatetimeIndex(frame.index.get_level_values(0)).year.astype(str)
        table = frame.groupby([names, years_of]).sum()
        mean = held.groupby([held.index.get_level_values(1),
                             pd.DatetimeIndex(held.index.get_level_values(0)).year.astype(str)]
                            ).mean()
        table["net_bps"] = table["net_cost"] / mean.reindex(table.index) * BASIS_POINTS
        table.index.names = ["class", "year"]
    else:
        raise ValueError(f"Unknown class costs by {by!r}. Known: run, year.")

    if by == "run":
        table.index.name = "class"

    return table


def _daily(found: pd.DataFrame,
           assets: pd.DataFrame) -> pd.DataFrame:
    """*found* on the run's days and classes, 0 where it has nothing."""
    return found.reindex(index=assets.index[1:], columns=assets.columns).fillna(0.0)


def _flows_in_book(flows: pd.DataFrame,
                   rate: pd.DataFrame,
                   column: str) -> pd.DataFrame:
    """Each class's *column* of its flows each day, in the book's currency."""
    if flows.empty:
        return pd.DataFrame(index=rate.index, columns=rate.columns, dtype=float)

    by_day = flows.pivot_table(index=flows.index, columns="CLASS", values=column, aggfunc="sum")

    return by_day / rate.reindex(index=by_day.index, columns=by_day.columns)


def _trading_by_class(flows: pd.DataFrame,
                      rate: pd.DataFrame,
                      assets: pd.DataFrame,
                      trades: pd.DataFrame,
                      cash: pd.DataFrame,
                      costs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fund's trading costs each day split between the classes: the
    trading flows caused to the classes that dealt, by the size of each's
    flow; the rest by each's share of the assets at the close before."""
    days = pd.DatetimeIndex(trades.index).normalize()
    total = trades["COST"].astype(float).groupby(days).sum()
    futures = costs.reindex(columns=["futures_rolls", "futures_trading"], fill_value=0.0)
    total = total.add(futures.astype(float).sum(axis=1), fill_value=0.0)
    caused = flow_costs(trades, cash, costs)["trading_cost"]

    sizes = _flows_in_book(flows, rate, "AMOUNT").abs()
    shares = sizes.div(sizes.sum(axis=1).where(lambda found: found > 0), axis=0)
    flow_trading = shares.mul(caused.reindex(shares.index).fillna(0.0), axis=0)

    rest = total.sub(caused, fill_value=0.0)
    before = assets.shift(1)
    split = before.div(before.sum(axis=1).where(lambda found: found > 0), axis=0)
    other_trading = split.mul(rest.reindex(split.index).fillna(0.0), axis=0)

    return flow_trading, other_trading


def _carry_earned(hedges: pd.DataFrame,
                  last: pd.Timestamp) -> pd.DataFrame:
    """The carry each class's forwards earned, on the day each was struck:
    its locked-in carry for the share of its term it was held, until the
    class's next hedge trade or the run's end."""
    if hedges.empty:
        return pd.DataFrame(dtype=float)

    rows = []

    for name, trades in hedges.groupby("CLASS"):
        struck = pd.DatetimeIndex(trades.index)
        ends = [*struck[1:], last]

        for day, end, expiry, carry in zip(struck, ends, trades["EXPIRY"], trades["CARRY"],
                                           strict=True):
            term = (pd.Timestamp(expiry) - day).days
            held = min((end - day).days, term)
            rows.append({"DATE": day, "CLASS": name,
                         "CARRY": float(carry) * held / term if term > 0 else 0.0})

    frame = pd.DataFrame(rows)

    return frame.pivot_table(index="DATE", columns="CLASS", values="CARRY", aggfunc="sum")
