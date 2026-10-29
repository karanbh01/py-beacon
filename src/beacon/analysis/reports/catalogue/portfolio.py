# src/beacon/analysis/reports/catalogue/portfolio.py
"""
The statistics of what the fund held and did: its holdings and groups, its
trading and costs, its cash and flows, its income, and an ETF's premium and a
fund's futures.
"""
# BN-398/BN-400.
from collections.abc import Callable
from typing import Any

import pandas as pd

from ..context import FUND, Context
from ..statistics import (
    FIGURES,
    NO_FLOWS,
    NO_FUTURES,
    NO_GROUPING,
    NO_QUOTES,
    NO_SHARE_CLASSES,
    PORTFOLIO,
    TABLE,
    Statistic,
    Summarised,
    Unavailable,
)
from .build import frame

HOLDINGS, TRADING, CASH, INCOME, ETF, FUTURES = ("portfolio", "trading_and_costs",
                                                 "cash_and_flows", "income", "etf", "futures")

GROUPING = ("attribution.grouping",)


def needs(code: str,
          read: Callable[[Context], Any]) -> Callable[[Context], Any]:
    """*read*, or `Unavailable` with *code* when the run lacks the table it
    reads (the portfolio figures raise KeyError naming it)."""
    def compute(context: Context) -> Any:
        try:
            return read(context)
        except KeyError as error:
            raise Unavailable(code) from error

    return compute


def _holdings_count(context: Context) -> pd.DataFrame:
    counts = {book: found for book in (FUND, *context.comparators)
              if (found := context.portfolio.holdings_count(book)) is not None}

    return pd.DataFrame(counts)


def _group_weights(context: Context) -> pd.DataFrame:
    grouping = context.options.attribution.grouping
    held = context.portfolio.group_weights(grouping)
    parts = {FUND: held}

    if context.tracked is not None:
        active = context.portfolio.active_group_weights(grouping, context.tracked)

        if active is not None:
            parts["active"] = active

    return frame(pd.concat(parts, axis=1))


def _turnover(column: str) -> Callable[[Context], Summarised]:
    def compute(context: Context) -> Summarised:
        years = context.portfolio.turnover("year")[[column]]

        return Summarised(years, {"a_year": float(years[column].mean()) if len(years) else None})

    return compute


def _trades(context: Context) -> pd.DataFrame:
    return context.portfolio.turnover("year")[["trades", "buys", "sells", "mean_trade"]]


def _with_flows(read: Callable[[Context], Any]) -> Callable[[Context], Any]:
    def compute(context: Context) -> Any:
        if context.run.investor_flows is None:
            raise Unavailable(NO_FLOWS)

        return read(context)

    return compute


def _premium(context: Context) -> dict[str, Any]:
    found = context.portfolio.etf_premium()

    if found is None:
        raise Unavailable(NO_QUOTES)

    return found


def statistics() -> list[Statistic]:
    """The portfolio statistics, in order."""
    def table(id_: str,
              label: str,
              category: str,
              description: str,
              read: Callable[[Context], Any],
              unit: str | None = None,
              settings: tuple[str, ...] = ()) -> Statistic:
        return Statistic(id_, label, category, TABLE, PORTFOLIO, description, read, unit=unit,
                         settings=settings)

    return [
        table("holdings_count", "Holdings count", HOLDINGS, "How many names the fund and its "
              "comparators held each day.", _holdings_count, unit="count"),
        table("group_weights", "Group weights and active group weights", HOLDINGS,
              "The fund's weight in each group, and less the tracked comparator's.",
              needs(NO_GROUPING, _group_weights), unit="fraction",
              settings=(*GROUPING, "window.tracked")),
        table("characteristics", "Characteristics", HOLDINGS, "Weighted market cap, P/E, P/B "
              "and dividend yield of the fund and its comparators at each rebalance.",
              lambda context: frame(context.portfolio.characteristics())),
        table("one_way_turnover", "One-way turnover", TRADING, "The lesser of buys and sells "
              "over the NAV, each year, and a year on average.", _turnover("one_way"),
              unit="fraction"),
        table("two_way_turnover", "Two-way turnover", TRADING, "All buys and sells over the "
              "NAV, each year, and a year on average.", _turnover("two_way"), unit="fraction"),
        table("trade_counts", "Trades, buys, sells and mean trade", TRADING, "Counts and "
              "values of the trading, each year.", _trades),
        table("costs", "Costs by kind", TRADING, "Fixed costs, market impact, futures rolls "
              "and trading, creation fees received, and the net, in money and in bp of "
              "assets a year.", lambda context: context.portfolio.costs()),
        Statistic("cost_per_traded", "Cost per unit traded", TRADING, FIGURES, PORTFOLIO,
                  "Trading costs in bp of the value traded.",
                  lambda context: context.portfolio.cost_per_traded(), unit="bps",
                  figure_names=("field",)),
        table("cash_band_events", "Cash band events", CASH, "Each time the cash left its "
              "band: raise or spend-down, cash before and after, gap closed, cost.",
              lambda context: context.portfolio.cash_band_events()),
        table("cash_trading", "Cash trading by reason", CASH, "Every trade made to raise or "
              "spend cash, by what it was for.",
              lambda context: frame(context.portfolio.cash_trading())),
        table("aum", "AUM and units", CASH, "The fund's assets and units outstanding each "
              "day.", lambda context: context.portfolio.aum()),
        table("net_flows", "Net flows", CASH, "Each month's flows and their share of "
              "starting AUM.", _with_flows(lambda context: context.portfolio.net_flows())),
        table("flow_costs", "Flow costs", CASH, "The cost of trading each day's flows, less "
              "creation fees received.",
              _with_flows(lambda context: context.portfolio.flow_costs())),
        table("dealing_costs", "Dealing costs", CASH, "Each subscription or redemption: "
              "trading caused, what dealers paid in, net cost to the fund and in bp, and for "
              "an ETF the AP's cost.",
              _with_flows(lambda context: context.portfolio.dealing_costs())),
        table("class_costs", "Costs by share class", CASH, "Each class's fees, hedge costs "
              "less carry earned, its flows' trading, its share of the rest, and the net in "
              "bp.", needs(NO_SHARE_CLASSES, lambda context: context.portfolio.class_costs())),
        table("income_yields", "Income yields", INCOME, "Distribution yield and trailing "
              "dividend yield over 12 months, each day.",
              lambda context: context.portfolio.income(), unit="fraction"),
        table("dividend_growth", "Dividend growth", INCOME, "Each year's dividends and their "
              "growth.", lambda context: context.portfolio.dividend_growth()),
        Statistic("etf_premium", "ETF premium", ETF, FIGURES, PORTFOLIO, "Mean, volatility "
                  "and extremes of the premium to NAV, and the mean spread.", _premium,
                  unit="fraction", figure_names=("field",)),
        table("futures_trading", "Futures trading", FUTURES, "Futures trades by reason, "
              "contracts and notional, cost per contract and in bp, turnover.",
              needs(NO_FUTURES, lambda context: context.portfolio.futures_trading())),
        table("futures_basis", "Futures basis", FUTURES, "The futures' price over their "
              "index, split into carry and mispricing.",
              needs(NO_FUTURES, lambda context: context.portfolio.futures_basis())),
        table("futures_basis_risk", "Futures basis risk", FUTURES, "Tracking error from the "
              "pricing basis and from the underlying mismatch.",
              needs(NO_FUTURES, lambda context: context.portfolio.futures_basis_risk())),
    ]
