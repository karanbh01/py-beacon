# src/beacon/analysis/reports/catalogue/columns.py
"""
The columns of the table statistics whose columns are in different units,
each with its label and unit, so a client formats a table from the
catalogue rather than guessing. A table whose columns are all in one unit
(a column a portfolio, a factor or a group) says so with the statistic's
own ``unit`` instead.
"""
# BN-455 (beacon-ui, 2026-11-24). The rest of the tables list their columns
# where they are defined.
from ....backtest.cash_band import RAISE, SPEND_DOWN
from ....catalogue import BPS, COUNT, DAYS, FRACTION, MONEY, RATIO
from ....portfolio.cash_flows import REDEMPTION, SUBSCRIPTION
from ...risk.liquidity import HORIZONS
from ..statistics import Column

BAND_KINDS = {RAISE: "Raise", SPEND_DOWN: "Spend-down", "all": "All"}
FLOW_KINDS = {SUBSCRIPTION: "Subscription", REDEMPTION: "Redemption"}

WEIGHT_FIGURES = (Column("names", "Names", COUNT), Column("herfindahl", "Herfindahl", RATIO),
                  Column("normalised_herfindahl", "Normalised Herfindahl", RATIO),
                  Column("effective_number", "Effective number", COUNT))
TOP_WEIGHTS = (Column("largest", "Largest weight", FRACTION),
               *(Column(f"top_{k}", f"Top {k}", FRACTION) for k in (5, 10, 20, 25)),
               *(Column(f"names_for_{share}", f"Names for {share}%", COUNT)
                 for share in (50, 80, 90)))
FLOW_MONEY = (Column("flow", "Flow", MONEY), Column("trading_cost", "Trading cost", MONEY),
              Column("fees_received", "Fees received", MONEY),
              Column("net_cost", "Net cost", MONEY))

COLUMNS: dict[str, tuple[Column, ...]] = {
    "trade_counts": (Column("trades", "Trades", COUNT), Column("buys", "Bought", MONEY),
                     Column("sells", "Sold", MONEY), Column("mean_trade", "Mean trade", MONEY)),
    "costs": (Column("total", "Total", MONEY), Column("bps_a_year", "bp a year", BPS)),
    "cash_band_events": (
        Column("kind", "Kind", labels=BAND_KINDS), Column("cash_before", "Cash before", FRACTION),
        Column("cash_after", "Cash after", FRACTION), Column("target", "Target", FRACTION),
        Column("lower", "Lower edge", FRACTION), Column("upper", "Upper edge", FRACTION),
        Column("traded", "Traded", MONEY), Column("cost", "Cost", MONEY),
        Column("trades", "Trades", COUNT), Column("gap_before", "Gap before", FRACTION),
        Column("gap_after", "Gap after", FRACTION), Column("gap_closed", "Gap closed", FRACTION),
        Column("cost_bps", "Cost", BPS)),
    "cash_band": (
        Column("events", "Events", COUNT), Column("traded", "Traded", MONEY),
        Column("cost", "Cost", MONEY), Column("mean_cost", "Mean cost", MONEY),
        Column("cost_bps", "Cost", BPS), Column("mean_gap_closed", "Mean gap closed", FRACTION),
        Column("mean_gap_after", "Mean gap left", FRACTION)),
    "cash_trading": (Column("trades", "Trades", COUNT), Column("traded", "Traded", MONEY),
                     Column("cost", "Cost", MONEY)),
    "aum": (Column("aum", "Assets", MONEY), Column("units", "Units", COUNT)),
    "net_flows": (Column("net_flows", "Net flows", MONEY),
                  Column("share_of_aum", "Share of assets", FRACTION)),
    "flow_costs": FLOW_MONEY,
    "dealing_costs": (Column("kind", "Kind", labels=FLOW_KINDS), *FLOW_MONEY,
                      Column("net_bps", "Net cost", BPS), Column("ap_cost", "AP's cost", MONEY)),
    "class_costs": (
        Column("fees", "Fees", MONEY), Column("hedge_costs", "Hedge costs", MONEY),
        Column("hedge_carry", "Hedge carry", MONEY), Column("flow_trading", "Flow trading", MONEY),
        Column("other_trading", "Other trading", MONEY),
        Column("dealing_received", "Dealing received", MONEY),
        Column("net_cost", "Net cost", MONEY), Column("net_bps_a_year", "Net bp a year", BPS)),
    "dividend_growth": (Column("dividends", "Dividends", MONEY),
                        Column("growth", "Growth", FRACTION)),
    "futures_trading": (
        Column("trades", "Trades", COUNT), Column("contracts", "Contracts", COUNT),
        Column("notional", "Notional", MONEY), Column("cost", "Cost", MONEY),
        Column("cost_per_contract", "Cost a contract", MONEY), Column("cost_bps", "Cost", BPS),
        Column("turnover", "Turnover", FRACTION)),
    "futures_basis": (Column("mean_bps", "Mean", BPS), Column("volatility_bps", "Volatility", BPS),
                      Column("lowest_bps", "Lowest", BPS), Column("highest_bps", "Highest", BPS)),
    "futures_basis_risk": (Column("mean_a_year", "Mean a year", FRACTION),
                           Column("tracking_error", "Tracking error", FRACTION),
                           Column("days", "Days", DAYS)),
    "forecast_check": (
        Column("days", "Days", DAYS),
        Column("expected_volatility", "Expected volatility", FRACTION),
        Column("realised_volatility", "Realised volatility", FRACTION),
        Column("volatility_ratio", "Realised over expected", RATIO),
        Column("expected_tracking_error", "Expected tracking error", FRACTION),
        Column("realised_tracking_error", "Realised tracking error", FRACTION),
        Column("tracking_error_ratio", "Realised over expected", RATIO)),
    "risk_contributions": (Column("weight", "Weight", FRACTION),
                           Column("marginal", "Marginal contribution", FRACTION),
                           Column("contribution", "Contribution", FRACTION),
                           Column("share", "Share of risk", FRACTION)),
    "measured_contributions": (Column("share", "Share of risk", FRACTION),
                               Column("volatility", "Volatility", FRACTION)),
    "herfindahl": WEIGHT_FIGURES,
    "top_weights": TOP_WEIGHTS,
    "concentration_curve": (
        Column("name", "Name"), Column("weight", "Weight", FRACTION),
        Column("cumulative", "Cumulative weight", FRACTION),
        Column("top_k_herfindahl", "Herfindahl of the top k", RATIO),
        Column("top_k_effective_number", "Effective number of the top k", COUNT),
        Column("marginal_herfindahl", "Herfindahl added", RATIO),
        Column("marginal_effective_number", "Effective number added", COUNT)),
    "crisis_concentration": (*WEIGHT_FIGURES, *TOP_WEIGHTS),
    "gini_and_entropy": (Column("gini", "Gini coefficient", RATIO),
                         Column("entropy", "Entropy", RATIO),
                         Column("entropy_effective_number", "Effective number", COUNT)),
    "concentration_by_risk": (
        Column("risk_herfindahl", "Herfindahl of risk", RATIO),
        Column("effective_risk_sources", "Effective risk sources", COUNT),
        *(Column(f"top_{k}_risk", f"Top {k}'s risk", FRACTION) for k in (5, 10, 20, 25)),
        Column("diversification_ratio", "Diversification ratio", RATIO),
        Column("average_correlation", "Average correlation", RATIO)),
    # Its sellable shares follow the horizons the liquidity options set;
    # these are the defaults'.
    "days_to_sell": (*(Column(f"sellable_{days}d", f"Sellable in {days} days", FRACTION)
                       for days in HORIZONS),
                     Column("days_to_sell_all", "Days to sell all", DAYS),
                     Column("mean_days", "Mean days", DAYS),
                     Column("no_traded_value", "No traded value", FRACTION)),
}

__all__ = ["COLUMNS"]
