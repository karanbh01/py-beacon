# src/beacon/analysis/placement.py
"""
Where each statistic sits in a run's analytics: one of the nine categories
of `result.analytics.portfolio`, or ``instrument`` for a row a name. A
statistic's ``part`` is its place, which the analysis works out together and
times together.
"""
# BN-436 (Karan, 2026-11-24): the parts follow the result API's categories,
# so the analytics, the reports and the API agree.
PERFORMANCE, RISK, RELATIVE, HOLDINGS, TRADING, FLOWS, INCOME, VEHICLE, ATTRIBUTION = (
    "performance", "risk", "relative", "holdings", "trading", "flows", "income", "vehicle",
    "attribution")
INSTRUMENT = "instrument"

# A whole portfolio's categories, in the order they are read.
CATEGORIES = (PERFORMANCE, RISK, RELATIVE, HOLDINGS, TRADING, FLOWS, INCOME, VEHICLE,
              ATTRIBUTION)
PARTS = (*CATEGORIES, INSTRUMENT)

# Each place's statistics. Those folded into another's form (trailing_returns
# is total_return(trailing=...)) and those the result already holds (aum) are
# kept here too, for the reports that show them.
PLACED = {
    PERFORMANCE: """
        total_return annualised_return money_weighted_return mean_return best_and_worst
        positive_share hit_rate gains_and_losses streaks month_of_year sharpe_ratio
        sortino_ratio calmar_ratio omega_ratio kappa_3 upside_potential_ratio martin_ratio
        sterling_ratio burke_ratio gain_to_pain probabilistic_sharpe_ratio deflated_sharpe_ratio
        lo_sharpe_ratio rolling_outcomes crisis_returns crisis_paths crisis_summary regimes
        trailing_returns trailing_statistics period_returns""",
    RISK: """
        volatility max_drawdown drawdown drawdown_episodes risk_rating risk_distribution
        downside_deviation semi_variance skew excess_kurtosis tail_ratio serial_correlation
        return_histogram return_distribution bootstrap_ranges value_at_risk expected_shortfall
        stress_windows portfolio_risk_return efficient_frontier factor_exposures factor_returns
        factor_betas expected_risk forecast_check bias_test risk_contributions
        measured_contributions
        diversification_ratio average_correlation concentration_by_risk absorption_ratio
        herfindahl top_weights concentration_curve crisis_concentration gini_and_entropy
        concentration_by_outcome
        drift days_to_sell""",
    RELATIVE: """
        tracking_difference tracking_error information_ratio information_ratio_t
        annualised_relative_return beta correlation alpha r_squared residual_risk treynor_ratio
        capture batting_average downside_tracking_error relative_drawdowns
        tracking_difference_by_year""",
    HOLDINGS: """
        holdings_count holdings_range holdings_distribution characteristics group_weights
        group_exposures style_box style_classification""",
    TRADING: """
        one_way_turnover two_way_turnover trade_counts costs cost_per_traded cash_trading
        cash_band cash_band_events""",
    FLOWS: "net_flows flow_costs dealing_costs class_costs aum",
    INCOME: "income_yields dividend_growth",
    VEHICLE: "etf_premium futures_trading futures_basis futures_basis_risk",
    ATTRIBUTION: """
        contribution active_attribution brinson currency_attribution factor_attribution
        implementation_breakdown""",
    INSTRUMENT: "member_risk_return style_positions",
}

PART_OF = {statistic_id: part for part, ids in PLACED.items() for statistic_id in ids.split()}

# Statistics given as a form of another's analytic, with the call that gives
# them; their ids stay, for the reports.
FOLDED = {"trailing_returns": "performance.total_return(trailing=[...])",
          "period_returns": 'performance.total_return(by="month")',
          "tracking_difference_by_year": 'relative.tracking_difference(by="year")',
          "trailing_statistics": "analytics.table([...], trailing=[...])"}

# Statistics not repeated as an analytic, being what the result records.
DROPPED = {"aum": "result.fund.aum", "cash_band_events": "result.fund.cash_band.events"}
