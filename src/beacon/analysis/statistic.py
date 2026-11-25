# src/beacon/analysis/statistic.py
"""
Every statistic's id, as an enum, so ids autocomplete and a typo fails
where it is written:

    from beacon.analysis import Statistic

    AnalysisRequest(statistics=[Statistic.SHARPE_RATIO, Statistic.TRACKING_ERROR])

Each member is its id, a `str`, so a member goes wherever an id does, and
``Statistic("sharpe_ratio")`` gives the member for an id.
"""
# BN-432 (Karan, 2026-11-24). A test holds the members to the catalogue,
# in its order, with the catalogue's categories as headings.
from enum import StrEnum


class Statistic(StrEnum):
    """The id of every statistic a run's analysis gives."""
    # Returns
    TOTAL_RETURN = "total_return"
    ANNUALISED_RETURN = "annualised_return"
    MONEY_WEIGHTED_RETURN = "money_weighted_return"
    TRAILING_RETURNS = "trailing_returns"
    TRAILING_STATISTICS = "trailing_statistics"
    PERIOD_RETURNS = "period_returns"
    BEST_AND_WORST = "best_and_worst"
    POSITIVE_SHARE = "positive_share"
    GAINS_AND_LOSSES = "gains_and_losses"
    STREAKS = "streaks"
    MEAN_RETURN = "mean_return"
    HIT_RATE = "hit_rate"
    MONTH_OF_YEAR = "month_of_year"

    # Risk
    VOLATILITY = "volatility"

    # Drawdown
    MAX_DRAWDOWN = "max_drawdown"
    DRAWDOWN = "drawdown"
    DRAWDOWN_EPISODES = "drawdown_episodes"

    # Distribution
    DOWNSIDE_DEVIATION = "downside_deviation"
    SEMI_VARIANCE = "semi_variance"
    SKEW = "skew"
    EXCESS_KURTOSIS = "excess_kurtosis"
    TAIL_RATIO = "tail_ratio"
    SERIAL_CORRELATION = "serial_correlation"
    RETURN_HISTOGRAM = "return_histogram"
    RETURN_DISTRIBUTION = "return_distribution"
    BOOTSTRAP_RANGES = "bootstrap_ranges"

    # Risk-adjusted
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    KAPPA_3 = "kappa_3"
    UPSIDE_POTENTIAL_RATIO = "upside_potential_ratio"
    MARTIN_RATIO = "martin_ratio"
    STERLING_RATIO = "sterling_ratio"
    BURKE_RATIO = "burke_ratio"
    GAIN_TO_PAIN = "gain_to_pain"
    PROBABILISTIC_SHARPE_RATIO = "probabilistic_sharpe_ratio"
    DEFLATED_SHARPE_RATIO = "deflated_sharpe_ratio"
    LO_SHARPE_RATIO = "lo_sharpe_ratio"

    # Risk
    RISK_RATING = "risk_rating"
    RISK_DISTRIBUTION = "risk_distribution"
    MEMBER_RISK_RETURN = "member_risk_return"
    PORTFOLIO_RISK_RETURN = "portfolio_risk_return"
    EFFICIENT_FRONTIER = "efficient_frontier"

    # Relative
    TRACKING_DIFFERENCE = "tracking_difference"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"
    INFORMATION_RATIO_T = "information_ratio_t"
    ANNUALISED_RELATIVE_RETURN = "annualised_relative_return"
    BETA = "beta"
    CORRELATION = "correlation"
    ALPHA = "alpha"
    R_SQUARED = "r_squared"
    RESIDUAL_RISK = "residual_risk"
    TREYNOR_RATIO = "treynor_ratio"
    CAPTURE = "capture"
    BATTING_AVERAGE = "batting_average"
    DOWNSIDE_TRACKING_ERROR = "downside_tracking_error"
    RELATIVE_DRAWDOWNS = "relative_drawdowns"
    TRACKING_DIFFERENCE_BY_YEAR = "tracking_difference_by_year"

    # Rolling
    ROLLING_OUTCOMES = "rolling_outcomes"

    # Crises and regimes
    CRISIS_RETURNS = "crisis_returns"
    CRISIS_PATHS = "crisis_paths"
    CRISIS_SUMMARY = "crisis_summary"
    REGIMES = "regimes"

    # Portfolio
    HOLDINGS_COUNT = "holdings_count"
    HOLDINGS_RANGE = "holdings_range"
    HOLDINGS_DISTRIBUTION = "holdings_distribution"
    GROUP_WEIGHTS = "group_weights"
    GROUP_EXPOSURES = "group_exposures"
    CHARACTERISTICS = "characteristics"

    # Trading and costs
    ONE_WAY_TURNOVER = "one_way_turnover"
    TWO_WAY_TURNOVER = "two_way_turnover"
    TRADE_COUNTS = "trade_counts"
    COSTS = "costs"
    COST_PER_TRADED = "cost_per_traded"

    # Cash and flows
    CASH_BAND_EVENTS = "cash_band_events"
    CASH_TRADING = "cash_trading"
    AUM = "aum"
    NET_FLOWS = "net_flows"
    FLOW_COSTS = "flow_costs"
    DEALING_COSTS = "dealing_costs"
    CLASS_COSTS = "class_costs"

    # Income
    INCOME_YIELDS = "income_yields"
    DIVIDEND_GROWTH = "dividend_growth"

    # ETF
    ETF_PREMIUM = "etf_premium"

    # Futures
    FUTURES_TRADING = "futures_trading"
    FUTURES_BASIS = "futures_basis"
    FUTURES_BASIS_RISK = "futures_basis_risk"

    # Portfolio
    STYLE_POSITIONS = "style_positions"
    STYLE_BOX = "style_box"
    STYLE_CLASSIFICATION = "style_classification"

    # Factors
    FACTOR_EXPOSURES = "factor_exposures"
    FACTOR_RETURNS = "factor_returns"
    FACTOR_BETAS = "factor_betas"

    # Value at risk
    VALUE_AT_RISK = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    STRESS_WINDOWS = "stress_windows"

    # Risk contributions
    EXPECTED_RISK = "expected_risk"
    FORECAST_CHECK = "forecast_check"
    BIAS_TEST = "bias_test"
    RISK_CONTRIBUTIONS = "risk_contributions"
    MEASURED_CONTRIBUTIONS = "measured_contributions"
    DIVERSIFICATION_RATIO = "diversification_ratio"
    AVERAGE_CORRELATION = "average_correlation"

    # Concentration
    HERFINDAHL = "herfindahl"
    TOP_WEIGHTS = "top_weights"
    CONCENTRATION_CURVE = "concentration_curve"
    GINI_AND_ENTROPY = "gini_and_entropy"
    CONCENTRATION_BY_RISK = "concentration_by_risk"
    ABSORPTION_RATIO = "absorption_ratio"
    CONCENTRATION_BY_OUTCOME = "concentration_by_outcome"
    DRIFT = "drift"

    # Liquidity
    DAYS_TO_SELL = "days_to_sell"

    # Attribution
    CONTRIBUTION = "contribution"
    ACTIVE_ATTRIBUTION = "active_attribution"
    BRINSON = "brinson"
    CURRENCY_ATTRIBUTION = "currency_attribution"
    FACTOR_ATTRIBUTION = "factor_attribution"
    IMPLEMENTATION_BREAKDOWN = "implementation_breakdown"
