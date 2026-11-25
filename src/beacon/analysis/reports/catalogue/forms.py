# src/beacon/analysis/reports/catalogue/forms.py
"""
The keywords each statistic's analytic takes, which pick the form it is
given in (`result.analytics`):

- ``of``: the portfolio measured, for figures given a portfolio each and
  tables with a column each;
- ``start`` and ``end``: the window, for a figure of a window of returns;
- ``by``: calendar periods, or the groups a table is split by, such as a
  trade's reason (`GROUPINGS`);
- ``trailing``: windows ending on the last day;
- ``rolling`` (and ``step``): a rolling window, for a statistic that rolls;
- ``against``: the comparator, for a relative figure;
- ``as_of``: one day, for a statistic that runs through time;
- ``annualise``: whether an annualised figure is given over any window
  (True), per day (False), or only over a window long enough for it (None,
  `beacon.analysis.annualising`);
- ``grid``: the monthly returns as a calendar grid, years by months.
"""
# BN-434 (Karan, 2026-11-24). The sets are the formats each statistic was
# given in when the 0.11.0 result API was agreed.
from ...annualising import ANNUALISED
from ..statistics import FIGURES, StatisticSpec

OF, START, END, BY, TRAILING, ROLLING, STEP, AGAINST, AS_OF, ANNUALISE, GRID = (
    "of", "start", "end", "by", "trailing", "rolling", "step", "against", "as_of", "annualise",
    "grid")
KEYWORDS = (OF, START, END, BY, TRAILING, ROLLING, STEP, AGAINST, AS_OF, ANNUALISE, GRID)

BOOK, COMPARATOR = "book", "comparator"


def _ids(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Figures of a window of returns, which any window can be measured over.
WINDOW = _ids("""
    total_return annualised_return money_weighted_return best_and_worst positive_share
    gains_and_losses streaks mean_return hit_rate volatility max_drawdown drawdown
    downside_deviation
    semi_variance skew excess_kurtosis tail_ratio serial_correlation return_histogram
    return_distribution bootstrap_ranges sharpe_ratio sortino_ratio calmar_ratio omega_ratio
    kappa_3 upside_potential_ratio martin_ratio sterling_ratio burke_ratio gain_to_pain
    probabilistic_sharpe_ratio deflated_sharpe_ratio lo_sharpe_ratio risk_distribution
    member_risk_return portfolio_risk_return efficient_frontier tracking_difference
    tracking_error information_ratio information_ratio_t annualised_relative_return beta
    correlation alpha r_squared residual_risk treynor_ratio capture batting_average
    downside_tracking_error rolling_outcomes crisis_summary regimes holdings_range
    holdings_distribution one_way_turnover two_way_turnover trade_counts costs cost_per_traded
    class_costs etf_premium futures_trading futures_basis futures_basis_risk factor_betas
    value_at_risk expected_shortfall measured_contributions concentration_by_outcome
    contribution active_attribution brinson currency_attribution factor_attribution
    implementation_breakdown""")

# Figures given in each calendar period.
PERIODS = _ids("""
    total_return volatility sharpe_ratio max_drawdown positive_share annualised_return hit_rate
    tracking_difference annualised_relative_return tracking_error information_ratio
    correlation beta batting_average""")

# Figures given over each trailing window.
TRAILING_WINDOWS = _ids("""
    total_return annualised_return volatility max_drawdown sharpe_ratio tracking_difference
    tracking_error""")

# Tables with a row each day or rebalance.
THROUGH_TIME = _ids("""
    holdings_count group_weights characteristics aum flow_costs income_yields factor_exposures
    drawdown
    factor_returns expected_risk diversification_ratio average_correlation herfindahl
    top_weights gini_and_entropy concentration_by_risk drift days_to_sell""")

# Statistics measured against a comparator, whose figures are not given
# one per comparator.
AGAINST_TRACKED = _ids("""
    risk_rating risk_distribution relative_drawdowns rolling_outcomes crisis_returns
    crisis_summary regimes""")

# Dated tables with a column for each portfolio, or each portfolio's fields.
BY_BOOK_TABLES = _ids("holdings_count characteristics factor_exposures drawdown")

# Figures given for the fund alone, never its index or benchmark.
FUND_ONLY = _ids("money_weighted_return")

# BN-446: tables split by groups of their own rather than by windowing the
# run, and what ``by`` takes for each. A grouping field of the run's names
# (`FIELD`), such as ``"sector"``, is any the run kept.
FIELD = "field"
GROUPINGS: dict[str, tuple[str, ...]] = {
    "costs": ("reason", "year"),
    "trade_counts": ("reason",),
    "dealing_costs": ("kind", "year"),
    "brinson": ("month", "quarter", "year"),
    "herfindahl": ("month", "quarter", "year"),
    "risk_contributions": (FIELD,),
}

# Statistics given as a calendar grid too.
GRIDDED = _ids("total_return")


def forms_of(statistic: StatisticSpec) -> tuple[str, ...]:
    """The keywords *statistic* takes, in `KEYWORDS`'s order."""
    names = statistic.figure_names
    first = names[0] if names else None
    found = set()

    if ((statistic.kind == FIGURES and first == BOOK and statistic.id not in FUND_ONLY)
            or statistic.id in BY_BOOK_TABLES):
        found.add(OF)

    if statistic.id in WINDOW:
        found.update((START, END))

    if statistic.id in PERIODS or statistic.id in GROUPINGS:
        found.add(BY)

    if statistic.id in TRAILING_WINDOWS:
        found.add(TRAILING)

    if statistic.rolling is not None:
        found.update((ROLLING, STEP))

    if (statistic.kind == FIGURES and first == COMPARATOR) or statistic.id in AGAINST_TRACKED:
        found.add(AGAINST)

    if statistic.id in THROUGH_TIME:
        found.add(AS_OF)

    # BN-465.
    if statistic.id in ANNUALISED:
        found.add(ANNUALISE)

    if statistic.id in GRIDDED:
        found.add(GRID)

    return tuple(keyword for keyword in KEYWORDS if keyword in found)
