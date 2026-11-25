# src/beacon/analysis/analytics/risk.py
"""
A run's risk analytics: volatility and drawdown, the distribution of returns,
value at risk, factors, the risk model, concentration and liquidity.

    risk = result.analytics.portfolio.risk
    risk.volatility()                       # the fund's, over the run
    risk.value_at_risk(of="all")
    risk.herfindahl(as_of="2024-06-28")
"""
# BN-436 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out. The factor, risk model and
# liquidity figures read the data store the run was made from.
from typing import Any

from ..reports.context import FUND
from .category import Category, Day, Windows


class RiskCategory(Category):
    """A run's risk analytics: volatility and drawdown, the distribution of
    returns, value at risk, factors, the risk model, concentration and
    liquidity. Each method is a statistic, named by its id; its keywords pick
    the form it is given in (`beacon.analysis.analytics.forms`)."""

    name = "risk"
    statistics = ("volatility", "max_drawdown", "drawdown", "drawdown_episodes",
                  "risk_rating", "risk_distribution", "downside_deviation", "semi_variance",
                  "skew", "excess_kurtosis", "tail_ratio", "serial_correlation",
                  "return_histogram", "return_distribution", "bootstrap_ranges",
                  "value_at_risk", "expected_shortfall", "stress_windows",
                  "portfolio_risk_return", "efficient_frontier",
                  "factor_exposures", "factor_returns", "factor_betas",
                  "expected_risk", "risk_contributions", "measured_contributions",
                  "diversification_ratio", "average_correlation",
                  "concentration_by_risk", "absorption_ratio", "herfindahl",
                  "top_weights", "concentration_curve", "gini_and_entropy",
                  "concentration_by_outcome", "drift", "days_to_sell")

    # -- volatility and drawdown ----------------------------------------------

    def volatility(self,
                   *,
                   of: str = FUND,
                   start: Day = None,
                   end: Day = None,
                   by: str | None = None,
                   trailing: Windows = None,
                   rolling: int | str | None = None,
                   step: str | None = None,
                   annualise: bool | None = None) -> Any:
        """The annualised standard deviation of daily returns."""
        return self._get("volatility", of=of, start=start, end=end, by=by, trailing=trailing,
                         rolling=rolling, step=step, annualise=annualise)

    def max_drawdown(self,
                     *,
                     of: str = FUND,
                     start: Day = None,
                     end: Day = None,
                     by: str | None = None,
                     trailing: Windows = None,
                     rolling: int | str | None = None,
                     step: str | None = None) -> Any:
        """The deepest fall from a running peak."""
        return self._get("max_drawdown", of=of, start=start, end=end, by=by, trailing=trailing,
                         rolling=rolling, step=step)

    def drawdown(self,
                 *,
                 of: str = FUND,
                 start: Day = None,
                 end: Day = None,
                 as_of: Day = None) -> Any:
        """How far below its running peak each portfolio stood each day: the
        underwater curve."""
        return self._get("drawdown", of=of, start=start, end=end, as_of=as_of)

    def drawdown_episodes(self) -> Any:
        """Each fall from a peak: its depth, start, trough, recovery, and
        how long each part took."""
        return self._get("drawdown_episodes")

    def risk_rating(self,
                    *,
                    against: str | None = None) -> Any:
        """The factsheet's 1 to 7 risk rating, on the volatility of weekly
        returns over the last five years, and the band the tracking error
        puts the fund in, from passive to unconstrained."""
        return self._get("risk_rating", against=against)

    def risk_distribution(self,
                          *,
                          start: Day = None,
                          end: Day = None,
                          against: str | None = None) -> Any:
        """The days the fund spent at each rating and band, on each day's
        volatility and tracking error over the year ending that day."""
        return self._get("risk_distribution", start=start, end=end, against=against)

    # -- distribution ---------------------------------------------------------

    def downside_deviation(self,
                           *,
                           of: str = FUND,
                           start: Day = None,
                           end: Day = None,
                           rolling: int | str | None = None,
                           step: str | None = None) -> Any:
        """The spread of returns below a target (the daily risk-free
        rate)."""
        return self._get("downside_deviation", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def semi_variance(self,
                      *,
                      of: str = FUND,
                      start: Day = None,
                      end: Day = None,
                      rolling: int | str | None = None,
                      step: str | None = None) -> Any:
        """The variance of returns below their mean."""
        return self._get("semi_variance", of=of, start=start, end=end, rolling=rolling, step=step)

    def skew(self,
             *,
             of: str = FUND,
             start: Day = None,
             end: Day = None,
             rolling: int | str | None = None,
             step: str | None = None) -> Any:
        """How lopsided returns are; negative means a longer left tail."""
        return self._get("skew", of=of, start=start, end=end, rolling=rolling, step=step)

    def excess_kurtosis(self,
                        *,
                        of: str = FUND,
                        start: Day = None,
                        end: Day = None,
                        rolling: int | str | None = None,
                        step: str | None = None) -> Any:
        """How fat the tails are beyond a normal distribution's."""
        return self._get("excess_kurtosis", of=of, start=start, end=end, rolling=rolling, step=step)

    def tail_ratio(self,
                   *,
                   of: str = FUND,
                   start: Day = None,
                   end: Day = None,
                   rolling: int | str | None = None,
                   step: str | None = None) -> Any:
        """The 95th percentile return over the size of the 5th."""
        return self._get("tail_ratio", of=of, start=start, end=end, rolling=rolling, step=step)

    def serial_correlation(self,
                           *,
                           of: str = FUND,
                           start: Day = None,
                           end: Day = None,
                           rolling: int | str | None = None,
                           step: str | None = None) -> Any:
        """The correlation of each day's return with the days before."""
        return self._get("serial_correlation", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def return_histogram(self,
                         *,
                         start: Day = None,
                         end: Day = None) -> Any:
        """Daily returns counted into bins, with what a normal distribution
        would put in each."""
        return self._get("return_histogram", start=start, end=end)

    def return_distribution(self,
                            *,
                            start: Day = None,
                            end: Day = None) -> Any:
        """The fund's, the index's and the benchmark's returns counted into
        the same bins, as shares, with the share a normal distribution of
        the fund's would put in each."""
        return self._get("return_distribution", start=start, end=end)

    def bootstrap_ranges(self,
                         *,
                         start: Day = None,
                         end: Day = None) -> Any:
        """Confidence ranges for the main figures from resampled returns."""
        return self._get("bootstrap_ranges", start=start, end=end)

    # -- value at risk --------------------------------------------------------

    def value_at_risk(self,
                      *,
                      of: str = FUND,
                      start: Day = None,
                      end: Day = None,
                      rolling: int | str | None = None,
                      step: str | None = None) -> Any:
        """The loss exceeded one day (or more) in twenty or a hundred:
        historical, normal and Cornish-Fisher."""
        return self._get("value_at_risk", of=of, start=start, end=end, rolling=rolling, step=step)

    def expected_shortfall(self,
                           *,
                           of: str = FUND,
                           start: Day = None,
                           end: Day = None,
                           rolling: int | str | None = None,
                           step: str | None = None) -> Any:
        """The mean loss on the days worse than the value at risk:
        historical and normal."""
        return self._get("expected_shortfall", of=of, start=start, end=end, rolling=rolling,
                         step=step)

    def stress_windows(self) -> Any:
        """Each book's return and deepest fall in each crisis window."""
        return self._get("stress_windows")

    # -- risk and return ------------------------------------------------------

    def portfolio_risk_return(self,
                              *,
                              start: Day = None,
                              end: Day = None) -> Any:
        """The fund's, the index's and the benchmark's annualised return and
        volatility, measured as the members' are."""
        return self._get("portfolio_risk_return", start=start, end=end)

    def efficient_frontier(self,
                           *,
                           start: Day = None,
                           end: Day = None) -> Any:
        """The long-only, fully invested mean-variance frontier of the
        largest latest index members, drawn with the returns the run went on
        to see."""
        return self._get("efficient_frontier", start=start, end=end)

    # -- factors --------------------------------------------------------------

    def factor_exposures(self,
                         *,
                         of: str = FUND,
                         as_of: Day = None) -> Any:
        """Each book's standardised exposure to the factors at each
        rebalance, and the active exposures."""
        return self._get("factor_exposures", of=of, as_of=as_of)

    def factor_returns(self,
                       *,
                       as_of: Day = None) -> Any:
        """Each day's return to each factor from a cap-weighted cross-
        sectional regression."""
        return self._get("factor_returns", as_of=as_of)

    def factor_betas(self,
                     *,
                     start: Day = None,
                     end: Day = None,
                     rolling: int | str | None = None,
                     step: str | None = None) -> Any:
        """Returns-based betas of the fund to the factors, with
        t-statistics, R-squared and alpha."""
        return self._get("factor_betas", start=start, end=end, rolling=rolling, step=step)

    # -- risk model -----------------------------------------------------------

    def expected_risk(self,
                      *,
                      as_of: Day = None) -> Any:
        """From the factor model at each rebalance, split into factor and
        specific."""
        return self._get("expected_risk", as_of=as_of)

    def risk_contributions(self,
                           *,
                           by: str | None = None) -> Any:
        """Each name's marginal contribution, contribution and share of
        expected risk at the last rebalance.

        ``by`` a grouping field of the names, such as ``"sector"``, adds
        them up by the group each name was in that day."""
        return self._get("risk_contributions", by=by)

    def measured_contributions(self,
                               *,
                               start: Day = None,
                               end: Day = None) -> Any:
        """Each name's share of the fund's measured variance over the
        run."""
        return self._get("measured_contributions", start=start, end=end)

    def diversification_ratio(self,
                              *,
                              as_of: Day = None) -> Any:
        """The weighted sum of names' volatilities over the book's, at each
        rebalance."""
        return self._get("diversification_ratio", as_of=as_of)

    def average_correlation(self,
                            *,
                            as_of: Day = None) -> Any:
        """The correlation between holdings the weights and volatilities
        imply, at each rebalance."""
        return self._get("average_correlation", as_of=as_of)

    def concentration_by_risk(self,
                              *,
                              as_of: Day = None) -> Any:
        """Herfindahl of risk shares and effective number of risk sources;
        top names' share of risk."""
        return self._get("concentration_by_risk", as_of=as_of)

    def absorption_ratio(self) -> Any:
        """Share of the held names' total variance in the risk model's
        largest eigenvalues, one per independent factor, at the last
        rebalance."""
        return self._get("absorption_ratio")

    # -- concentration --------------------------------------------------------

    def herfindahl(self,
                   *,
                   by: str | None = None,
                   as_of: Day = None) -> Any:
        """Herfindahl of weights, its normalised form, and the effective
        number of holdings, at each rebalance.

        ``by="month"``, ``"quarter"`` or ``"year"`` gives each period's mean
        of the daily figures."""
        return self._get("herfindahl", by=by, as_of=as_of)

    def top_weights(self,
                    *,
                    as_of: Day = None) -> Any:
        """Largest weight, top 5, 10, 20, 25, and the fewest names making
        50%, 80%, 90%, at each rebalance."""
        return self._get("top_weights", as_of=as_of)

    def concentration_curve(self) -> Any:
        """Names largest first at the last rebalance: each one's weight, the
        cumulative weight to it, and how the top k's concentration changes
        as each is added."""
        return self._get("concentration_curve")

    def gini_and_entropy(self,
                         *,
                         as_of: Day = None) -> Any:
        """Gini coefficient from the Lorenz curve, and entropy with its
        effective number, at each rebalance."""
        return self._get("gini_and_entropy", as_of=as_of)

    def concentration_by_outcome(self,
                                 *,
                                 start: Day = None,
                                 end: Day = None) -> Any:
        """Top contributors' share of the return, and breadth."""
        return self._get("concentration_by_outcome", start=start, end=end)

    def drift(self,
              *,
              as_of: Day = None) -> Any:
        """The share of the book that would trade to return to the last
        rebalance's weights, each day."""
        return self._get("drift", as_of=as_of)

    # -- liquidity ------------------------------------------------------------

    def days_to_sell(self,
                     *,
                     as_of: Day = None) -> Any:
        """Days to sell the book at its participation in daily traded value,
        and the share sellable in 1, 5 and 20 days."""
        return self._get("days_to_sell", as_of=as_of)
