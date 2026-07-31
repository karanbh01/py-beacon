# src/beacon/analysis/attribution.py
"""
Performance attribution: which constituents produced an index's return.

## The identity, and why it needs care

Within a single period the decomposition is exact:

    R_t = Σ_i w_{i,t-1} × r_{i,t}

where ``w`` are the index's own drifting weights. Since BN-103 made the
weighting scheme drive the level, this holds to better than 1e-12 for every
scheme, on rebalance days as well as ordinary ones.

Over multiple periods it does **not** carry over. Returns compound while
contributions add, so the arithmetic sum of daily contributions falls short of
the compounded total — by 1.02 percentage points over a 130-day window on a
modest fixture, which is far too large to write off as a residual.

The fix is Carino linking. Scale each period's contributions by ``k_t / K``
where ``k_t = ln(1+R_t)/R_t`` and ``K = ln(1+R)/R`` for the total return ``R``.
Then

    Σ_i Σ_t (k_t/K) c_{i,t} = (1/K) Σ_t ln(1+R_t) = ln(1+R)/K = R

exactly. The residual is reported regardless, and should sit at machine
epsilon; a residual that is not tiny means an assumption has broken.

## What the drags are, and what they are not

Cap drag and cost drag are **comparisons**, not terms in the identity above.
Each is the difference between two returns — the index as built versus a
counterfactual — so adding them to a decomposition of a single return would be
mixing two different questions. They are reported alongside, each named for the
counterfactual it implies.
"""
import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from ..exceptions import CalculationError
from ..plot.base import PlotAccessor

logger = logging.getLogger(__name__)

# Below this, a period return is treated as zero for the linking coefficient,
# whose limit as R -> 0 is 1. Computing ln(1+R)/R directly there loses
# precision to cancellation.
NEGLIGIBLE_RETURN = 1e-12


@dataclass(frozen=True)
class Contribution:
    """One constituent's share of the total return.

    Attributes:
        asset_id: The constituent.
        contribution: Its linked contribution. These sum to the total return.
        average_weight: Mean weight across the window, for context — a large
            contribution from a small average weight is a different story from
            the same contribution from a large one.
        total_return: The constituent's own return over the window.
    """
    asset_id: str
    contribution: float
    average_weight: float
    total_return: float


@dataclass(frozen=True)
class AttributionResult:
    """A decomposition of one return into per-constituent contributions.

    Attributes:
        start: First date of the window, ISO 8601.
        end: Last date, ISO 8601.
        periods: Return periods decomposed.
        total_return: The return being explained.
        contributions: Per constituent, largest first. Sums to *total_return*
            up to *residual*.
        residual: total_return minus the sum of contributions. Reported
            always, expected to be at machine epsilon after linking. It is
            never folded into a constituent.
        cap_drag: Capped return minus uncapped return, when the index applies
            a cap. Negative when capping cost the index. None when uncapped.
        cost_drag: Portfolio return minus its gross return, when a backtest is
            supplied. Negative by construction — costs only subtract.
    """

    #: Charts for this result. A descriptor that resolves on first
    #: access, so matplotlib is imported only when something is drawn.
    plot = PlotAccessor("AttributionPlots")
    start: str
    end: str
    periods: int
    total_return: float
    contributions: list[Contribution]
    residual: float
    cap_drag: float | None = None
    cost_drag: float | None = None
    _weights: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    @property
    def explained(self) -> float:
        """Sum of the contributions."""
        return float(sum(item.contribution for item in self.contributions))

    def reconciles(self,
                   tolerance: float = 1e-9) -> bool:
        """Whether the contributions account for the total return."""
        return abs(self.residual) <= tolerance

    def to_frame(self) -> pd.DataFrame:
        """Contributions as a DataFrame, largest first."""
        return pd.DataFrame([
            {"asset_id": item.asset_id,
             "contribution": item.contribution,
             "average_weight": item.average_weight,
             "total_return": item.total_return}
            for item in self.contributions
        ])


def carino_factor(period_return: float) -> float:
    """The Carino coefficient for one period.

    ``ln(1+R)/R``, with the removable singularity at R = 0 filled in with its
    limit of 1.

    Args:
        period_return: The period's total return.

    Returns:
        float: The coefficient.

    Raises:
        CalculationError: If the return is -100% or worse, where the logarithm
            is undefined. An index that goes to zero cannot have its return
            attributed, and silently substituting a number would hide that.
    """
    if period_return <= -1.0:
        raise CalculationError(
            "Attribution",
            f"a period return of {period_return:.4%} wipes out the index; the "
            "linking coefficient is undefined there.")

    if abs(period_return) < NEGLIGIBLE_RETURN:
        return 1.0

    return math.log1p(period_return) / period_return


def link_contributions(contributions: pd.DataFrame,
                       period_returns: pd.Series) -> pd.Series:
    """Scale per-period contributions so they sum to the compounded return.

    Args:
        contributions: Periods on the index, constituents on the columns. Each
            row must sum to that period's return.
        period_returns: The total return of each period.

    Returns:
        pd.Series: One linked contribution per constituent. Their sum equals
        the compounded total return exactly.

    Raises:
        CalculationError: If any period return is -100% or worse.
    """
    if contributions.empty:
        return pd.Series(dtype=float)

    total = float((1.0 + period_returns).prod() - 1.0)

    factors = period_returns.map(carino_factor)
    total_factor = carino_factor(total)

    scaled = contributions.mul(factors / total_factor, axis=0)

    return scaled.sum(axis=0)


def drifted_weights(snapshots: dict[pd.Timestamp, dict[str, float]],
                    prices: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the index's daily weights from its rebalance snapshots.

    Weights are set at a rebalance and then drift with relative performance
    until the next one, because the index holds fixed units in between. Given
    the weight at a rebalance and prices since, the drifted weight is

        w_i,t ∝ w_i,rebalance × (p_i,t / p_i,rebalance)

    normalised across constituents. The unit scale cancels, so nothing beyond
    the snapshot and prices is needed.

    Args:
        snapshots: Rebalance date -> weights on that date.
        prices: Dates on the index, constituents on the columns.

    Returns:
        pd.DataFrame: Weights for every date at or after the first rebalance,
        each row summing to 1.

    Raises:
        CalculationError: If there are no snapshots to start from.
    """
    if not snapshots:
        raise CalculationError(
            "Attribution", "the index has no weight snapshots to attribute from.")

    rebalances = sorted(snapshots)
    rows: dict[pd.Timestamp, dict[str, float]] = {}

    for date in prices.index:
        active = [r for r in rebalances if r <= date]
        if not active:
            continue

        rows[date] = _weights_on(snapshots[active[-1]], prices, active[-1], date)

    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def _weights_on(snapshot: dict[str, float],
                prices: pd.DataFrame,
                rebalance: pd.Timestamp,
                date: pd.Timestamp) -> dict[str, float]:
    """Drift one snapshot forward to *date* using relative price moves."""
    values: dict[str, float] = {}

    for asset_id, weight in snapshot.items():
        if asset_id not in prices.columns:
            continue

        base = prices.at[rebalance, asset_id]
        current = prices.at[date, asset_id]

        if pd.isna(base) or pd.isna(current) or base == 0:
            continue

        values[asset_id] = weight * (current / base)

    total = sum(values.values())
    if total <= 0:
        return dict.fromkeys(values, 0.0)

    return {asset_id: value / total for asset_id, value in values.items()}


def attribute(period_returns: pd.Series,
              weights: pd.DataFrame,
              asset_returns: pd.DataFrame,
              cap_drag: float | None = None,
              cost_drag: float | None = None) -> AttributionResult:
    """Decompose a return series into per-constituent contributions.

    Args:
        period_returns: The return being explained, per period.
        weights: Weights per period, constituents on the columns. Aligned to
            *period_returns*; the weight used for a period is the one held at
            its start.
        asset_returns: Constituent returns per period.
        cap_drag: Optional capped-minus-uncapped return.
        cost_drag: Optional cost effect on the portfolio return.

    Returns:
        AttributionResult: The decomposition, with contributions summing to the
        compounded total return and a residual reported separately.

    Raises:
        CalculationError: If the inputs cannot be aligned, or a period wipes
            out the index.
    """
    common = period_returns.index.intersection(weights.index).intersection(
        asset_returns.index).sort_values()

    if len(common) == 0:
        raise CalculationError(
            "Attribution",
            "the return, weight and constituent-return series share no dates.")

    returns = period_returns.loc[common]
    assets = sorted(set(weights.columns) & set(asset_returns.columns))

    # The weight that earns a period's return is the one held at its start,
    # hence the shift. Using the end-of-period weight would credit a
    # constituent for a move it was not yet holding.
    lagged = weights[assets].reindex(common).shift(1)
    contributions = lagged * asset_returns[assets].reindex(common)
    contributions = contributions.dropna(how="all")

    linked = link_contributions(contributions, returns.loc[contributions.index])
    total = float((1.0 + returns.loc[contributions.index]).prod() - 1.0)

    rows = [
        Contribution(asset_id=asset_id,
                     contribution=float(linked.get(asset_id, 0.0)),
                     average_weight=float(lagged[asset_id].mean()),
                     total_return=float(
                         (1.0 + asset_returns[asset_id].reindex(
                             contributions.index).fillna(0.0)).prod() - 1.0))
        for asset_id in assets
    ]
    rows.sort(key=lambda item: item.contribution, reverse=True)

    return AttributionResult(
        start=contributions.index[0].isoformat(),
        end=contributions.index[-1].isoformat(),
        periods=len(contributions),
        total_return=total,
        contributions=rows,
        residual=total - float(linked.sum()),
        cap_drag=cap_drag,
        cost_drag=cost_drag,
        _weights=lagged)


def cost_drag(total_costs: float,
              initial_capital: float) -> float:
    """Direct effect of transaction costs on a portfolio's return.

    Costs paid as a fraction of starting capital, negated so it reads as a
    drag. This is the **direct** effect only: it excludes the compounding of
    the capital that was spent rather than invested, which is second-order but
    not zero over a long window. Reporting the direct figure keeps the number
    explainable — it is exactly the money that left the portfolio — and a
    caller wanting the full effect can difference a zero-cost run instead.

    Args:
        total_costs: Sum of transaction costs paid.
        initial_capital: Capital the portfolio started with.

    Returns:
        float: A non-positive drag.

    Raises:
        CalculationError: If *initial_capital* is not positive.
    """
    if initial_capital <= 0:
        raise CalculationError(
            "CostDrag", f"initial_capital must be positive, got {initial_capital}.")

    return -abs(total_costs) / initial_capital


def cap_drag(capped_weights: dict[pd.Timestamp, dict[str, float]],
             uncapped_weights: dict[pd.Timestamp, dict[str, float]],
             prices: pd.DataFrame) -> float:
    """What capping cost, or gained, over the window.

    The capped index's return minus the return of the same methodology left
    uncapped. Negative when the cap held back a name that went on to
    outperform, which is the usual case and the reason the number is worth
    reporting.

    Both paths are built by drifting their own snapshots forward, so the
    comparison isolates the effect of the cap rather than of any other
    difference.

    Args:
        capped_weights: Rebalance date -> capped weights.
        uncapped_weights: Rebalance date -> weights before capping.
        prices: Constituent prices over the window.

    Returns:
        float: Capped total return minus uncapped total return.
    """
    asset_returns = prices.pct_change()

    capped_return = _path_return(capped_weights, prices, asset_returns)
    uncapped_return = _path_return(uncapped_weights, prices, asset_returns)

    return capped_return - uncapped_return


def _path_return(snapshots: dict[pd.Timestamp, dict[str, float]],
                 prices: pd.DataFrame,
                 asset_returns: pd.DataFrame) -> float:
    """Compounded return of an index following *snapshots*."""
    weights = drifted_weights(snapshots, prices)
    assets = sorted(set(weights.columns) & set(asset_returns.columns))

    periods = (weights[assets].shift(1) * asset_returns[assets]).sum(axis=1)
    periods = periods.loc[weights[assets].shift(1).dropna(how="all").index]

    return float((1.0 + periods).prod() - 1.0)


class Attribution:
    """Kept for the original portfolio-versus-benchmark helper."""

    def simple_performance_attribution(self,
                                       portfolio_returns: pd.Series,
                                       benchmark_returns: pd.Series) -> dict[str, float]:
        """Total return difference between a portfolio and a benchmark.

        Args:
            portfolio_returns: Portfolio periodic returns.
            benchmark_returns: Benchmark periodic returns, same length.

        Returns:
            dict: total_portfolio_return, total_benchmark_return and
            active_return.

        Raises:
            TypeError: If either input is not a Series.
            ValueError: If the lengths differ or the inputs are empty.
        """
        if (not isinstance(portfolio_returns, pd.Series)
                or not isinstance(benchmark_returns, pd.Series)):
            raise TypeError("portfolio_returns and benchmark_returns must be pandas Series.")
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError(
                "Portfolio returns and benchmark returns Series must be of the same length.")
        if portfolio_returns.empty:
            raise ValueError("Input Series cannot be empty.")

        total_portfolio_return = (1 + portfolio_returns).prod() - 1
        total_benchmark_return = (1 + benchmark_returns).prod() - 1

        return {
            "total_portfolio_return": float(total_portfolio_return),
            "total_benchmark_return": float(total_benchmark_return),
            "active_return": float(total_portfolio_return - total_benchmark_return),
        }


def simple_performance_attribution(portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> dict[str, float]:
    """Total return difference between a portfolio and a benchmark."""
    return Attribution().simple_performance_attribution(portfolio_returns,
                                                        benchmark_returns)
