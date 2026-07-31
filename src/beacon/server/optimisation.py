# src/beacon/server/optimisation.py
"""
Running an optimisation, and reading the frontier and exposures off it.

A solve is fast; estimating the risk model it needs is not, because that means
pulling a price history for every name and building a covariance. So a run is a
job like a backtest, and the result carries enough that the frontier and
exposures panes read it rather than re-solving.

## Where the inputs come from

* **Target weights** — the index's latest completed run, via its rebalance
  snapshots. Optimising against an index nobody has calculated is not a thing
  that can be done, so it is a 404 rather than a silent default.
* **Risk model** — estimated from the constituents' own price history over the
  run's window. Shrunk toward constant correlation, because a covariance
  estimated on a few hundred observations across a similar number of names is
  badly conditioned and an optimiser inverts it.
* **Expected returns** — the historical mean, annualised, and this is a
  modelling choice worth stating plainly: **historical mean returns are a poor
  forecast**. They are used because they are the only return estimate derivable
  from the data the server holds, and because a frontier has to be drawn
  against something. A caller with a real forecast should supply it. The field
  says so.

## Factor exposures without a factor file

Exposures need loadings, and there is no fundamentals data (that is the
features layer, still to be designed). So the factors here are the ones that
*are* derivable from price and share count:

* **size** — log market capitalisation
* **momentum** — trailing return, excluding the most recent month
* **volatility** — trailing standard deviation of returns

Value and quality are absent rather than approximated. A momentum factor built
from prices is the real thing; a value factor faked without book values would
not be, and labelling one as such would be worse than not having it.
"""
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError
from ..optimise import minimise_tracking_error
from ..optimise.frontier import efficient_frontier
from ..risk.factors import fit_factor_model, z_scores
from ..risk.model import estimate_risk_model
from .jobs import ProgressReporter
from .schemas import (
    ConstraintSet,
    ExposuresView,
    FactorExposure,
    FrontierPoint,
    FrontierView,
    OptimisationRunRequest,
    OptimisationRunResult,
    RiskDecomposition,
    WeightRow,
)

logger = logging.getLogger(__name__)

# Trading days per year, for annualising the estimates.
PERIODS_PER_YEAR = 252

# Momentum skips the most recent month, the conventional construction: the last
# few weeks of a price series carry short-term reversal, which is a different
# effect pointing the other way.
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21

# Shrinkage toward constant correlation. A covariance estimated on a few
# hundred observations across a similar number of names is badly conditioned,
# and an optimiser inverts it.
SHRINKAGE = 0.2

FRONTIER_POINTS = 15


def constituent_prices(fetcher: DataFetcher,
                       identifiers: list[str],
                       start: str | None = None,
                       end: str | None = None) -> pd.DataFrame:
    """Close prices for a set of names, names on the columns.

    Raises:
        DataNotFoundError: If none of them can be priced.
    """
    series: dict[str, pd.Series] = {}

    for identifier in identifiers:
        frame = fetcher.fetch_market_data(identifier, start, end)
        if not frame.empty and "CLOSE" in frame.columns:
            series[identifier] = frame["CLOSE"]

    if not series:
        raise DataNotFoundError("prices for any of these identifiers",
                                source="MarketData")

    return pd.DataFrame(series).sort_index()


def expected_returns_from(prices: pd.DataFrame) -> dict[str, float]:
    """Annualised historical mean returns.

    A poor forecast, and deliberately the honest one: it is what the data
    supports. Anything more sophisticated invented here would look like a view
    the server does not have.
    """
    daily = prices.pct_change().dropna(how="all").mean()

    return {str(name): float(value) * PERIODS_PER_YEAR
            for name, value in daily.items()}


def factor_exposures(prices: pd.DataFrame,
                     fetcher: DataFetcher,
                     as_of: pd.Timestamp) -> pd.DataFrame:
    """Size, momentum and volatility loadings, standardised.

    Args:
        prices: Constituent prices.
        fetcher: For share counts.
        as_of: Date the loadings are measured at.

    Returns:
        pd.DataFrame: z-scored exposures, names on the index.
    """
    returns = prices.pct_change()

    raw = pd.DataFrame({
        "size": _size(prices, fetcher, as_of),
        "momentum": _momentum(prices),
        "volatility": returns.std() * np.sqrt(PERIODS_PER_YEAR),
    })

    return z_scores(raw.dropna(how="all").fillna(0.0))


def _size(prices: pd.DataFrame,
          fetcher: DataFetcher,
          as_of: pd.Timestamp) -> pd.Series:
    """Log market capitalisation, falling back to log price.

    The log matters: market caps span orders of magnitude, and a raw one would
    make the largest name dominate a z-score by construction rather than by
    being unusual.
    """
    values = {}

    for name in prices.columns:
        shares = fetcher.fetch_shares_outstanding(str(name), as_of)
        last = float(prices[name].dropna().iloc[-1])
        values[name] = np.log(last * shares) if shares else np.log(last)

    return pd.Series(values)


def _momentum(prices: pd.DataFrame) -> pd.Series:
    """Trailing return, skipping the most recent month.

    The skip is the conventional construction: the last few weeks carry
    short-term reversal, which is a different effect pointing the other way,
    and folding it in muddies both.
    """
    if len(prices) <= MOMENTUM_SKIP + 1:
        return pd.Series(0.0, index=prices.columns)

    recent = prices.iloc[-(MOMENTUM_SKIP + 1)]
    lookback = min(MOMENTUM_LOOKBACK, len(prices) - 1)
    earlier = prices.iloc[-lookback]

    return (recent / earlier - 1.0).astype(float)


def build_optimisation_job(run_id: str,
                           request: OptimisationRunRequest,
                           constraint_set: ConstraintSet,
                           constraints: list[Any],
                           target_weights: dict[str, float],
                           label_for: dict[str, str],
                           fetcher: DataFetcher
                           ) -> Callable[[ProgressReporter],
                                         Awaitable[dict[str, Any]]]:
    """Build the coroutine that runs an optimisation.

    Returns:
        A coroutine function suitable for JobRegistry.submit.
    """
    async def run(report: ProgressReporter) -> dict[str, Any]:
        await report(0.1, "Loading constituent prices.")
        identifiers = sorted(target_weights)
        prices = constituent_prices(fetcher, identifiers, request.start, request.end)

        await report(0.35, "Estimating the risk model.")
        returns = prices.pct_change().dropna(how="all")
        risk_model = estimate_risk_model(returns, intensity=SHRINKAGE)

        await report(0.6, "Solving.")
        result = minimise_tracking_error(target_weights, constraints, risk_model)

        await report(0.85, "Assembling the result.")
        payload = assemble_optimisation(run_id, request, constraint_set,
                                        result, target_weights, label_for,
                                        prices, risk_model, fetcher)

        await report(1.0, "Complete.")

        return payload.model_dump()

    return run


def assemble_optimisation(run_id: str,
                          request: OptimisationRunRequest,
                          constraint_set: ConstraintSet,
                          result: Any,
                          target_weights: dict[str, float],
                          label_for: dict[str, str],
                          prices: pd.DataFrame,
                          risk_model: Any,
                          fetcher: DataFetcher) -> OptimisationRunResult:
    """Build the wire payload from a completed solve.

    Carries the prices' own summary rather than the frames themselves: the
    frontier and exposures panes re-derive what they need from the identifiers
    and the window, which keeps a stored run small.
    """
    weights = result.weights
    active = result.active_weights

    rows = [WeightRow(asset_id=str(name),
                      index_weight=float(target_weights.get(str(name), 0.0)),
                      optimal_weight=float(weights[name]),
                      active_weight=float(active[name]))
            for name in weights.index]
    rows.sort(key=lambda row: abs(row.active_weight), reverse=True)

    return OptimisationRunResult(
        run_id=run_id,
        index_id=request.index_id,
        constraint_set_id=constraint_set.id,
        start=str(prices.index[0].date()),
        end=str(prices.index[-1].date()),
        weights=rows,
        active_sum=float(active.sum()),
        tracking_error=result.tracking_error(),
        turnover=result.turnover(),
        holdings=result.holdings,
        binding=[_binding(label, label_for) for label in result.binding_labels()],
        heuristic=result.heuristic,
        converged=result.diagnostics.converged,
        iterations=result.diagnostics.iterations,
        objective=result.diagnostics.objective,
        solver_message=result.diagnostics.message)


def _binding(label: str,
             label_for: dict[str, str]) -> dict[str, str | None]:
    """A binding constraint, traced back to the row that produced it."""
    return {"label": label, "row_id": label_for.get(label)}


def build_frontier(run: dict[str, Any],
                   constraints: list[Any],
                   fetcher: DataFetcher,
                   risk_free_rate: float) -> FrontierView:
    """Trace the frontier over the run's universe and window."""
    identifiers = [row["asset_id"] for row in run["weights"]]
    prices = constituent_prices(fetcher, identifiers, run["start"], run["end"])

    returns = prices.pct_change().dropna(how="all")
    risk_model = estimate_risk_model(returns, intensity=SHRINKAGE)
    expected = expected_returns_from(prices)

    frontier = efficient_frontier(risk_model, expected, points=FRONTIER_POINTS,
                                  constraints=constraints,
                                  risk_free_rate=risk_free_rate)

    return FrontierView(
        run_id=run["run_id"],
        risk_free_rate=risk_free_rate,
        expected_returns=expected,
        points=[_frontier_point(point) for point in frontier.points],
        minimum_variance=_frontier_point(frontier.minimum_variance),
        tangency=_frontier_point(frontier.tangency),
        monotonic=frontier.is_monotonic())


def _frontier_point(point: Any) -> FrontierPoint:
    """One frontier point on the wire."""
    return FrontierPoint(expected_return=point.expected_return,
                         volatility=point.volatility,
                         sharpe_ratio=point.sharpe_ratio,
                         weights={str(name): float(value)
                                  for name, value in point.weights.items()},
                         binding=list(point.binding),
                         heuristic=point.heuristic)


def build_exposures(run: dict[str, Any],
                    fetcher: DataFetcher) -> ExposuresView:
    """Factor exposures of the active position, and its risk decomposition."""
    identifiers = [row["asset_id"] for row in run["weights"]]
    prices = constituent_prices(fetcher, identifiers, run["start"], run["end"])

    as_of = pd.Timestamp(run["end"])
    exposures = factor_exposures(prices, fetcher, as_of)

    returns = prices.pct_change().dropna(how="all")
    model = fit_factor_model(returns, exposures)

    optimal = {row["asset_id"]: row["optimal_weight"] for row in run["weights"]}
    index = {row["asset_id"]: row["index_weight"] for row in run["weights"]}

    decomposition = model.decompose_active_risk(optimal, index)

    return ExposuresView(
        run_id=run["run_id"],
        factors=list(model.factor_names),
        r_squared=model.r_squared,
        index_exposures=_exposure_rows(model.portfolio_exposures(index)),
        optimal_exposures=_exposure_rows(model.portfolio_exposures(optimal)),
        active_exposures=_exposure_rows(decomposition.exposures),
        risk=RiskDecomposition(
            total_variance=decomposition.total_variance,
            factor_variance=decomposition.factor_variance,
            specific_variance=decomposition.specific_variance,
            tracking_error=decomposition.tracking_error,
            factor_share=decomposition.factor_share,
            residual=decomposition.residual,
            reconciles=decomposition.reconciles(),
            contributions={str(name): float(value) for name, value
                           in decomposition.factor_contributions.items()}))


def _exposure_rows(exposures: pd.Series) -> list[FactorExposure]:
    """A factor exposure series on the wire."""
    return [FactorExposure(factor=str(name), exposure=float(value))
            for name, value in exposures.items()]


def target_weights_from(run: dict[str, Any],
                        as_of: str | None) -> dict[str, float]:
    """The index weights an optimisation is measured against.

    Reads the rebalance in force on *as_of* from a stored backtest, so the
    optimiser and the weights pane agree about what the index held.
    """
    snapshots = run.get("rebalances")
    if not snapshots:
        raise DataNotFoundError(
            "rebalance snapshots on this run",
            source="the run predates composition being stored; re-run the backtest")

    if as_of is None:
        return dict(snapshots[-1]["weights"])

    date = pd.Timestamp(as_of)
    eligible = [entry for entry in snapshots
                if pd.Timestamp(entry["date"]) <= date]

    if not eligible:
        raise DataNotFoundError(
            f"a rebalance on or before {date.date()}",
            source=f"the first rebalance is {snapshots[0]['date']}")

    return dict(eligible[-1]["weights"])
