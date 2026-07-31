# src/beacon/server/risk.py
"""
Estimating a risk model, and serving what it says about itself.

Estimation is a job because it means pulling a price history for every name in
the universe before any matrix arithmetic happens. The read is cheap and serves
the stored result.

## The diagnostics are the interesting part

A correlation matrix looks equally plausible whether or not it can be trusted,
so the endpoint reports how it was made and how well conditioned it is rather
than only the numbers:

* **intensity** — how much weight went on the structured target. Zero means the
  raw sample covariance, which on a short history across many names is mostly
  noise.
* **condition number** — largest eigenvalue over smallest. An optimiser inverts
  this matrix, and a large condition number means the inverse amplifies
  estimation error rather than reflecting it.
* **positive semi-definite** — computed from the eigenvalues, not asserted.
  A matrix that fails this can produce a negative portfolio variance, and a
  caller about to invert it needs to know rather than be reassured.

`average_correlation` is reported alongside because it is the sanity check a
person can actually do: a diversified equity universe sits somewhere around
0.3–0.6, and a figure far outside that says the window or the universe is not
what someone thought.
"""
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError
from ..risk.model import estimate_risk_model
from .jobs import ProgressReporter
from .schemas import (
    RiskDiagnosticsPayload,
    RiskModelRequest,
    RiskModelView,
    TableFrame,
)

logger = logging.getLogger(__name__)

# Matrices are rounded to this many decimals on the wire. A correlation is
# meaningful to perhaps three; carrying seventeen makes a payload large and
# implies a precision the estimate does not have.
WIRE_DECIMALS = 8


def constituent_returns(fetcher: DataFetcher,
                        identifiers: list[str],
                        start: str | None,
                        end: str | None) -> pd.DataFrame:
    """Daily returns for a set of names, names on the columns.

    Raises:
        DataNotFoundError: If fewer than two names can be priced. A covariance
            over one asset is a variance, and the endpoint promises a matrix.
    """
    series: dict[str, pd.Series] = {}

    for identifier in identifiers:
        frame = fetcher.fetch_market_data(identifier, start, end)
        if not frame.empty and "CLOSE" in frame.columns:
            series[identifier] = frame["CLOSE"]

    if len(series) < 2:
        raise DataNotFoundError(
            f"prices for at least two of {identifiers}",
            source=f"only {len(series)} could be priced")

    prices = pd.DataFrame(series).sort_index()

    return prices.pct_change().dropna(how="all")


def build_estimation_job(model_id: str,
                         request: RiskModelRequest,
                         identifiers: list[str],
                         fetcher: DataFetcher
                         ) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build the coroutine that estimates a risk model.

    Returns:
        A coroutine function suitable for JobRegistry.submit.
    """
    async def run(report: ProgressReporter) -> dict[str, Any]:
        await report(0.1, f"Loading prices for {len(identifiers)} identifier(s).")
        returns = constituent_returns(fetcher, identifiers,
                                      request.start, request.end)

        await report(0.6, "Estimating the covariance.")
        model = estimate_risk_model(returns,
                                    target=request.target,
                                    intensity=request.intensity,
                                    repair=request.repair)

        await report(0.9, "Assembling the result.")
        payload = assemble_risk_model(model_id, request, model)

        await report(1.0, "Complete.")

        return payload.model_dump()

    return run


def assemble_risk_model(model_id: str,
                        request: RiskModelRequest,
                        model: Any) -> RiskModelView:
    """Build the wire payload from an estimated model."""
    diagnostics = model.diagnostics

    return RiskModelView(
        model_id=model_id,
        asset_ids=model.asset_ids,
        start=request.start,
        end=request.end,
        correlation=_matrix(model.correlation),
        covariance=_matrix(model.covariance),
        volatilities={str(name): float(value)
                      for name, value in model.volatilities().items()},
        diagnostics=RiskDiagnosticsPayload(
            observations=diagnostics.observations,
            assets=diagnostics.assets,
            target=diagnostics.target,
            intensity=diagnostics.intensity,
            average_correlation=diagnostics.average_correlation,
            condition_number=diagnostics.condition_number,
            smallest_eigenvalue=diagnostics.smallest_eigenvalue,
            positive_semi_definite=diagnostics.positive_semi_definite,
            repaired=diagnostics.repaired))


def _matrix(frame: pd.DataFrame) -> TableFrame:
    """A square matrix on the wire, rounded to a sensible precision."""
    return TableFrame(**_payload(frame.round(WIRE_DECIMALS)))


def _payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Row-oriented frame payload, with the index as plain strings."""
    return {"index": [str(label) for label in frame.index],
            "columns": [str(label) for label in frame.columns],
            "data": [[None if pd.isna(value) else float(value) for value in row]
                     for row in frame.to_numpy()]}
