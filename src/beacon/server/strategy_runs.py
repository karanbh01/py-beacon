# src/beacon/server/strategy_runs.py
"""
A stored strategy's runs, and the weights a target stands for.

Runs are kept per strategy: a backtest of a stored strategy is the job kind
`backtest:{strategy_id}`, and its saved record is filed under the strategy's
id. A run is served only when it carries its strategy's id, so a run saved
before runs were kept per strategy, filed under an index's id, is never taken
for a strategy of the same name.

A target for the optimiser or a risk model is an index or a strategy. An
index's weights are calculated, so no run is needed; a strategy's are what its
latest run chose at each rebalance: the index's for a plain index strategy, its
replication's for a tracker, its construction's for an active strategy.
"""
# BN-313.
from typing import Any

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError
from .backtest_settings import ACTIVE, INDEX_TRACKING
from .index_level import calculate_index
from .jobs import JobRegistry
from .runs import snapshot_at
from .schemas import IndexDocument, rebalance_snapshots
from .store import DocumentStore


def backtest_kind(strategy_id: str) -> str:
    """The job kind a strategy's backtests run under."""
    return f"backtest:{strategy_id}"


def latest_run(registry: JobRegistry,
               strategy_id: str) -> dict[str, Any] | None:
    """The most recent successful run of a strategy, or None.

    A run of this kind that does not carry the strategy's id was saved before
    runs were kept per strategy, under an index of the same name, and is not
    this strategy's. Runs since are always newer, so only the newest needs the
    check.
    """
    run = registry.latest_result(backtest_kind(strategy_id))

    if run is None or (run.get("settings") or {}).get("strategy_id") != strategy_id:
        return None

    return run


def require_run(registry: JobRegistry,
                strategy_id: str) -> dict[str, Any]:
    """A strategy's latest run, or not-found pointing at how to make one."""
    run = latest_run(registry, strategy_id)

    if run is None:
        raise DataNotFoundError(f"a completed backtest of strategy '{strategy_id}'",
                                source=f"run POST /strategies/{strategy_id}/backtest first")

    return run


def strategy_steps(run: dict[str, Any]) -> list[dict[str, Any]]:
    """What a run's strategy chose at each rebalance, as `{date, weights}`
    rows in date order."""
    kind = ((run.get("settings") or {}).get("strategy") or {}).get("type")
    steps = (run.get("active") if kind == ACTIVE
             else run.get("replication") if kind == INDEX_TRACKING
             else run.get("rebalances"))

    if not steps:
        raise DataNotFoundError("the weights this run's strategy chose",
                                source="the run carries no rebalances")

    return [{"date": step["date"], "weights": dict(step["weights"])} for step in steps]


def strategy_weights(run: dict[str, Any],
                     as_of: str | None) -> tuple[str, dict[str, float]]:
    """The rebalance date and the weights a run's strategy chose in force on
    *as_of*; the last when None."""
    steps = strategy_steps(run)

    if as_of is None:
        return steps[-1]["date"], steps[-1]["weights"]

    eligible = [step for step in steps if pd.Timestamp(step["date"]) <= pd.Timestamp(as_of)]

    if not eligible:
        raise DataNotFoundError(f"a rebalance on or before {as_of}",
                                source=f"the run's first rebalance is {steps[0]['date']}")

    return eligible[-1]["date"], eligible[-1]["weights"]


def index_weights(document: IndexDocument,
                  index_store: DocumentStore,
                  fetcher: DataFetcher,
                  as_of: str | None) -> tuple[str, dict[str, float]]:
    """The rebalance date and an index's weights in force on *as_of*, from
    its calculation up to that date; the latest when None."""
    result = calculate_index(document, index_store, fetcher, end=as_of)
    snapshot = snapshot_at(rebalance_snapshots(result), as_of)

    return snapshot.date, dict(snapshot.weights)
