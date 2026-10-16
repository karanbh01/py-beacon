# src/beacon/server/strategy_schemas.py
"""Request and response models for stored strategies."""
# Added with BN-312. Kept apart from `schemas.py`, which is far past the size
# this project keeps a module to.
from pydantic import BaseModel, ConfigDict, Field

from .schemas import (
    Finding,
    Identifier,
    ImplementationSpec,
    StrategySpec,
    TolerantCollection,
)

# A strategy whose index is not stored: deleted, or not copied to this machine.
INDEX_NOT_FOUND = "INDEX_NOT_FOUND"
# A strategy whose index is stored but cannot be read.
INDEX_UNREADABLE = "INDEX_UNREADABLE"


class StrategyDocument(BaseModel):
    """A stored strategy: what a backtest holds and how it trades.

    Everything else a backtest needs (the period, capital, currency,
    dividends, costs, flows, vehicle, share classes and modelling assumptions)
    belongs to the backtest, not the strategy.
    """
    model_config = ConfigDict(extra="forbid")
    id: Identifier = Field(description="The strategy's id.")
    name: str = Field(min_length=1, description="Display name.")
    description: str | None = Field(default=None, description="Optional description.")
    index_id: Identifier = Field(
        description="The stored index the strategy holds: the one tracked, or "
                    "the benchmark an active strategy is measured against.")
    strategy: StrategySpec = Field(
        default_factory=StrategySpec,
        description="What the backtest holds: the index in full (the default), "
                    "the index through a replication, or an active strategy.")
    implementation: ImplementationSpec | None = Field(
        default=None,
        description="How the strategy trades: screens, caps, minimum positions, "
                    "market impact, execution limits, the cash buffer and "
                    "equitising it. Null trades without them.")


class StrategyProblem(BaseModel):
    """Why a stored strategy cannot be run as it stands."""
    code: str = Field(
        description=f"'{INDEX_NOT_FOUND}' when its index is not stored, "
                    f"'{INDEX_UNREADABLE}' when it is stored but cannot be read.")
    message: str = Field(description="What is wrong and how to fix it.")


class StoredStrategy(StrategyDocument):
    """A stored strategy as listed and read, with any problem it has."""
    model_config = ConfigDict(extra="forbid")
    problem: StrategyProblem | None = Field(
        default=None,
        description="Why it cannot be run as it stands, such as an index that "
                    "has gone; null when it can.")


class StrategyCollection(TolerantCollection):
    """Response of `GET /strategies`."""
    strategies: list[StoredStrategy] = Field(
        description="Every strategy the server could read, each with any problem it has.")


class SavedStrategy(BaseModel):
    """Response of a strategy save: the document as stored, and any warnings."""
    strategy: StrategyDocument = Field(description="The strategy as stored.")
    findings: list[Finding] = Field(
        description="Warnings the save carried; a save with errors is refused instead.")


class StrategyDeletion(BaseModel):
    """Response of `DELETE /strategies/{strategy_id}`."""
    strategy_id: str = Field(description="The strategy that was removed.")
    backtest_record_deleted: bool = Field(
        description="Whether its saved backtest record went with it.")
    backtest_results_deleted: int = Field(
        description="How many job records and persisted run results of its "
                    "backtests were forgotten with it.")
