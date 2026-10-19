# src/beacon/server/holdings_schemas.py
"""Response models for a strategy's holdings workflow at a date."""
# BN-326. Kept apart from `schemas.py`, which is far past the size this project
# keeps a module to.
from typing import Any

from pydantic import BaseModel, Field

# The kinds of step a holding passes on its way from the universe to the book.
UNIVERSE = "universe"
INDEX_RULE = "index_rule"
INDEX_CAP = "index_cap"
STRATEGY = "strategy"
IMPLEMENTATION_SCREEN = "implementation_screen"
CASH_BUFFER = "cash_buffer"
IMPLEMENTATION_CAP = "implementation_cap"
STAGE_KINDS = (UNIVERSE, INDEX_RULE, INDEX_CAP, STRATEGY, IMPLEMENTATION_SCREEN, CASH_BUFFER,
               IMPLEMENTATION_CAP)


class HoldingStage(BaseModel):
    """One step of the workflow, in order."""
    key: str = Field(description="What a row's `excluded_at` and `implementation` name it "
                                 "by: a rule's id, a screen's type, or the kind for the "
                                 "single steps.")
    label: str = Field(description="What a column header reads, such as 'Minimum price'.")
    kind: str = Field(description=f"One of {', '.join(STAGE_KINDS)}.")
    remaining: int = Field(description="How many names are left after it.")


class HoldingRow(BaseModel):
    """One name's way through the workflow at the rebalance in force."""
    identifier: str = Field(description="The name.")
    reference: dict[str, Any] | None = Field(
        default=None,
        description="The reference columns asked for with `reference_fields`, as of "
                    "`as_of`, keyed in lower case; null when the name had no reference "
                    "row then.")
    excluded_at: str | None = Field(
        default=None, description="The key of the stage it left at; null for a name "
                                  "still in the book.")
    index_weight_uncapped: float | None = Field(
        default=None, description="Its index weight before the index's cap.")
    index_weight: float | None = Field(
        default=None, description="Its index weight after the index's cap.")
    benchmark_weight: float | None = Field(
        default=None, description="Its weight in the backtest's benchmark index; null "
                                  "without one, or for a name it does not hold.")
    strategy_weight: float | None = Field(
        default=None, description="What the strategy chose before the implementation: "
                                  "the index weight for a strategy that holds its index, "
                                  "the replication's or the construction's otherwise.")
    implementation: dict[str, float | None] = Field(
        default_factory=dict,
        description="By implementation stage key, the weight it carried after that "
                    "stage; null after the stage that removed it.")
    portfolio_weight: float | None = Field(
        default=None, description="The weight the rebalance traded to.")
    weight_drift: float | None = Field(
        default=None, description="Its held weight less its portfolio weight as of "
                                  "`as_of`; null on the rebalance date itself.")
    current_weight: float | None = Field(
        default=None, description="Its weight in the book as of `as_of`.")
    shares_held: float | None = Field(
        default=None, description="The units of it the book held as of `as_of`.")


class HoldingsView(BaseModel):
    """Response of `GET /strategies/{strategy_id}/holdings`."""
    strategy_id: str = Field(description="The strategy whose latest run this reads.")
    index_id: str = Field(description="The index it holds.")
    as_of: str = Field(description="The date asked about, YYYY-MM-DD.")
    rebalance_date: str = Field(description="The rebalance in force on `as_of`.")
    rebalance_dates: list[str] = Field(description="Every rebalance of the run, in order.")
    stages: list[HoldingStage] = Field(description="The workflow's steps, in order.")
    rows: list[HoldingRow] = Field(
        description="One per name in the universe at the rebalance, those that left "
                    "included, then any name held that the universe did not name.")
    cash_weight: float | None = Field(
        default=None, description="The share of the book the rebalance left in cash.")
