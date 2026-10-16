# src/beacon/server/strategies.py
"""
Stored strategies: what makes one valid, what stops one running, and what
uses an index.

A strategy is checked by the same code that builds a backtest's settings, so a
strategy that saves is one a backtest accepts, and the two cannot disagree
about what is valid.
"""
# BN-312.
from typing import Any

from .backtest_settings import build as build_settings
from .documents import read_collection, stored, validated
from .schemas import BacktestRequest, Finding, IndexDocument
from .store import SCHEMA_VERSION_KEY, DocumentStore
from .strategy_schemas import (
    INDEX_NOT_FOUND,
    INDEX_UNREADABLE,
    StoredStrategy,
    StrategyDocument,
    StrategyProblem,
)


def read_strategy(document_id: str,
                  document: dict[str, Any]) -> StrategyDocument:
    """A stored strategy as its model: the build function for reading one.

    The model refuses fields it does not know, so a client sending a
    backtest's vehicle or period on a strategy hears about it, and the
    store's own version stamp is set aside before it is checked.
    """
    fields = {key: value for key, value in document.items() if key != SCHEMA_VERSION_KEY}

    return StrategyDocument.model_validate(fields)


def problem_of(document: StrategyDocument,
               indices: DocumentStore) -> StrategyProblem | None:
    """Why a strategy cannot be run as it stands, or None when it can.

    Its index can go after it is saved: deleted through a store the refusal
    never saw, or not copied to this machine with the strategy.
    """
    index = stored(indices, document.index_id, validated(IndexDocument))

    if not index.present:
        return StrategyProblem(
            code=INDEX_NOT_FOUND,
            message=f"Its index '{document.index_id}' is not stored. Point the "
                    f"strategy at another index, or restore that one.")

    if not index.readable:
        return StrategyProblem(
            code=INDEX_UNREADABLE,
            message=f"Its index '{document.index_id}' is stored but cannot be "
                    f"read. Repair or replace that index, or point the "
                    f"strategy at another.")

    return None


def with_problem(document: StrategyDocument,
                 indices: DocumentStore) -> StoredStrategy:
    """A strategy as listed and read, with any problem it has."""
    return StoredStrategy(**document.model_dump(),
                          problem=problem_of(document, indices))


def validate_strategy(document: StrategyDocument,
                      indices: DocumentStore) -> list[Finding]:
    """Every problem with a strategy, each at its field's path.

    The strategy and implementation are checked as a backtest checks them,
    so their findings sit at the same paths (`strategy.replication`,
    `implementation.caps.0`). Its index must be stored and readable.
    """
    _, findings = build_settings(BacktestRequest(strategy=document.strategy,
                                                 implementation=document.implementation))
    problem = problem_of(document, indices)

    if problem is not None:
        findings = [Finding(path="index_id", severity="error",
                            code=problem.code.lower(), message=problem.message),
                    *findings]

    return findings


def strategies_using(strategies: DocumentStore,
                     index_ids: set[str]) -> list[str]:
    """The ids of the stored strategies that hold any of these indices.

    A strategy that cannot be read is not counted: what it holds cannot be
    known, and refusing a delete over it would leave the index undeletable
    until the strategy was repaired.
    """
    documents, _ = read_collection(strategies, read_strategy, "strategy")

    return sorted(document.id for document in documents if document.index_id in index_ids)
