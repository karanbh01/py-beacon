# src/beacon/server/block_names.py
"""
A stored analytics block in the names it is sent with now: one kept before
0.11.0 named a table's portfolio column ``book``.

Kept apart from `beacon.server.analysis` so the document store, which the
jobs and so the analysis import, can migrate a stored job's block too.
"""
# BN-463 (Karan, 2026-11-26): the dimension was "book" before 0.11.0.
from typing import Any

TABLE = "table"


def portfolio_named(block: dict[str, Any]) -> dict[str, Any]:
    """*block* with each table's ``book`` column named ``portfolio``."""
    for value in block.get("statistics", {}).values():
        point = value.get("point") if value.get("kind") == TABLE else None

        if isinstance(point, dict) and "book" in point.get("columns", []):
            point["columns"] = ["portfolio" if column == "book" else column
                                for column in point["columns"]]

    return block


__all__ = ["portfolio_named"]
