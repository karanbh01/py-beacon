# src/beacon/analysis/analytics/profile/trading.py
"""A name's profile's trades and events: what the strategy did with it, and
what happened to it."""
# BN-473 (Karan, 2026-11-25). The rebalances where it came close to failing a
# rule join the events once BN-474 keeps the margins.
import pandas as pd

from ...panels import ACTIONS, REBALANCE_WEIGHTS, TRADES
from ...reports.context import FUND
from ...reports.statistics import RUN_PREDATES, Unavailable
from .selection import near_misses, selection

IDENTIFIER = "IDENTIFIER"
# An event: its date, kind, detail and amount.
Event = tuple[pd.Timestamp, str, str | None, float | None]
TRADE_COLUMNS = {"SIDE": "side", "QUANTITY": "quantity", "PRICE": "price", "VALUE": "notional",
                 "COST": "cost", "IMPACT": "impact", "REASON": "reason", "TAGS": "tags"}

ADDED, REMOVED = "added_to_{}", "removed_from_{}"
CAPPED, SCREENED_OUT, CORPORATE_ACTION = "capped", "screened_out", "corporate_action"
NEAR_MISS = "near_miss"
# The portfolios whose holding of a name is an event, and the order a day's
# events are listed in.
HOLDERS = (FUND, "index")
ORDER = (*(ADDED.format(book) for book in HOLDERS), CAPPED, NEAR_MISS, SCREENED_OUT,
         *(REMOVED.format(book) for book in HOLDERS), CORPORATE_ACTION)
EVENT_LABELS = {ADDED.format(FUND): "Added to the fund",
                REMOVED.format(FUND): "Sold out of the fund",
                ADDED.format("index"): "Added to the index",
                REMOVED.format("index"): "Removed from the index", CAPPED: "Capped",
                NEAR_MISS: "Came close to failing a rule",
                SCREENED_OUT: "Screened out", CORPORATE_ACTION: "Corporate action"}


def trades(name: str,
           tables: dict[str, pd.DataFrame],
           days: pd.DatetimeIndex) -> pd.DataFrame:
    """Every trade in *name* over *days*, in the order made: its side,
    shares, price and notional in the portfolio's currency, cost and market
    impact, and why it was made and its tags.

    Raises:
        Unavailable: If the run was kept before its trades were.
    """
    kept = tables.get(TRADES)

    if kept is None:
        raise Unavailable(RUN_PREDATES)

    dates = pd.DatetimeIndex(kept.index)
    mine = kept.loc[(kept[IDENTIFIER].astype(str) == name) & (dates >= days[0])
                    & (dates <= days[-1])]
    found = mine[[column for column in TRADE_COLUMNS if column in mine.columns]]
    found = found.rename(columns=TRADE_COLUMNS)

    for column in ("side", "reason", "tags"):
        if column in found:
            found[column] = found[column].astype(object)

    return found.rename_axis("DATE")


def events(name: str,
           tables: dict[str, pd.DataFrame],
           days: pd.DatetimeIndex,
           portfolios: dict[str, pd.DataFrame],
           near_miss: float) -> pd.DataFrame:
    """What happened to *name* over *days*, oldest first: each day the fund
    or the index began or stopped holding it, each rebalance that capped it,
    that it passed a rule with a margin under *near_miss*, or that screened
    it out, and each corporate action on it; a row an event, with its
    ``event``, a ``detail`` (the screen, the rule and what it measured, the
    action's type) and an ``amount`` (the weight capped to, the margin, the
    action's amount)."""
    rows: list[Event] = []

    for book in HOLDERS:
        if book in portfolios:
            rows += _holding(name, book, portfolios[book])

    rows += _rebalances(name, tables.get(REBALANCE_WEIGHTS))
    rows += _near_misses(name, tables, days, near_miss)
    rows += _actions(name, tables.get(ACTIONS))
    found = pd.DataFrame(rows, columns=["DATE", "event", "detail", "amount"])
    found = found.loc[(found["DATE"] >= days[0]) & (found["DATE"] <= days[-1])]
    found = found.assign(order=found["event"].map(ORDER.index)).sort_values(["DATE", "order"],
                                                                           kind="stable")
    found = found.drop(columns="order").set_index(pd.DatetimeIndex(found["DATE"], name="DATE"))

    found = found.drop(columns="DATE").astype({"event": object, "detail": object,
                                               "amount": float})
    found["detail"] = found["detail"].where(found["detail"].notna(), None)

    return found


def _holding(name: str,
             book: str,
             weights: pd.DataFrame) -> list[Event]:
    """Each day *portfolio* began or stopped holding *name*."""
    if name not in weights.columns:
        return []

    held = weights[name].fillna(0.0).sort_index() > 0.0
    before = held.shift(1, fill_value=False)
    began, stopped = held & ~before, ~held & before

    return ([(day, ADDED.format(book), None, None) for day in held.index[began]]
            + [(day, REMOVED.format(book), None, None) for day in held.index[stopped]])


def _rebalances(name: str,
                kept: pd.DataFrame | None) -> list[Event]:
    """Each rebalance that capped *name* or screened it out."""
    if kept is None:
        return []

    mine = kept.loc[kept[IDENTIFIER].astype(str) == name]
    rows = [(day, CAPPED, None, float(capped))
            for day, capped in mine["CAPPED"].dropna().items()]

    return rows + [(day, SCREENED_OUT, str(screen), None)
                   for day, screen in mine["REMOVED"].dropna().items()]


def _near_misses(name: str,
                 tables: dict[str, pd.DataFrame],
                 days: pd.DatetimeIndex,
                 within: float) -> list[Event]:
    """Each rebalance *name* passed a rule with a margin under *within*;
    none for a run kept before its margins were."""
    try:
        close = near_misses(selection(name, tables, days), within)
    except Unavailable:
        return []

    return [(pd.Timestamp(day), NEAR_MISS, f"{rule}: {quantity}", float(found))
            for day, rule, quantity, found in zip(close.index, close["rule_name"],
                                                  close["measure"], close["margin"],
                                                  strict=True)]


def _actions(name: str,
             kept: pd.DataFrame | None) -> list[Event]:
    """Each corporate action on *name*."""
    if kept is None:
        return []

    mine = kept.loc[kept[IDENTIFIER].astype(str) == name]

    return [(pd.Timestamp(day), CORPORATE_ACTION, str(kind), float(amount))
            for day, kind, amount in zip(mine["EX_DATE"], mine["TYPE"], mine["VALUE"],
                                         strict=True)]
