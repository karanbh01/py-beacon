# src/beacon/server/holdings.py
"""
A strategy's holdings workflow at a date, one row per name, from the universe
to the shares held.

What a run decided at each rebalance is saved with it, inside the strategy's
record under a key of its own: how each rebalance's universe narrowed (the
index's selection), each name's weight after the implementation's screens,
and its target and final weights, and the book's units held, recorded only on
the days they change, with its daily NAV. The view joins that to the run's own
index and strategy weights: the index weights before and after the index's
cap, the strategy's choice, each implementation stage, the weight traded to,
and, as of the date asked about, the weight held and its drift.

A held weight is the units held times the day's price in the book's currency,
over the day's NAV: what the book's own record says, without storing every
day's weights.
"""
# BN-326. Karan's call on 2026-10-19: Holdings Detail shows the whole workflow
# from the engine rather than stitched together in the app.
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pandas as pd

from .. import catalogue
from ..backtest.implementation import STALE, TOO_SMALL
from ..backtest.result import BacktestResult
from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError
from ..index.calculation.selection import STALENESS_POSITION, UNIVERSE_POSITION
from ..index.result import IndexResult
from . import stage_descriptions
from .holding_reference import HoldingReference
from .holdings_schemas import (
    CASH_BUFFER,
    IMPLEMENTATION_CAP,
    IMPLEMENTATION_SCREEN,
    INDEX_CAP,
    INDEX_RULE,
    STRATEGY,
    UNIVERSE,
    HoldingRow,
    HoldingStage,
    HoldingsView,
)
from .runs import snapshot_at, snapshots_from
from .schemas import IndexDocument, SeriesPayload, StageParam
from .strategy_runs import strategy_steps
from .types import specs_for
from .weights import prices_for

# Where a strategy's record keeps what its run decided at each rebalance.
HOLDINGS_KEY = "holdings"
# How far back a price is looked for when the date asked about has none.
PRICE_LOOKBACK_DAYS = 10


def holdings_document(result: BacktestResult,
                      index_result: IndexResult) -> dict[str, Any]:
    """What a run decided at each rebalance, for its holdings workflow."""
    quantities = _quantities(result.portfolio.positions)
    selections = getattr(index_result, "selections", {}) or {}

    return {
        "selections": {str(pd.Timestamp(date).date()): {
            "universe": summary.universe,
            "steps": [asdict(step) for step in summary.steps],
            "exclusions": summary.exclusions} for date, summary in selections.items()},
        "implementation": [{"date": str(step.date.date()), "target": step.target,
                            "screened": step.screened, "weights": step.weights,
                            "removed": step.removed, "capped": step.capped,
                            "cash_weight": step.cash_weight}
                           for step in result.rebalance_steps],
        "units": {name: [[str(day.date()), float(units)] for day, units in changes.items()]
                  for name, changes in quantities.items()},
        "nav": SeriesPayload.from_series(result.portfolio.nav).model_dump(),
    }


def build_holdings(strategy_id: str,
                   index: IndexDocument,
                   run: dict[str, Any],
                   document: dict[str, Any] | None,
                   fetcher: DataFetcher,
                   date: str | None,
                   reference: HoldingReference,
                   benchmark_on: Callable[[str], dict[str, float] | None],
                   universe_on: Callable[[list[str], str], dict[str, float | None]] | None = None,
                   universe_name: str | None = None) -> HoldingsView:
    """The workflow in force on *date*, the run's last day when None.

    *benchmark_on* gives the run's benchmark index's weights on a rebalance
    date, or None for a run without one; *universe_on* the universe's weights
    of some names on a date.

    Raises:
        DataNotFoundError: If *date* is before the run's first rebalance.
    """
    steps = strategy_steps(run)
    dates = [step["date"] for step in steps]
    as_of = date or str(run["level"]["index"][-1])[:10]
    in_force = [step for step in steps if step["date"] <= as_of]

    if not in_force:
        raise DataNotFoundError(f"a rebalance on or before {as_of}",
                                source=f"the run's first rebalance is {dates[0]}")

    step = in_force[-1]
    snapshot = snapshot_at(snapshots_from(run), step["date"])
    saved = document or {}
    selection = (saved.get("selections") or {}).get(snapshot.date)
    plan = next((entry for entry in saved.get("implementation") or []
                 if entry["date"] == step["date"]), None)
    kind = ((run.get("settings") or {}).get("strategy") or {}).get("type")
    currency = (run.get("settings") or {}).get("currency") or index.currency
    stages, excluded = _stages(index, run, selection, len(snapshot.weights), step, plan,
                               kind, currency, universe_name)
    buffer = float(((run.get("settings") or {}).get("implementation") or {})
                   .get("cash_buffer") or 0.0)
    held = _held(saved, fetcher, as_of, currency)
    target = plan["weights"] if plan is not None else step["weights"]
    benchmark = benchmark_on(step["date"])

    names = list(dict.fromkeys([*(selection["universe"] if selection else snapshot.weights),
                                *step["weights"], *target, *held]))
    universe = (selection["universe"] if selection else list(snapshot.weights))
    shares = universe_on(universe, snapshot.date) if universe_on is not None else {}
    rows = [HoldingRow(
        identifier=name,
        universe_weight=shares.get(name),
        reference=reference.of(name, as_of),
        excluded_at=excluded.get(name),
        index_weight_uncapped=snapshot.uncapped_weights.get(name),
        index_weight=snapshot.weights.get(name),
        benchmark_weight=benchmark.get(name) if benchmark is not None else None,
        strategy_weight=step["weights"].get(name),
        implementation=_implementation_weights(name, plan, stages, buffer),
        portfolio_weight=target.get(name),
        weight_drift=(held[name][1] - target.get(name, 0.0)
                      if name in held and as_of != step["date"] else None),
        current_weight=held[name][1] if name in held else None,
        shares_held=held[name][0] if name in held else None) for name in names]

    return HoldingsView(strategy_id=strategy_id, index_id=index.id, as_of=as_of,
                        rebalance_date=step["date"], rebalance_dates=dates, stages=stages,
                        rows=rows, cash_weight=plan["cash_weight"] if plan else None)


def _stages(index: IndexDocument,
            run: dict[str, Any],
            selection: dict[str, Any] | None,
            constituents: int,
            step: dict[str, Any],
            plan: dict[str, Any] | None,
            kind: str | None,
            currency: str,
            universe_name: str | None = None) -> tuple[list[HoldingStage], dict[str, str]]:
    """The workflow's steps in order, and the step each excluded name left at."""
    stages: list[HoldingStage] = []
    excluded: dict[str, str] = {}

    if selection is not None:
        stages, excluded = _index_stages(index, selection, universe_name)

    if index.pipeline is not None and index.pipeline.weighting.max_weight is not None:
        # A cap moves weight and keeps every name.
        remaining = stages[-1].remaining if stages else constituents
        stages.append(_stage(INDEX_CAP, "Index cap", INDEX_CAP, remaining,
                             stage_descriptions.index_cap(index.pipeline.weighting.max_weight)))

    if kind not in (None, "index"):
        settings = (run.get("settings") or {}).get("strategy") or {}
        stages.append(_stage(STRATEGY, _strategy_label(settings), STRATEGY,
                             len(step["weights"]),
                             stage_descriptions.strategy(settings, currency)))

    if plan is not None:
        implementation, removed = _implementation_stages(run, plan, currency)
        stages += implementation
        excluded.update(removed)

    return stages, excluded


def _stage(key: str,
           label: str,
           kind: str,
           remaining: int,
           about: stage_descriptions.Described) -> HoldingStage:
    """One stage, with what it is and does (BN-328)."""
    return HoldingStage(key=key, label=label, kind=kind, remaining=remaining, type=about.type,
                        description=about.description,
                        params=[StageParam(**param) for param in about.params])


def _index_stages(index: IndexDocument,
                  selection: dict[str, Any],
                  universe_name: str | None) -> tuple[list[HoldingStage], dict[str, str]]:
    """The universe and the index's rules, as the selection narrowed them."""
    rules = index.pipeline.selection if index.pipeline is not None else []
    labels = {spec.name: spec.label for spec in specs_for(catalogue.SELECTION)}
    keys: dict[int, str] = {}
    stages = []

    for step in selection["steps"]:
        position = step["position"]

        if position == UNIVERSE_POSITION:
            key, label, kind = UNIVERSE, "Universe", UNIVERSE
            about = stage_descriptions.universe(
                index.universe.universe_id if index.universe else None, universe_name,
                step["remaining"])
        elif position == STALENESS_POSITION:
            key, label, kind = "stale", "Stale price", INDEX_RULE
            about = stage_descriptions.stale_price()
        else:
            rule = rules[position - 1] if 0 < position <= len(rules) else None
            key = rule.id if rule is not None else f"rule {position}"
            label = labels.get(step["rule_name"], step["rule_name"])
            kind = INDEX_RULE
            about = stage_descriptions.rule(step["rule_name"],
                                            rule.params if rule is not None else None,
                                            index.currency)

        keys[position] = key
        stages.append(_stage(key, label, kind, step["remaining"], about))

    excluded = {name: keys.get(int(position), str(position))
                for name, position in selection["exclusions"].items()}

    return stages, excluded


def _implementation_stages(run: dict[str, Any],
                           plan: dict[str, Any],
                           currency: str) -> tuple[list[HoldingStage], dict[str, str]]:
    """The implementation's screens, its cash buffer and its caps."""
    settings = (run.get("settings") or {}).get("implementation") or {}
    labels = {spec.name: spec.label for spec in specs_for(catalogue.SCREEN)}
    screens = [{"type": STALE}, *(settings.get("screens") or [])]
    stages = []
    left = set(plan["target"])

    for screen in screens:
        kind = screen["type"]
        left -= {name for name, by in plan["removed"].items() if by == kind}
        about = (stage_descriptions.stale_price() if kind == STALE
                 else stage_descriptions.screen(kind, screen.get("params"), currency))
        stages.append(_stage(kind, "Stale price" if kind == STALE else labels.get(kind, kind),
                             IMPLEMENTATION_SCREEN, len(left), about))

    if settings.get("cash_buffer"):
        stages.append(_stage(CASH_BUFFER, "Cash buffer", CASH_BUFFER, len(left),
                             stage_descriptions.cash_buffer(settings["cash_buffer"])))

    # Only where something could cap: the implementation's caps or minimum
    # position, or a vehicle's limit that did.
    if (settings.get("caps") or settings.get("minimum_position") or plan["capped"]
            or TOO_SMALL in plan["removed"].values()):
        stages.append(_stage(IMPLEMENTATION_CAP, "Capacity caps", IMPLEMENTATION_CAP,
                             len(plan["weights"]),
                             stage_descriptions.capacity(settings, currency)))
    removed = {name: IMPLEMENTATION_CAP if by == TOO_SMALL else by
               for name, by in plan["removed"].items()}

    return stages, removed


def _implementation_weights(name: str,
                            plan: dict[str, Any] | None,
                            stages: list[HoldingStage],
                            buffer: float) -> dict[str, float | None]:
    """A name's weight after each implementation stage; *buffer* is the cash
    buffer's share, which scales every weight after the screens."""
    if plan is None:
        return {}

    out = plan["removed"].get(name)
    weights: dict[str, float | None] = {}
    gone = False

    for stage in stages:
        if stage.kind == IMPLEMENTATION_SCREEN:
            gone = gone or out == stage.key
            weights[stage.key] = None if gone else plan["screened"].get(name)
        elif stage.kind == CASH_BUFFER:
            weights[stage.key] = (None if gone or name not in plan["screened"]
                                  else plan["screened"][name] * (1.0 - buffer))
        elif stage.kind == IMPLEMENTATION_CAP:
            weights[stage.key] = plan["weights"].get(name)

    return weights


def _held(document: dict[str, Any],
          fetcher: DataFetcher,
          as_of: str,
          currency: str) -> dict[str, tuple[float, float]]:
    """Each name the book held on *as_of*: its units, and its weight, the
    units at the day's price over the day's NAV."""
    units = {name: _units_on(changes, as_of) for name, changes in
             (document.get("units") or {}).items()}
    units = {name: count for name, count in units.items() if count}
    nav = _value_on(document.get("nav"), as_of)

    if not units or not nav:
        return {}

    start = str((pd.Timestamp(as_of) - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)).date())
    prices = prices_for(fetcher, sorted(units), start, as_of, currency)
    last = prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)

    return {name: (count, float(count * last[name] / nav))
            for name, count in units.items() if name in last and pd.notna(last[name])}


def _units_on(changes: list[list[Any]],
              day: str) -> float:
    """The units held on *day*, from the days they changed."""
    held = 0.0

    for changed, units in changes:
        if changed > day:
            break

        held = float(units)

    return held


def _value_on(series: dict[str, Any] | None,
              day: str) -> float | None:
    """A stored series' value on *day*, or the last before it."""
    if not series:
        return None

    pairs = [(str(label)[:10], value) for label, value in zip(series["index"], series["data"],
                                                               strict=True)]
    before = [value for label, value in pairs if label <= day and value is not None]

    return float(before[-1]) if before else None


def _quantities(positions: pd.DataFrame) -> dict[str, pd.Series]:
    """Each name's units on the days they changed, 0 from the day it left."""
    if positions.empty:
        return {}

    panel = positions.pivot_table(index="DATE", columns="ASSET_ID", values="QUANTITY",
                                  aggfunc="last").fillna(0.0).sort_index()
    # The days a name's units differ from the day before's, and its first
    # if it held any.
    changed = panel.ne(panel.shift())
    changed.iloc[0] = panel.iloc[0].ne(0.0)

    return {str(name): panel[name][changed[name]] for name in panel.columns}


def _strategy_label(settings: dict[str, Any]) -> str:
    """What the strategy stage's column reads."""
    if settings.get("type") == "index_tracking":
        replication = (settings.get("replication") or {}).get("type", "")
        labels = {spec.name: spec.label for spec in specs_for(catalogue.REPLICATION)}

        return f"{labels.get(replication, replication)} replication".strip()

    construction = (settings.get("construction") or {}).get("type", "")
    labels = {spec.name: spec.label for spec in specs_for(catalogue.CONSTRUCTION)}

    return labels.get(construction, "Active strategy")
