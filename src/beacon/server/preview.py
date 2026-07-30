# src/beacon/server/preview.py
"""
Constituent derivation waterfall.

Shows how a universe narrows to an index: one rung per selection rule, each
naming what it removed, then weighting and capping. The point is
attributability — every excluded asset reports the rule that excluded it, so a
methodology author can see *why* a name is missing rather than only *that* it
is.

The narrowing itself is not done here. It was, once: this module re-walked the
rules to record provenance because `select_constituents` returned only
survivors, and a test asserted the two agreed. Since BN-102 there is one walk,
in `beacon.index.calculation.selection`, and the calculator is the one throwing
information away. A preview and the run it previews can no longer disagree,
because they are the same code.

What remains here is presentation: mapping the core's rule *positions* onto the
stored document's rule *ids*. Those ids belong to the document rather than to
the rules — a rule object knows its type, not which line of a saved definition
it came from — so the mapping is the server's job and stays the server's job.
"""
import pandas as pd

from ..asset.base import Asset
from ..data.fetcher import DataFetcher
from ..index.calculation import IndexCalculator, SelectionResult
from ..index.calculation.selection import SelectionStep
from .definitions import build_index_definition
from .schemas import IndexDocument, PreviewAsset, PreviewResponse, PreviewStep


def _as_preview_step(step: SelectionStep,
                     rule_ids: list[str]) -> PreviewStep:
    """Render one core rung as the wire shape, attaching the document's rule id.

    The universe rung has no rule and therefore no id; every other position
    indexes the document's selection list one-for-one, because the definition
    the calculator was built from was derived from that list in order.
    """
    if step.is_universe:
        return PreviewStep(position=step.position, remaining=step.remaining)

    return PreviewStep(position=step.position,
                       rule_id=rule_ids[step.position - 1],
                       rule_type=step.rule_name,
                       remaining=step.remaining,
                       excluded=list(step.excluded))


def _exclusions_by_rule_id(selection: SelectionResult,
                           rule_ids: list[str]) -> dict[str, tuple[str, int]]:
    """Translate positional provenance into ``id -> (rule_id, position)``."""
    return {asset_id: (rule_ids[position - 1], position)
            for asset_id, position in selection.exclusions.items()}


def build_preview(document: IndexDocument,
                  fetcher: DataFetcher,
                  as_of: str | None = None) -> PreviewResponse:
    """Derive the index from its universe, showing every step.

    Args:
        document: A validated index definition.
        fetcher: Data source the rules and weighting scheme read from.
        as_of: Date to evaluate at, YYYY-MM-DD. Defaults to the base date.

    Returns:
        PreviewResponse: The waterfall, per-asset outcomes, and final weights.
    """
    definition = build_index_definition(document)
    date = pd.Timestamp(as_of) if as_of else pd.Timestamp(definition.base_date)
    calculator = IndexCalculator(definition, fetcher)

    universe = calculator.resolve_universe(date)
    rule_ids = [rule.id for rule in document.pipeline.selection]

    selection = calculator.select_with_provenance(universe, date)
    steps = [_as_preview_step(step, rule_ids) for step in selection.steps]
    exclusions = _exclusions_by_rule_id(selection, rule_ids)

    raw_weights = calculator.calculate_constituent_weights(selection.survivors, date)
    weights, cap_report = calculator.cap_weights(raw_weights)

    by_id = {asset.asset_id: weight for asset, weight in weights.items()}
    uncapped_by_id = {asset.asset_id: weight for asset, weight in raw_weights.items()}

    assets = _asset_rows(universe, by_id, uncapped_by_id, cap_report.capped, exclusions)

    return PreviewResponse(index_id=document.id,
                           as_of=date.strftime("%Y-%m-%d"),
                           steps=steps,
                           assets=assets,
                           weights=by_id,
                           total_weight=sum(by_id.values()),
                           cap=cap_report.cap,
                           cap_redistributed=cap_report.redistributed)


def _asset_rows(universe: list[Asset],
                weights: dict[str, float],
                uncapped: dict[str, float],
                capped: dict[str, float],
                exclusions: dict[str, tuple[str, int]]) -> list[PreviewAsset]:
    """Build one row per universe member, included or not."""
    rows = []

    for asset in universe:
        identifier = asset.asset_id
        excluded_by, excluded_at = exclusions.get(identifier, (None, None))
        included = identifier in weights

        rows.append(PreviewAsset(
            identifier=identifier,
            included=included,
            excluded_by=excluded_by,
            excluded_at=excluded_at,
            weight=weights.get(identifier),
            uncapped_weight=uncapped.get(identifier) if identifier in capped else None,
            capped=identifier in capped))

    return rows
