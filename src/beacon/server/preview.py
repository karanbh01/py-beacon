# src/beacon/server/preview.py
"""
Constituent derivation waterfall.

Shows how a universe narrows to an index: one rung per selection rule, each
naming what it removed, then weighting and capping. The point is
attributability — every excluded asset reports the rule that excluded it, so a
methodology author can see *why* a name is missing rather than only *that* it
is.

`IndexCalculator.select_constituents` answers only "which assets survive", so
the waterfall re-walks the rules here to record provenance. It calls the same
`rule.is_eligible` in the same order, and a test asserts the survivors match
`select_constituents` exactly — that guard is what keeps preview from drifting
away from what a real run would produce.
"""
import pandas as pd

from ..asset.base import Asset
from ..data.fetcher import DataFetcher
from ..index.calculation import IndexCalculator
from ..index.methodology import EligibilityRuleBase
from .definitions import build_index_definition
from .schemas import IndexDocument, PreviewAsset, PreviewResponse, PreviewStep


def _evaluate(rule: EligibilityRuleBase,
              asset: Asset,
              date: pd.Timestamp,
              fetcher: DataFetcher) -> bool:
    """Apply one rule to one asset, treating a failure as exclusion.

    `IndexCalculator._passes_all_rules` swallows rule exceptions and excludes
    the asset. Preview mirrors that rather than surfacing the error, so the
    waterfall matches what a real run would do.
    """
    try:
        return rule.is_eligible(asset, date, fetcher)
    except Exception:
        return False


def _walk_rules(universe: list[Asset],
                rules: list[EligibilityRuleBase],
                rule_ids: list[str],
                date: pd.Timestamp,
                fetcher: DataFetcher) -> tuple[list[Asset], list[PreviewStep],
                                               dict[str, tuple[str, int]]]:
    """Run the rules in order, recording what each one removed.

    Returns:
        tuple: the surviving assets, one step per rung (starting with the
        universe), and a mapping of excluded identifier -> (rule_id, position).
    """
    surviving = list(universe)
    steps = [PreviewStep(position=0, remaining=len(surviving))]
    exclusions: dict[str, tuple[str, int]] = {}

    for position, (rule, rule_id) in enumerate(zip(rules, rule_ids, strict=True), start=1):
        kept: list[Asset] = []
        removed: list[str] = []

        for asset in surviving:
            if _evaluate(rule, asset, date, fetcher):
                kept.append(asset)
            else:
                removed.append(asset.asset_id)
                # First rule to exclude an asset owns it: later rules never
                # see it, which is what makes the funnel attributable.
                exclusions[asset.asset_id] = (rule_id, position)

        surviving = kept
        steps.append(PreviewStep(position=position,
                                 rule_id=rule_id,
                                 rule_type=rule.rule_name,
                                 remaining=len(surviving),
                                 excluded=sorted(removed)))

    return surviving, steps, exclusions


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

    surviving, steps, exclusions = _walk_rules(
        universe, definition.eligibility_rules, rule_ids, date, fetcher)

    raw_weights = calculator.calculate_constituent_weights(surviving, date)
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
