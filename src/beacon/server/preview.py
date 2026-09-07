# src/beacon/server/preview.py
"""
What an index definition resolves to at a date, in whichever face it has.

For a **rule pipeline**, that is the constituent derivation waterfall: how a
universe narrows to an index, one rung per selection rule, each naming what it
removed, then weighting and capping. The point is attributability — every
excluded asset reports the rule that excluded it, so a methodology author can
see *why* a name is missing rather than only *that* it is.

For a **derivation** (BN-170) there is no waterfall to show. An optimised index
eliminates nothing: it reallocates exactly the names its parent published, and
the solve moves every weight at once. Answering with an empty funnel would be a
worse lie than refusing, so the answer is a different shape — the parent's
weights beside the solved ones, and which constraints cost something. Both
faces come back through one endpoint and one response model, with `steps` and
`solve` mutually exclusive, exactly as `pipeline` and `derivation` are on the
document being previewed.

Neither face computes anything itself. The pipeline walk is
`beacon.index.calculation.selection`'s, and the solve is
`beacon.index.derived.solve_snapshot` — the same call `calculate_derived_index`
makes at every rebalance. A preview and the run it previews cannot disagree,
because they are the same code.

What remains here is presentation: mapping the core's rule *positions* onto the
stored document's rule *ids*, and pairing a parent snapshot with its solution.
Those ids belong to the document rather than to the rules — a rule object knows
its type, not which line of a saved definition it came from — so the mapping is
the server's job and stays the server's job.
"""
import pandas as pd

from ..asset.base import Asset
from ..data.fetcher import DataFetcher
from ..exceptions import InvalidRuleError
from ..index.calculation import IndexCalculator, SelectionResult
from ..index.calculation.selection import SelectionStep
from ..index.derived import (
    OptimisedIndexDefinition,
    calculate_source,
    solve_snapshot,
)
from ..index.result import IndexResult
from ..optimise.constraints import HOLDING_THRESHOLD
from ..optimise.result import OptimisationResult
from .definitions import build_definition, build_index_definition
from .schemas import (
    IndexDocument,
    PreviewAsset,
    PreviewConstraint,
    PreviewResponse,
    PreviewSolve,
    PreviewStep,
)
from .store import DocumentStore


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
                  documents: DocumentStore,
                  as_of: str | None = None) -> PreviewResponse:
    """Resolve an index definition at a date, in whichever face it has.

    Args:
        document: A validated index definition, rule-driven or derived.
        fetcher: Data source the rules, weighting scheme and solve read from.
        documents: Where a derivation's source is resolved from. Unused for a
            rule-driven document, and required all the same: the draft route
            previews a document that was never saved, so the store is the only
            way its parent can be found.
        as_of: Date to evaluate at, YYYY-MM-DD. Defaults to the base date.

    Returns:
        PreviewResponse: Per-asset outcomes and final weights, plus either the
        waterfall (`steps`) or the optimisation (`solve`).

    Raises:
        DataNotFoundError: If a derivation names a source that is not stored.
        InvalidRuleError: If a derivation's parent published no snapshot on or
            before *as_of*.
        CalculationError: If a derivation's solve is infeasible — the solver's
            own message names the binding conflict.
    """
    if document.derivation is not None:
        return _derived_preview(document, fetcher, documents, as_of)

    return _pipeline_preview(document, fetcher, as_of)


def _pipeline_preview(document: IndexDocument,
                      fetcher: DataFetcher,
                      as_of: str | None) -> PreviewResponse:
    """Derive the index from its universe, showing every step."""
    definition = build_index_definition(document)
    date = pd.Timestamp(as_of) if as_of else pd.Timestamp(definition.base_date)
    calculator = IndexCalculator(definition, fetcher)

    universe = calculator.resolve_universe(date)

    # `build_index_definition` above refuses an optimiser-derived document, so
    # by here the pipeline is present; the assertion says so to the reader and
    # to the type checker rather than repeating the refusal.
    assert document.pipeline is not None
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


def _derived_preview(document: IndexDocument,
                     fetcher: DataFetcher,
                     documents: DocumentStore,
                     as_of: str | None) -> PreviewResponse:
    """Solve the parent's snapshot at a date, and report the pair.

    The parent is calculated only as far as *as_of*, which is both cheaper and
    the honest window: nothing after that date can affect what the index looked
    like on it.
    """
    definition = build_definition(document, documents)

    # `document.derivation` is not None on this path, so `build_definition`
    # returned the derived object; the assertion says so to the type checker
    # rather than re-deciding what the caller already decided.
    assert isinstance(definition, OptimisedIndexDefinition)

    date = pd.Timestamp(as_of) if as_of else pd.Timestamp(definition.base_date)

    # Never short of the parent's own base date, which the calculator refuses
    # as a window. An as-of before the parent existed is still a real question,
    # and it deserves the answer `_snapshot_date` gives — "that index had
    # published nothing by then" — rather than a window error about a date the
    # client did not send.
    window = max(date, pd.Timestamp(definition.source.base_date))

    parent = calculate_source(definition.source, fetcher,
                              end_date=window.strftime("%Y-%m-%d"))

    rebalance = _snapshot_date(parent, date, definition)
    source_weights = parent.weight_snapshots[rebalance]

    result = solve_snapshot(definition, source_weights)
    solved = {str(asset): float(weight)
              for asset, weight in result.weights.items()}

    return PreviewResponse(
        index_id=document.id,
        as_of=date.strftime("%Y-%m-%d"),
        solve=_solve_block(definition, rebalance, result),
        assets=_derived_asset_rows(source_weights, solved),
        weights=solved,
        total_weight=sum(solved.values()))


def _snapshot_date(parent: IndexResult,
                   date: pd.Timestamp,
                   definition: OptimisedIndexDefinition) -> pd.Timestamp:
    """The parent snapshot a preview at *date* is about.

    The latest rebalance on or before the as-of date, which is the composition
    actually in force that day — a derivation solves only where its parent
    published weights, so an as-of between two rebalances previews the earlier
    one rather than inventing a solve at a date the index never rebalanced on.

    Raises:
        InvalidRuleError: If the parent published nothing by then, which means
            the as-of date precedes the parent's first rebalance. The date came
            from the request, so this is the client's to correct.
    """
    applicable = [snapshot for snapshot in parent.weight_snapshots
                  if snapshot <= date]

    if not applicable:
        raise InvalidRuleError(
            f"index '{definition.index_id}'",
            f"its source '{definition.source.index_id}' had published no "
            f"weights by {date.date()}, so there is nothing to optimise at "
            f"that date. Preview it on or after the source's first rebalance")

    return max(applicable)


def _solve_block(definition: OptimisedIndexDefinition,
                 rebalance: pd.Timestamp,
                 result: OptimisationResult) -> PreviewSolve:
    """The optimisation as the wire shape.

    Every constraint is reported, with `binding` flagged, rather than only the
    binding ones: a binding constraint's slack is zero by construction, so the
    informative number is always the one on a constraint that did *not* bind.
    """
    return PreviewSolve(
        source_index_id=definition.source.index_id,
        rebalance_date=rebalance.strftime("%Y-%m-%d"),
        objective=definition.objective,
        binding=result.binding_labels(),
        constraints=[PreviewConstraint(label=slack.label,
                                       kind=slack.kind,
                                       slack=slack.slack,
                                       unit=slack.unit,
                                       binding=slack.is_binding)
                     for slack in result.slacks])


def _derived_asset_rows(source_weights: dict[str, float],
                        solved: dict[str, float]) -> list[PreviewAsset]:
    """One row per name, before and after the solve.

    The two sides are unioned rather than taken from either alone. They are the
    same universe today — the optimiser allocates over exactly the names the
    target names — but a constraint that pinned a name to zero should still
    appear as a row that went to nothing, and a row that exists on one side
    only should read as a real difference rather than vanish.
    """
    rows = []

    for identifier in sorted(set(source_weights) | set(solved)):
        before = source_weights.get(identifier, 0.0)
        after = solved.get(identifier, 0.0)

        rows.append(PreviewAsset(
            identifier=identifier,
            # A solve gives every name a weight, most of them meaningfully; the
            # library's own holding threshold is what decides that a residual
            # is not a position.
            included=after > HOLDING_THRESHOLD,
            weight=after,
            source_weight=before,
            solved_weight=after,
            weight_delta=after - before))

    return rows
