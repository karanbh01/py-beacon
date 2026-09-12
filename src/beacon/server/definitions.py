# src/beacon/server/definitions.py
"""
Index definition documents: validation and materialisation.

A stored definition is a JSON document describing a rule pipeline. It is not an
`IndexDefinition` — the library object takes constructed rule and scheme
instances, which JSON cannot carry. This module owns both directions: checking
a document and turning a valid one into the library object.

Validation collects *findings* rather than raising at the first problem. A user
editing a pipeline needs every issue at once, each addressable to the rule that
caused it, not a single exception naming whichever one failed first.
"""
from typing import Any

from .. import catalogue
from ..exceptions import DataNotFoundError, InvalidRuleError

# Imported for its import side effect: the rule and scheme classes register
# themselves in the catalogue when their module loads, and nothing else here
# names them any more.
from ..index import methodology  # noqa: F401
from ..index.calculation.total_return import NET_TOTAL_RETURN
from ..index.capping import minimum_feasible_cap
from ..index.constructor import IndexDefinition
from ..index.derived import (
    OBJECTIVES,
    AnyIndexDefinition,
    OptimisedIndexDefinition,
)
from ..index.schedule import DAY_RULES, is_known_calendar
from .constraints import build_constraint_rows, validate_constraint_rows
from .documents import stored, validated
from .schemas import (
    DerivationPayload,
    Finding,
    IndexDocument,
    PipelineSpec,
    RuleSpec,
    UniverseRef,
)
from .store import DocumentStore


class PipelineValidationError(InvalidRuleError):
    """An invalid pipeline, carrying every finding.

    Subclasses InvalidRuleError so the existing exception mapping gives it 422
    and the INVALID_RULE code without a new registration. The findings ride
    along as an instance attribute, which is what the error envelope reads to
    build its structured `detail` — so the client receives every problem at
    once rather than a single message for the whole form.
    """
    def __init__(self,
                 rule_description: str,
                 reason: str,
                 findings: list[Finding]):
        super().__init__(rule_description, reason)
        self.findings = [finding.model_dump() for finding in findings]

# Selection rules and weighting schemes come from the catalogue the classes
# register themselves in (BN-117), not from a list here. There used to be four
# tables — which rules exist, what each accepts, and two more mapping names to
# constructors — and every one of them had to be updated by hand when a rule
# was added. Deriving them removes the possibility of the four disagreeing.
#
# Functions rather than constants so the answer stays live: a constant would
# snapshot the registry at import and silently omit anything registered after.
def selection_rules() -> dict[str, set[str]]:
    """Selection rule name -> the parameters it accepts."""
    return {name: catalogue.parameter_names(catalogue.SELECTION, name)
            for name in catalogue.registered_names(catalogue.SELECTION)}


def weighting_schemes() -> dict[str, set[str]]:
    """Weighting scheme name -> the parameters it accepts."""
    return {name: catalogue.parameter_names(catalogue.WEIGHTING, name)
            for name in catalogue.registered_names(catalogue.WEIGHTING)}

# IndexDefinition.get_rebalance_dates() supports exactly these.
REBALANCE_FREQUENCIES = ("MONTHLY", "QUARTERLY", "SEMI-ANNUAL", "ANNUAL")

# The calculator adjusts the divisor for a corporate action it is handed;
# SPECIAL_DIVIDEND is implemented and the rest are logged stubs. There is no
# switch to turn that off, so this is the only accepted value.
TREATMENT_CORPORATE_ACTIONS = ("ADJUST_DIVISOR",)



def _unknown_params(spec_params: dict[str, Any],
                    allowed: set[str]) -> list[str]:
    """Return the parameter names that are not accepted."""
    return sorted(set(spec_params) - allowed)


def _validate_selection_rule(rule: RuleSpec,
                             position: int) -> list[Finding]:
    """Check one selection rule against the library's rule set."""
    path = f"pipeline.selection[{position}]"

    known = selection_rules()

    if rule.type not in known:
        return [Finding(
            path=path,
            rule_id=rule.id,
            severity="error",
            code="UNKNOWN_RULE_TYPE",
            message=f"'{rule.type}' is not a known selection rule. "
                    f"Available: {', '.join(sorted(known))}.")]

    findings = [
        Finding(path=f"{path}.params.{name}",
                rule_id=rule.id,
                severity="error",
                code="UNKNOWN_PARAMETER",
                message=f"'{rule.type}' does not accept a '{name}' parameter.")
        for name in _unknown_params(rule.params, known[rule.type])
    ]

    findings.extend(_validate_rule_semantics(rule, path))

    return findings


def _validate_rule_semantics(rule: RuleSpec,
                             path: str) -> list[Finding]:
    """Check parameter values that the rule constructor would reject."""
    findings: list[Finding] = []
    minimum = rule.params.get("min_market_cap")
    maximum = rule.params.get("max_market_cap")

    if minimum is not None and maximum is not None and minimum > maximum:
        findings.append(Finding(
            path=f"{path}.params.min_market_cap",
            rule_id=rule.id,
            severity="error",
            code="INVALID_RANGE",
            message="min_market_cap cannot be greater than max_market_cap."))

    lookback = rule.params.get("lookback_days")
    if lookback is not None and lookback <= 0:
        findings.append(Finding(
            path=f"{path}.params.lookback_days",
            rule_id=rule.id,
            severity="error",
            code="INVALID_VALUE",
            message="lookback_days must be positive."))

    return findings


def _validate_schedule(document: IndexDocument) -> list[Finding]:
    """Check the scheduling metadata for combinations that cannot be honoured."""
    findings: list[Finding] = []

    if document.rebalance_day_rule not in DAY_RULES:
        findings.append(Finding(
            path="rebalance_day_rule",
            rule_id=None,
            severity="error",
            code="UNKNOWN_DAY_RULE",
            message=f"'{document.rebalance_day_rule}' is not a known day rule. "
                    f"Available: {', '.join(sorted(DAY_RULES))}."))

    if document.calendar is not None and not is_known_calendar(document.calendar):
        # An error rather than a warning: falling back to business days would
        # make the index compute differently from the one that was defined,
        # with nothing on screen to say so.
        findings.append(Finding(
            path="calendar",
            rule_id=None,
            severity="error",
            code="UNKNOWN_CALENDAR",
            message=f"'{document.calendar}' is not a trading calendar this "
                    f"server can use. Install the `calendars` extra, or use a "
                    f"MIC such as XNYS or XLON."))

    if (document.return_type == NET_TOTAL_RETURN
            and document.withholding_tax_rate == 0.0):
        # A net index withholding nothing is a gross index under another name.
        # Warned rather than blocked: zero is a legitimate rate for a domestic
        # index, and the surprise is worth flagging without stopping a save.
        findings.append(Finding(
            path="withholding_tax_rate",
            rule_id=None,
            severity="warning",
            code="NET_RETURN_WITHOUT_WITHHOLDING",
            message="A net-total-return index with a zero withholding rate "
                    "produces the same levels as a gross one."))

    if (document.withholding_tax_rate > 0.0
            and document.return_type != NET_TOTAL_RETURN):
        findings.append(Finding(
            path="withholding_tax_rate",
            rule_id=None,
            severity="warning",
            code="WITHHOLDING_NOT_APPLIED",
            message=f"A withholding rate is set but `return_type` is "
                    f"'{document.return_type}', so nothing is withheld. Only "
                    f"NET_TOTAL_RETURN applies it."))

    return findings


def _validate_weighting(pipeline: PipelineSpec,
                        universe: UniverseRef) -> list[Finding]:
    """Check the weighting group, including the unsupported cap slot."""
    weighting = pipeline.weighting
    findings: list[Finding] = []

    known = weighting_schemes()

    if weighting.scheme not in known:
        findings.append(Finding(
            path="pipeline.weighting.scheme",
            rule_id=weighting.id,
            severity="error",
            code="UNKNOWN_SCHEME",
            message=f"'{weighting.scheme}' is not a known weighting scheme. "
                    f"Available: {', '.join(sorted(known))}."))
    else:
        findings.extend(
            Finding(path=f"pipeline.weighting.params.{name}",
                    rule_id=weighting.id,
                    severity="error",
                    code="UNKNOWN_PARAMETER",
                    message=f"'{weighting.scheme}' does not accept a '{name}' parameter.")
            for name in _unknown_params(weighting.params,
                                        known[weighting.scheme]))

    findings.extend(_validate_cap(pipeline, universe))

    return findings


def _validate_cap(pipeline: PipelineSpec,
                  universe: UniverseRef) -> list[Finding]:
    """Check the weight cap against its bounds and against the universe.

    An infeasible cap is caught here rather than at calculation time: a cap
    of 5% across 10 names can distribute at most 50%, and discovering that
    mid-run is far worse than being told while editing.
    """
    weighting = pipeline.weighting
    cap = weighting.max_weight

    if cap is None:
        return []

    if not 0.0 < cap <= 1.0:
        return [Finding(
            path="pipeline.weighting.max_weight",
            rule_id=weighting.id,
            severity="error",
            code="INVALID_CAP",
            message=f"max_weight must be a fraction in (0, 1]; got {cap}.")]

    count = len(universe.identifiers)
    if not count:
        return []

    reachable = cap * count

    if reachable < 1.0:
        return [Finding(
            path="pipeline.weighting.max_weight",
            rule_id=weighting.id,
            severity="error",
            code="INFEASIBLE_CAP",
            message=f"A cap of {cap:.4%} cannot be satisfied by {count} "
                    f"universe members: the total would reach at most "
                    f"{reachable:.4%}. The smallest feasible cap is "
                    f"{minimum_feasible_cap(count):.4%}.")]

    # Feasible only because every single member is included. Selection rules
    # can only shrink that set, so one exclusion makes the cap impossible —
    # worth warning about while editing rather than failing mid-run.
    if reachable < 1.0 + cap:
        return [Finding(
            path="pipeline.weighting.max_weight",
            rule_id=weighting.id,
            severity="warning",
            code="TIGHT_CAP",
            message=f"A cap of {cap:.4%} needs all {count} universe members to "
                    "be selected. If any rule excludes one, the cap becomes "
                    "infeasible.")]

    return []


def _validate_treatment(pipeline: PipelineSpec) -> list[Finding]:
    """Check the treatment group."""
    treatment = pipeline.treatment

    if treatment.corporate_actions not in TREATMENT_CORPORATE_ACTIONS:
        return [Finding(
            path="pipeline.treatment.corporate_actions",
            rule_id=None,
            severity="error",
            code="UNSUPPORTED_TREATMENT",
            message=f"'{treatment.corporate_actions}' is not supported. "
                    f"Available: {', '.join(TREATMENT_CORPORATE_ACTIONS)}.")]

    return []


def _validate_details(document: IndexDocument) -> list[Finding]:
    """Check the scalar fields IndexDefinition validates in its constructor.

    Identity only: these hold for either face of a document, so an optimised
    index is checked against them exactly as a rule-driven one is.
    """
    findings: list[Finding] = []

    if document.base_value <= 0:
        findings.append(Finding(path="base_value",
                                rule_id=None,
                                severity="error",
                                code="INVALID_VALUE",
                                message="base_value must be positive."))

    if document.rebalancing_frequency not in REBALANCE_FREQUENCIES:
        findings.append(Finding(
            path="rebalancing_frequency",
            rule_id=None,
            severity="error",
            code="UNSUPPORTED_FREQUENCY",
            message=f"'{document.rebalancing_frequency}' is not supported. "
                    f"Available: {', '.join(REBALANCE_FREQUENCIES)}."))

    return findings


def _validate_pipeline_face(pipeline: PipelineSpec,
                            universe: UniverseRef) -> list[Finding]:
    """Check the rule-driven face: universe, selection, weighting, treatment."""
    findings: list[Finding] = []

    if not universe.identifiers:
        findings.append(Finding(
            path="universe.identifiers",
            rule_id=None,
            severity="error",
            code="EMPTY_UNIVERSE",
            message="The universe must contain at least one identifier."))

    if not pipeline.selection:
        findings.append(Finding(
            path="pipeline.selection",
            rule_id=None,
            severity="warning",
            code="NO_SELECTION_RULES",
            message="No selection rules: every universe member will be a "
                    "constituent."))

    seen_ids: set[str] = set()
    for position, rule in enumerate(pipeline.selection):
        findings.extend(_validate_selection_rule(rule, position))

        if rule.id in seen_ids:
            findings.append(Finding(
                path=f"pipeline.selection[{position}].id",
                rule_id=rule.id,
                severity="error",
                code="DUPLICATE_RULE_ID",
                message=f"Rule id '{rule.id}' is used more than once; findings "
                        "would not be addressable."))
        seen_ids.add(rule.id)

    findings.extend(_validate_weighting(pipeline, universe))
    findings.extend(_validate_treatment(pipeline))

    return findings


def _validate_derivation_face(derivation: DerivationPayload) -> list[Finding]:
    """Check the optimiser-derived face: objective and constraints.

    The objective is checked against the library's own tuple rather than a
    list here, and the message names the accepted set, because the whole point
    of the field being a plain string is that a client learns what it may send
    from the server rather than from a copy of it.

    The constraint rows go through exactly the checker
    `/optimise/constraint-sets` uses, re-addressed under `derivation`, so a
    row that is refused in one place is refused identically in the other.
    """
    findings: list[Finding] = []

    if derivation.objective not in OBJECTIVES:
        findings.append(Finding(
            path="derivation.objective",
            rule_id=None,
            severity="error",
            code="UNKNOWN_OBJECTIVE",
            message=f"'{derivation.objective}' is not an objective this "
                    f"server can solve. Accepted: {', '.join(OBJECTIVES)}."))

    findings.extend(validate_constraint_rows(derivation.constraints,
                                             prefix="derivation.constraints"))

    return findings


def validate_document(document: IndexDocument) -> list[Finding]:
    """Collect every finding for a definition document.

    Args:
        document: The definition to check — either face. Which checks run is
            decided by `derivation`, the same discriminator a client branches
            on.

    Returns:
        list[Finding]: Every problem found, each carrying the path and, where
        applicable, the id of the rule responsible. Empty when the definition
        is valid and unremarkable; warnings alone do not block saving.
    """
    findings = _validate_details(document)

    if document.derivation is not None:
        findings.extend(_validate_derivation_face(document.derivation))
    elif document.pipeline is not None and document.universe is not None:
        findings.extend(_validate_pipeline_face(document.pipeline,
                                                document.universe))

    findings.extend(_validate_schedule(document))

    return findings


def has_errors(findings: list[Finding]) -> bool:
    """Whether any finding blocks saving."""
    return any(finding.severity == "error" for finding in findings)


def build_index_definition(document: IndexDocument) -> IndexDefinition:
    """Materialise a valid rule-driven document into an IndexDefinition.

    Args:
        document: A document that has already passed validate_document()
            without errors, carrying a rule pipeline.

    Returns:
        IndexDefinition: The library object, ready for IndexCalculator.

    Raises:
        InvalidRuleError: If the document is optimiser-derived. An
            `OptimisedIndexDefinition` is not an `IndexDefinition` — the
            calculator must never receive one by accident — so callers that
            can handle either use :func:`build_definition`.
        ValueError: If the document is invalid after all — the library's own
            constructor validation is the final word.
    """
    pipeline = document.pipeline
    universe = document.universe

    if pipeline is None or universe is None:
        raise InvalidRuleError(
            f"index '{document.id}'",
            "it is optimiser-derived, so it has no rule pipeline to "
            "materialise. Its methodology is its derivation, which only the "
            "backtest path calculates")

    rules = [
        catalogue.classes(catalogue.SELECTION)[rule.type](**rule.params)
        for rule in pipeline.selection
    ]
    weighting = pipeline.weighting
    scheme = catalogue.classes(catalogue.WEIGHTING)[weighting.scheme](
        **weighting.params)

    return IndexDefinition(index_id=document.id,
                           index_name=document.name,
                           base_date=document.base_date,
                           base_value=document.base_value,
                           currency=document.currency,
                           eligibility_rules=rules,
                           weighting_scheme=scheme,
                           rebalancing_frequency=document.rebalancing_frequency,
                           description=document.description,
                           universe_identifiers=list(universe.identifiers),
                           max_constituent_weight=weighting.max_weight,
                           rebalance_day_rule=document.rebalance_day_rule,
                           calendar=document.calendar,
                           return_type=document.return_type,
                           withholding_tax_rate=document.withholding_tax_rate,
                           effective_lag_sessions=document.effective_lag_sessions)


def build_definition(document: IndexDocument,
                     documents: DocumentStore) -> AnyIndexDefinition:
    """Materialise a document of either face into a library definition.

    A rule-driven document builds an :class:`IndexDefinition`; an
    optimiser-derived one builds an
    :class:`~beacon.index.derived.OptimisedIndexDefinition` over its source,
    resolved through the store — recursively, so a chain of derivations builds
    a chain of definitions and the whole thing calculates through one path.

    Args:
        document: The stored definition to materialise.
        documents: Where the sources of a derivation are read from.

    Returns:
        AnyIndexDefinition: The library object, ready for
        :class:`~beacon.backtest.main.Backtest`.

    Raises:
        DataNotFoundError: If a derivation names a source that is not stored.
        InvalidRuleError: If a derivation chain returns to itself, or runs
            deeper than :data:`MAX_DERIVATION_DEPTH`.
    """
    return _built_definition(document, documents, ())


# How many derivations deep a chain may go. Chained optimisation is allowed by
# design ("depth is naturally limited by sanity"), so this is not a modelling
# limit — it is the backstop for a chain whose links a visited set cannot see,
# and a number no honest chain reaches.
MAX_DERIVATION_DEPTH = 16


def _built_definition(document: IndexDocument,
                      documents: DocumentStore,
                      chain: tuple[str, ...]) -> AnyIndexDefinition:
    """One link of a derivation chain, refusing a chain that eats itself.

    Two guards, because they fail differently. The *visited set* — the ids
    already on the way down, carried in `chain` — catches a cycle at the exact
    link that closes it, so the message can print the loop; without it the
    recursion would run until the interpreter's stack gave out, which reaches
    the client as a 500 that says nothing. The *depth cap* is the backstop for
    a chain that is not a cycle but is absurd, and keeps the recursion bounded
    no matter what a stored document says.
    """
    derivation = document.derivation

    if derivation is None:
        return build_index_definition(document)

    if document.id in chain:
        loop = " -> ".join((*chain, document.id))
        raise InvalidRuleError(
            f"index '{document.id}'",
            f"its derivation chain returns to itself ({loop}), so there is no "
            f"source index to optimise. Re-point one of the derivations at a "
            f"rule-driven index")

    if len(chain) >= MAX_DERIVATION_DEPTH:
        raise InvalidRuleError(
            f"index '{document.id}'",
            f"its derivation chain is more than {MAX_DERIVATION_DEPTH} "
            f"indices deep ({' -> '.join(chain)}), which is deeper than this "
            f"server will resolve")

    # Existence is not enough: a source that is stored and unreadable is a
    # source this document can never be calculated from, and reading it answers
    # not-found (BN-174), so saying so here keeps one story. It went through a
    # strict read and a bare `model_validate` before, which made a corrupt
    # parent a 500 on every save of a child (BN-177).
    held = stored(documents, derivation.source_index_id, validated(IndexDocument))

    if held.document is None:
        raise DataNotFoundError(
            f"source index '{derivation.source_index_id}', which "
            f"'{document.id}' is derived from",
            source="DocumentStore")

    source = _built_definition(held.document,
                               documents,
                               (*chain, document.id))

    return OptimisedIndexDefinition(
        index_id=document.id,
        index_name=document.name,
        source=source,
        objective=derivation.objective,
        constraints=build_constraint_rows(derivation.constraints),
        base_date=document.base_date,
        base_value=document.base_value,
        currency=document.currency,
        description=document.description)
