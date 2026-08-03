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
from ..exceptions import InvalidRuleError

# Imported for its import side effect: the rule and scheme classes register
# themselves in the catalogue when their module loads, and nothing else here
# names them any more.
from ..index import methodology  # noqa: F401
from ..index.capping import minimum_feasible_cap
from ..index.constructor import IndexDefinition
from .schemas import Finding, IndexDocument, RuleSpec


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


def _validate_weighting(document: IndexDocument) -> list[Finding]:
    """Check the weighting group, including the unsupported cap slot."""
    weighting = document.pipeline.weighting
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

    findings.extend(_validate_cap(document))

    return findings


def _validate_cap(document: IndexDocument) -> list[Finding]:
    """Check the weight cap against its bounds and against the universe.

    An infeasible cap is caught here rather than at calculation time: a cap
    of 5% across 10 names can distribute at most 50%, and discovering that
    mid-run is far worse than being told while editing.
    """
    weighting = document.pipeline.weighting
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

    count = len(document.universe.identifiers)
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


def _validate_treatment(document: IndexDocument) -> list[Finding]:
    """Check the treatment group."""
    treatment = document.pipeline.treatment

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
    """Check the scalar fields IndexDefinition validates in its constructor."""
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

    if not document.universe.identifiers:
        findings.append(Finding(
            path="universe.identifiers",
            rule_id=None,
            severity="error",
            code="EMPTY_UNIVERSE",
            message="The universe must contain at least one identifier."))

    if not document.pipeline.selection:
        findings.append(Finding(
            path="pipeline.selection",
            rule_id=None,
            severity="warning",
            code="NO_SELECTION_RULES",
            message="No selection rules: every universe member will be a "
                    "constituent."))

    return findings


def validate_document(document: IndexDocument) -> list[Finding]:
    """Collect every finding for a definition document.

    Args:
        document: The definition to check.

    Returns:
        list[Finding]: Every problem found, each carrying the path and, where
        applicable, the id of the rule responsible. Empty when the definition
        is valid and unremarkable; warnings alone do not block saving.
    """
    findings = _validate_details(document)

    seen_ids: set[str] = set()
    for position, rule in enumerate(document.pipeline.selection):
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

    findings.extend(_validate_weighting(document))
    findings.extend(_validate_treatment(document))

    return findings


def has_errors(findings: list[Finding]) -> bool:
    """Whether any finding blocks saving."""
    return any(finding.severity == "error" for finding in findings)


def build_index_definition(document: IndexDocument) -> IndexDefinition:
    """Materialise a valid document into the library's IndexDefinition.

    Args:
        document: A document that has already passed validate_document()
            without errors.

    Returns:
        IndexDefinition: The library object, ready for IndexCalculator.

    Raises:
        ValueError: If the document is invalid after all — the library's own
            constructor validation is the final word.
    """
    rules = [
        catalogue.classes(catalogue.SELECTION)[rule.type](**rule.params)
        for rule in document.pipeline.selection
    ]
    weighting = document.pipeline.weighting
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
                           universe_identifiers=list(document.universe.identifiers),
                           max_constituent_weight=weighting.max_weight)
