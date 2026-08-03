# src/beacon/server/constraints.py
"""
Constraint sets: the stored form of what a portfolio is allowed to be.

One stored row maps to exactly one class in `beacon.optimise.constraints`. That
correspondence is the whole design — a client's constraint editor, the JSON it
saves, and the objects the solver receives are the same list in three
representations, so there is no translation layer where a rule can quietly
change meaning.

## Validation happens before the job, not inside it

A malformed constraint set is a bad request, and the client should learn that
from the save or the submission rather than from a job that fails a moment
later. So the document is validated up front and reports **every** problem it
finds, addressed to the row that caused it — a user fixing a constraint editor
needs all the errors, not the first one.

What cannot be checked here is feasibility: whether a set of individually valid
constraints can be satisfied together depends on the universe and the data, and
the optimiser answers that when it runs. It refuses rather than fudging, so the
failure still reaches the client with a message naming what is impossible.
"""
import logging

from .. import catalogue
from ..optimise.constraints import (
    Cardinality,
    Constraint,
    ExpectedReturnTarget,
    FullInvestment,
    GroupBounds,
    PositionBounds,
    TurnoverBudget,
)
from .schemas import ConstraintRow, ConstraintSet, Finding

logger = logging.getLogger(__name__)

# Row type to the class it becomes. The keys are what a client sends and what
# is stored; adding a constraint class means adding it here and nowhere else.
CONSTRAINT_TYPES = {
    "FullInvestment": FullInvestment,
    "PositionBounds": PositionBounds,
    "GroupBounds": GroupBounds,
    "TurnoverBudget": TurnoverBudget,
    "Cardinality": Cardinality,
    "ExpectedReturnTarget": ExpectedReturnTarget,
}

# Parameters each type accepts, so an unknown key is reported against the row
# that carries it rather than surfacing as a TypeError from a constructor.
def constraint_params() -> dict[str, set[str]]:
    """Constraint type -> the parameters it accepts.

    Read off the classes (BN-117) rather than kept here by hand. The table this
    replaced had to be edited whenever a constraint gained a parameter, and
    nothing failed when it was not — the validator simply rejected a parameter
    the solver would have accepted.
    """
    return {name: catalogue.parameter_names(catalogue.CONSTRAINT, name)
            for name in catalogue.registered_names(catalogue.CONSTRAINT)}

ERROR = "error"
WARNING = "warning"


def validate_constraint_set(document: ConstraintSet) -> list[Finding]:
    """Check a constraint set, reporting everything wrong with it.

    Args:
        document: The set to check.

    Returns:
        list: Findings, addressed to the row that caused each. Empty when the
        set is well formed — which is not the same as feasible.
    """
    findings: list[Finding] = []

    if not document.constraints:
        findings.append(Finding(
            path="constraints",
            severity=WARNING,
            code="NO_CONSTRAINTS",
            message="This set constrains nothing. A solve against it returns "
                    "the target unchanged."))

    for position, row in enumerate(document.constraints):
        findings.extend(_validate_row(position, row))

    findings.extend(_validate_set_shape(document))

    return findings


def _validate_row(position: int,
                  row: ConstraintRow) -> list[Finding]:
    """Check one row's type and parameters."""
    path = f"constraints[{position}]"

    if row.type not in CONSTRAINT_TYPES:
        return [Finding(
            path=path,
            rule_id=row.id,
            severity=ERROR,
            code="UNKNOWN_CONSTRAINT_TYPE",
            message=f"Unknown constraint type '{row.type}'. Available: "
                    f"{', '.join(sorted(CONSTRAINT_TYPES))}.")]

    accepted = constraint_params()[row.type]
    unexpected = sorted(set(row.params) - accepted)
    findings = [
        Finding(path=f"{path}.params.{key}",
                rule_id=row.id,
                severity=ERROR,
                code="UNKNOWN_PARAMETER",
                message=f"{row.type} does not take '{key}'. It accepts: "
                        f"{', '.join(sorted(accepted))}.")
        for key in unexpected
    ]

    findings.extend(_probe_construction(path, row))

    return findings


def _probe_construction(path: str,
                        row: ConstraintRow) -> list[Finding]:
    """Build the constraint to surface the class's own validation.

    Cheaper and more honest than restating every bound check here: the classes
    already reject a minimum above a maximum, a negative turnover budget and a
    cardinality below one, and duplicating those rules would let the two drift.
    """
    try:
        CONSTRAINT_TYPES[row.type](**row.params)
    except TypeError as exc:
        return [Finding(path=f"{path}.params",
                        rule_id=row.id,
                        severity=ERROR,
                        code="MISSING_PARAMETER",
                        message=f"{row.type} could not be built: {exc}")]
    except Exception as exc:
        # Broad on purpose. ValueError is what the classes raise today, but a
        # constraint that starts validating differently should surface as a
        # finding against its row rather than as a 500.
        return [Finding(path=f"{path}.params",
                        rule_id=row.id,
                        severity=ERROR,
                        code="INVALID_PARAMETER",
                        message=str(exc))]

    return []


def _validate_set_shape(document: ConstraintSet) -> list[Finding]:
    """Check the set as a whole, rather than row by row."""
    findings: list[Finding] = []

    kinds = [row.type for row in document.constraints]

    if kinds.count("FullInvestment") > 1:
        findings.append(Finding(
            path="constraints",
            severity=ERROR,
            code="DUPLICATE_FULL_INVESTMENT",
            message="Two full-investment constraints cannot both hold unless "
                    "they agree, and if they agree one is redundant."))

    if "FullInvestment" not in kinds and kinds:
        findings.append(Finding(
            path="constraints",
            severity=WARNING,
            code="NO_INVESTMENT_TARGET",
            message="Nothing fixes how much is invested, so the solve is free "
                    "to hold less than the whole portfolio."))

    if "Cardinality" in kinds:
        findings.append(Finding(
            path="constraints",
            severity=WARNING,
            code="NON_CONVEX_CONSTRAINT",
            message="A holding limit is not convex. It is honoured by a "
                    "heuristic — solve, keep the largest positions, re-solve — "
                    "so the answer satisfies the limit but is not proven "
                    "optimal."))

    seen: set[str] = set()
    for position, row in enumerate(document.constraints):
        if row.id and row.id in seen:
            findings.append(Finding(
                path=f"constraints[{position}].id",
                rule_id=row.id,
                severity=ERROR,
                code="DUPLICATE_ROW_ID",
                message=f"Row id '{row.id}' is used more than once, so a "
                        f"binding constraint could not be traced back to one "
                        f"row."))
        if row.id:
            seen.add(row.id)

    return findings


def build_constraints(document: ConstraintSet) -> list[Constraint]:
    """Turn a stored set into optimiser constraint objects.

    Args:
        document: A set that has already been validated.

    Returns:
        list: The constraints, in the order the rows carry them. Order matters
        only for reporting, since the solver applies them all at once.

    Raises:
        KeyError: If a row's type is unknown, which validation would have
            caught. Reaching here means the set was never validated.
    """
    return [CONSTRAINT_TYPES[row.type](**row.params)
            for row in document.constraints]


def has_errors(findings: list[Finding]) -> bool:
    """Whether any finding blocks saving or running."""
    return any(finding.severity == ERROR for finding in findings)


def label_map(document: ConstraintSet) -> dict[str, str]:
    """Constraint label to the row id that produced it.

    The optimiser reports binding constraints by their own generated labels —
    "maximum weight 10.0000% on AAA" — which say what bound but not which row
    of the editor to highlight. Building the same objects in the same order and
    reading their labels back gives the mapping, without the optimiser needing
    to know that a stored document exists.
    """
    mapping: dict[str, str] = {}

    for row in document.constraints:
        if not row.id:
            continue

        constraint = CONSTRAINT_TYPES[row.type](**row.params)
        for condition in constraint.conditions(_placeholder_assets(row)):
            mapping[condition.label] = row.id

    return mapping


def _placeholder_assets(row: ConstraintRow) -> list[str]:
    """Assets a row's labels need to be generated.

    Position bounds label themselves per asset, so they need the names they
    cover; everything else labels itself once and ignores this.
    """
    named = row.params.get("assets") or row.params.get("members") or []

    return [str(name) for name in named]


def constraint_types() -> dict[str, list[str]]:
    """Every constraint type and the parameters it accepts.

    Served so a client can build its editor from the same source the solver
    reads, rather than from a copy that drifts.
    """
    return {name: sorted(params)
            for name, params in constraint_params().items()}
