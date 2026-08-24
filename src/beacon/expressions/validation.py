# src/beacon/expressions/validation.py
"""
Checking an expression against the data it will run on.

`data.reference.sectr` must not silently select nothing. A screen that quietly
matches no instruments produces an empty index and no explanation, which is
the same failure the universe member validation exists to prevent (BN-132):
the result looks like a legitimate answer, so nobody investigates.

## Findings, not exceptions

Validation returns a list rather than raising on the first problem, in the
shape the pipeline and universe validation already use. A user fixing a screen
wants every mistake at once, and a client needs to point at the offending rule
rather than show a message with no anchor.

An expression with no findings is valid. `errors_in` is the same list filtered
to what actually blocks.

## "Did you mean" is the difference between useful and frustrating

`sectr` is one edit from `sector`. A validator that knows this and does not say
so has chosen to be unhelpful — the information is already in hand, and the
user is looking at a screen that is *nearly* right.

## The loaded data is the authority

Not the declaration in `namespaces`, and not a generated stub. A store may
carry reference columns nobody declared, so an undeclared name is checked
against what is actually loaded before it is called wrong. And a stub kept
from an older store autocompletes a field the data no longer has — which is
safe precisely because validation runs against the data and produces a finding
rather than a wrong selection.
"""
import difflib
import logging
from dataclasses import dataclass
from typing import Any

from ..data.fetcher import DataFetcher
from .core import Expression, Field, fields_in
from .namespaces import (
    ACTION_COLUMNS,
    ACTIONS,
    DERIVED_COLUMNS,
    FEATURES,
    MARKET,
    REFERENCE,
    column_for,
)

logger = logging.getLogger(__name__)

# Stable codes, so a client can branch on the kind of problem rather than
# matching on message text.
UNKNOWN_FIELD = "UNKNOWN_FIELD"
UNKNOWN_FEATURE_TYPE = "UNKNOWN_FEATURE_TYPE"
UNKNOWN_NAMESPACE = "UNKNOWN_NAMESPACE"

ERROR = "error"
WARNING = "warning"

# How close a name has to be before it is offered as a suggestion. High enough
# that an unrelated column is not proposed -- a wrong suggestion is worse than
# none, because it sends somebody to check the wrong thing.
SIMILARITY = 0.6
MAX_SUGGESTIONS = 3


@dataclass(frozen=True)
class Finding:
    """One problem with an expression.

    Deliberately a plain dataclass rather than the server's pydantic `Finding`:
    expressions are a library concern and must not depend on the server, and
    `as_dict` produces exactly the shape the API already returns.
    """
    path: str
    code: str
    message: str
    severity: str = ERROR

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "severity": self.severity,
                "code": self.code, "message": self.message}


def validate(expression: Expression,
             fetcher: DataFetcher) -> list[Finding]:
    """Every problem with an expression, checked against loaded data.

    Args:
        expression: The tree to check.
        fetcher: The data it will be resolved against.

    Returns:
        list[Finding]: Empty when the expression is valid.
    """
    findings = []

    for field in fields_in(expression):
        findings.extend(_check(field, fetcher))

    return findings


def errors_in(expression: Expression,
              fetcher: DataFetcher) -> list[Finding]:
    """Only the findings that block."""
    return [finding for finding in validate(expression, fetcher)
            if finding.severity == ERROR]


def is_valid(expression: Expression,
             fetcher: DataFetcher) -> bool:
    """Whether an expression can run against this data."""
    return not errors_in(expression, fetcher)


def _check(field: Field,
           fetcher: DataFetcher) -> list[Finding]:
    """One field."""
    if field.namespace == FEATURES:
        return _check_feature(field, fetcher)

    if field.namespace == MARKET:
        return _check_column(field, _market_names(fetcher))

    if field.namespace == REFERENCE:
        return _check_column(field, _reference_names(fetcher))

    if field.namespace == ACTIONS:
        return _check_column(field, _action_names(fetcher))

    return [Finding(field.path, UNKNOWN_NAMESPACE,
                    f"'{field.namespace}' is not a dataset.")]


def _check_column(field: Field,
                  available: list[str]) -> list[Finding]:
    """A market, reference or action column against what is loaded."""
    if column_for(field) in available:
        return []

    return [Finding(field.path, UNKNOWN_FIELD,
                    _no_such(field.name, field.namespace,
                             [name.lower() for name in available]))]


def _check_feature(field: Field,
                   fetcher: DataFetcher) -> list[Finding]:
    """A feature field, and the dataset it names."""
    features = fetcher.features
    types = features.types

    if not types:
        return [Finding(field.path, UNKNOWN_FEATURE_TYPE,
                        "no feature data is loaded, so nothing can screen on "
                        f"'{field.name}'.")]

    if field.dataset is not None and field.dataset not in types:
        return [Finding(
            field.path, UNKNOWN_FEATURE_TYPE,
            _no_such(field.dataset, "the loaded feature datasets", types))]

    if field.name in features.fields(field.dataset):
        return []

    where = (f"the '{field.dataset}' dataset" if field.dataset
             else "any loaded feature dataset")

    return [Finding(field.path, UNKNOWN_FIELD,
                    _no_such(field.name, where,
                             features.fields(field.dataset)))]


def _no_such(name: str,
             where: str,
             available: list[str]) -> str:
    """The message, with a suggestion when there is a near miss.

    The suggestion is the point. Somebody looking at a screen that is nearly
    right needs to know *which* character is wrong, and the edit distance is
    already computed by the time the error is written.
    """
    message = f"'{name}' is not in {where}."
    close = difflib.get_close_matches(name, available, n=MAX_SUGGESTIONS,
                                      cutoff=SIMILARITY)

    if close:
        suggestions = " or ".join(f"'{match}'" for match in close)

        return f"{message} Did you mean {suggestions}?"

    if available:
        listed = ", ".join(sorted(available)[:MAX_SUGGESTIONS])

        return f"{message} Available: {listed}..."

    return f"{message} Nothing is loaded to match it against."


def _market_names(fetcher: DataFetcher) -> list[str]:
    """Loaded market columns, plus the derived ones the server computes.

    Derived fields are columns nowhere -- they are computed per request
    (BN-133) -- so checking only the loaded frame would reject `market_cap`,
    which is the field most likely to be screened on.
    """
    return list(fetcher.market_columns) + [name.upper()
                                           for name in DERIVED_COLUMNS]


def _reference_names(fetcher: DataFetcher) -> list[str]:
    """Loaded reference columns. Empty when no reference data is loaded."""
    return list(fetcher.reference_columns or [])


def _action_names(fetcher: DataFetcher) -> list[str]:
    """The action fields a client can screen on.

    Taken from the declaration rather than the frame: `kind` and `status` are
    computed on the way out (BN-119) and are not columns in the stored table,
    so reading the frame would reject the two fields the client documentation
    tells people to branch on.
    """
    return [name.upper() for name in ACTION_COLUMNS]
