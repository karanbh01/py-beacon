# src/beacon/optimise/config.py
"""
The serialisable description of an optimisation.

Optimised indices have to be cacheable, and BN-160's fingerprint machinery
keys a calculation off catalogue-registered types whose constructor parameters
can be read back off the instance and carried as JSON. Rules and weighting
schemes already follow that convention; this module gives constraints the same
round trip — an instance becomes a ``{type, params}`` payload, and the payload
becomes an instance that solves identically.

The payload shape is deliberately the one that already exists twice: the
fingerprint machinery in `beacon.index.cache` hashes rules and schemes as
``{type, params}``, and the server's ``ConstraintRow`` carries ``type`` and
``params`` on the wire. A third shape would mean a translation layer, which is
where meanings drift.

An unregistered constraint class is refused, never guessed at: a payload built
from introspection of a class the catalogue does not know would be a key
nothing else can verify, which is exactly the incomplete-key case the BN-160
safety rule exists for.

:class:`OptimisationConfig` is the object form of the same description — what
an optimised run is asked to do, held together so `Backtest.run` (BN-167) can
take one argument rather than a growing list of them.
"""
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .. import catalogue
from ..exceptions import CalculationError
from ..risk.model import RiskModel
from .constraints import Constraint

logger = logging.getLogger(__name__)

# The one objective the solver offers today. A string rather than an enum so a
# stored config and a wire payload carry it verbatim.
MIN_TRACKING_ERROR = "min_tracking_error"


@dataclass(frozen=True)
class OptimisationConfig:
    """What an optimised run is asked to do.

    Attributes:
        objective: What to minimise. Only ``"min_tracking_error"`` exists
            today, and the field names it anyway so a stored config says what
            it meant when other objectives arrive.
        constraints: What the answer must satisfy, as
            :class:`~beacon.optimise.constraints.Constraint` instances. Empty
            means the solver's own default of full investment alone.
        risk_model: RESERVED — carried but unused at launch. The slot exists so
            covariance-aware optimised runs can arrive without changing this
            shape; nothing reads it yet, and passing one changes no result.
    """
    objective: str = MIN_TRACKING_ERROR
    constraints: Sequence[Constraint] = ()
    risk_model: RiskModel | None = None


def constraint_payload(constraint: Constraint) -> dict[str, Any]:
    """One constraint as its registered name plus parameter values.

    Parameter names come from the catalogue's constructor introspection, each
    value read off the instance attribute of the same name — the convention
    every registered type follows and the fingerprint machinery keys by.

    Args:
        constraint: The instance to describe.

    Returns:
        dict: ``{"type": class name, "params": {name: value}}``, with every
        value JSON-serialisable.

    Raises:
        CalculationError: If the class is not registered under the CONSTRAINT
            kind, keeps no attribute for one of its constructor parameters, or
            holds a value JSON cannot carry. Refused rather than guessed at —
            a payload nothing can rebuild is worse than no payload.
    """
    cls = type(constraint)

    if catalogue.classes(catalogue.CONSTRAINT).get(cls.__name__) is not cls:
        raise CalculationError(
            "Optimiser",
            f"the constraint {cls.__name__} is not registered in the "
            f"catalogue, so its configuration cannot be serialised.")

    params = {parameter.name: _readable_value(constraint, parameter.name)
              for parameter in catalogue.parameters_of(cls)}

    return {"type": cls.__name__, "params": params}


def constraint_from_payload(payload: Mapping[str, Any]) -> Constraint:
    """Rebuild a constraint from a ``{type, params}`` payload.

    The inverse of :func:`constraint_payload`, and the builder the server's
    constraint rows go through — one code path from a stored document to the
    object the solver receives.

    Args:
        payload: A mapping carrying ``type`` and, optionally, ``params``.

    Returns:
        Constraint: A fresh instance that solves identically to the one the
        payload was taken from.

    Raises:
        CalculationError: If the type names nothing registered. A constructor
            rejecting the params (a missing argument, a minimum above a
            maximum) propagates untouched, so the class's own message reaches
            the caller.
    """
    name = payload.get("type")
    registered = catalogue.classes(catalogue.CONSTRAINT)

    if not isinstance(name, str) or name not in registered:
        raise CalculationError(
            "Optimiser",
            f"unknown constraint type '{name}'. Available: "
            f"{', '.join(sorted(registered))}.")

    built = registered[name](**dict(payload.get("params") or {}))

    if not isinstance(built, Constraint):
        raise CalculationError(
            "Optimiser",
            f"'{name}' is registered under the CONSTRAINT kind but is not a "
            f"Constraint — the registration itself is wrong.")

    return built


def _readable_value(constraint: Constraint,
                    name: str) -> Any:
    """One constructor parameter read back off the instance, JSON-checked."""
    cls_name = type(constraint).__name__

    if not hasattr(constraint, name):
        raise CalculationError(
            "Optimiser",
            f"the constraint {cls_name} keeps no attribute for its "
            f"constructor parameter '{name}'.")

    value = getattr(constraint, name)
    if not _is_jsonable(value):
        raise CalculationError(
            "Optimiser",
            f"the constraint {cls_name} parameter '{name}' does not "
            f"serialise to JSON.")

    return value


def _is_jsonable(value: Any) -> bool:
    """Whether a JSON dump can carry this value."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False

    return True
