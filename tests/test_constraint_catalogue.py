# tests/test_constraint_catalogue.py
"""BN-166: constraints in the catalogue, and the {type, params} round trip.

The catalogue registration itself is covered by the completeness check in
test_catalogue.py; what this file proves is the round trip that registration
exists for — an instance becomes a JSON-serialisable ``{type, params}``
payload, the payload becomes an instance, and the two solve identically. That
equality is the acceptance criterion for cacheable optimised indices: a
fingerprint built from the payload only means something if the payload really
does determine the solve.
"""
import gc
import json

import pandas as pd
import pytest

from beacon import catalogue
from beacon.exceptions import CalculationError
from beacon.optimise import (
    Cardinality,
    ExpectedReturnTarget,
    FullInvestment,
    GroupBounds,
    OptimisationConfig,
    PositionBounds,
    TurnoverBudget,
    constraint_from_payload,
    constraint_payload,
    minimise_tracking_error,
)
from beacon.optimise.constraints import Constraint

TARGET = {"A": 0.5, "B": 0.3, "C": 0.2}

# A starting book well away from the target, so a turnover budget binds.
CURRENT_HOLDINGS = {"A": 0.2, "B": 0.2, "C": 0.6}

EXPECTED_RETURNS = {"A": 0.08, "B": 0.05, "C": 0.02}


def specimens() -> list[Constraint]:
    """One fully parameterised instance of every registered constraint.

    Each is built so it actually constrains the module's target — a bound the
    solve never touches would make "the round-tripped constraint solves
    identically" vacuously true.
    """
    return [
        FullInvestment(target=0.98),
        PositionBounds(minimum=0.05, maximum=0.45, assets=["A", "B"]),
        GroupBounds(name="tech", members=["A", "C"], minimum=0.1, maximum=0.6),
        TurnoverBudget(maximum=0.1, current_weights=CURRENT_HOLDINGS),
        ExpectedReturnTarget(expected_returns=EXPECTED_RETURNS, target=0.05),
        Cardinality(maximum=2),
    ]


def specimen_ids() -> list[str]:
    return [type(constraint).__name__ for constraint in specimens()]


def solved(constraint: Constraint) -> pd.Series:
    """The optimal weights under one constraint (plus a budget to solve on)."""
    rules = ([constraint] if isinstance(constraint, FullInvestment)
             else [FullInvestment(), constraint])

    return minimise_tracking_error(TARGET, rules).weights


class TestEverySpecimenIsCovered:

    def test_the_specimens_span_the_registry(self):
        """One specimen per registered class, so a newly registered constraint
        fails here until it gets a round-trip test rather than escaping one."""
        covered = {type(constraint).__name__ for constraint in specimens()}

        assert covered == catalogue.registered_names(catalogue.CONSTRAINT)


class TestPayload:

    @pytest.mark.parametrize("constraint", specimens(), ids=specimen_ids())
    def test_payload_carries_type_and_params(self,
                                             constraint):
        payload = constraint_payload(constraint)

        assert payload["type"] == type(constraint).__name__
        assert set(payload["params"]) == catalogue.parameter_names(
            catalogue.CONSTRAINT, payload["type"])

    @pytest.mark.parametrize("constraint", specimens(), ids=specimen_ids())
    def test_payload_is_json_serialisable(self,
                                          constraint):
        """The point of the getattr convention: the payload must survive the
        canonical JSON dump the fingerprint machinery hashes."""
        json.dumps(constraint_payload(constraint), sort_keys=True)

    def test_params_are_the_constructor_values(self):
        payload = constraint_payload(
            PositionBounds(minimum=0.05, maximum=0.45, assets=["A", "B"]))

        assert payload["params"]["minimum"] == 0.05
        assert payload["params"]["maximum"] == 0.45
        assert list(payload["params"]["assets"]) == ["A", "B"]

    def test_an_unregistered_subclass_is_refused(self):
        """The BN-160 uncacheable path: a class the catalogue does not know
        cannot be described, and guessing from its shape would produce a key
        nothing else can rebuild or verify."""
        class Unregistered(FullInvestment):
            """A constraint nobody registered."""

        try:
            with pytest.raises(CalculationError, match="not registered"):
                constraint_payload(Unregistered())
        finally:
            # Dropped so it cannot outlive this test and trip the catalogue
            # completeness check. __subclasses__ holds weak references, so
            # collecting is what actually removes it.
            del Unregistered
            gc.collect()


class TestRebuild:

    def test_an_unknown_type_is_refused_naming_the_alternatives(self):
        with pytest.raises(CalculationError, match="FullInvestment"):
            constraint_from_payload({"type": "NotAConstraint", "params": {}})

    def test_a_missing_type_is_refused(self):
        with pytest.raises(CalculationError, match="unknown constraint type"):
            constraint_from_payload({"params": {"target": 1.0}})

    def test_constructor_validation_still_speaks(self):
        """The classes' own rejections must reach the caller untranslated —
        the builder adds no second validation layer for them to drift from."""
        with pytest.raises(ValueError, match="minimum weight"):
            constraint_from_payload({"type": "PositionBounds",
                                     "params": {"minimum": 0.9, "maximum": 0.1}})


class TestRoundTripSolvesIdentically:
    """The acceptance criterion: payload → rebuild → the same answer."""

    @pytest.mark.parametrize("constraint", specimens(), ids=specimen_ids())
    def test_rebuilt_constraint_solves_identically(self,
                                                   constraint):
        """Through an actual JSON dump and load, not just the dict — the
        stored form is text, and text is what a cache key or a document
        round-trips (a tuple of assets comes back as a list, and must not
        matter)."""
        wire = json.loads(json.dumps(constraint_payload(constraint)))
        rebuilt = constraint_from_payload(wire)

        pd.testing.assert_series_equal(solved(constraint),
                                       solved(rebuilt),
                                       check_exact=True)

    @pytest.mark.parametrize("constraint", specimens(), ids=specimen_ids())
    def test_rebuilt_payload_is_stable(self,
                                       constraint):
        """Payloading the rebuilt instance reproduces the dump byte for byte,
        so a fingerprint hashed from either agrees."""
        first = constraint_payload(constraint)
        second = constraint_payload(constraint_from_payload(
            json.loads(json.dumps(first))))

        canonical = [json.dumps(payload, sort_keys=True)
                     for payload in (first, second)]

        assert canonical[0] == canonical[1]


class TestOptimisationConfig:

    def test_defaults(self):
        config = OptimisationConfig()

        assert config.objective == "min_tracking_error"
        assert config.constraints == ()
        assert config.risk_model is None

    def test_carries_constraints(self):
        rules = [FullInvestment(), Cardinality(maximum=2)]

        assert OptimisationConfig(constraints=rules).constraints == rules

    def test_risk_model_is_reserved(self):
        """The field exists for covariance-aware runs to arrive without a
        shape change, and nothing reads it at launch: a config carrying one
        must not alter what today's code does with the config."""
        config = OptimisationConfig(risk_model=None)

        assert config.risk_model is None
        assert "RESERVED" in (OptimisationConfig.__doc__ or "")
