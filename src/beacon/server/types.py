# src/beacon/server/types.py
"""
Serving the catalogue: turning registered classes into renderable type specs.

One adapter for both editors. The methodology editor and the optimiser's
constraint editor ask the same question — what types exist, and what does each
take — so they get the same answer shape and a client can render both with one
component. That was the point of putting the registry in `beacon.catalogue`
rather than beside either endpoint.
"""
from .. import catalogue
from .schemas import ParameterSpec, TypeSpec


def _parameter(parameter: catalogue.Parameter) -> ParameterSpec:
    """One catalogue parameter as its API shape."""
    return ParameterSpec(
        name=parameter.name,
        type=parameter.type,
        required=parameter.required,
        default=parameter.default,
        label=parameter.label,
        order=parameter.order,
        choices=list(parameter.choices) if parameter.choices else None,
        help=parameter.help)


def specs_for(kind: str) -> list[TypeSpec]:
    """Every registered type of one kind, ready to serve.

    Args:
        kind: catalogue.SELECTION, WEIGHTING or CONSTRAINT.

    Returns:
        list: Type specs, name-ordered, each carrying its parameters in the
        order a form should show them.
    """
    return [TypeSpec(name=entry.name,
                     label=entry.label,
                     summary=entry.summary,
                     parameters=[_parameter(parameter)
                                 for parameter in entry.parameters])
            for entry in catalogue.entries(kind)]
