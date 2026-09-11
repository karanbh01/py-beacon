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


def _parameter(parameter: catalogue.Parameter,
               schemas: dict[str, str]) -> ParameterSpec:
    """One catalogue parameter as its API shape.

    Args:
        parameter: The introspected parameter.
        schemas: Parameter name -> component schema, as the class declared it.
    """
    return ParameterSpec(
        name=parameter.name,
        type=parameter.type,
        required=parameter.required,
        default=parameter.default,
        label=parameter.label,
        order=parameter.order,
        choices=list(parameter.choices) if parameter.choices else None,
        help=parameter.help,
        # Looked up by name in what the CLASS declared, which is not the same
        # as inferring a ref from the name (BN-175): a rule that takes a tree
        # says so, and a rule that happens to call a scalar `expression` does
        # not acquire one.
        ref=schemas.get(parameter.name))


def specs_for(kind: str) -> list[TypeSpec]:
    """Every registered type of one kind, ready to serve.

    Args:
        kind: catalogue.SELECTION, WEIGHTING or CONSTRAINT.

    Returns:
        list: Type specs, name-ordered, each carrying its parameters in the
        order a form should show them, and — for a constraint — the unit its
        slack is reported in.
    """
    # The class, not just the entry: `UNIT` is declared on the constraint
    # class, and reading it here is what keeps the constraint editor and a
    # preview's solve block quoting one source rather than two. `PARAM_SCHEMAS`
    # is read the same way and for the same reason (BN-175) — a rule that takes
    # a structured parameter names the schema it takes, beside the code that
    # parses it.
    classes = catalogue.classes(kind)

    return [TypeSpec(name=entry.name,
                     label=entry.label,
                     summary=entry.summary,
                     parameters=[
                         _parameter(parameter,
                                    getattr(classes[entry.name],
                                            "PARAM_SCHEMAS", {}))
                         for parameter in entry.parameters],
                     slack_unit=getattr(classes[entry.name], "UNIT", None))
            for entry in catalogue.entries(kind)]
