# src/beacon/catalogue.py
"""
A catalogue of the configurable types a client can offer, built from the
classes themselves.

`RuleSpec` carries a free-text `type` and an open `params` dict, and nothing
published which types existed or what each accepted. So the methodology editor
could only be a text box and a list of key/value pairs: a user had to *know*
that the rule is spelled `MarketCapRule` and its parameter `min_market_cap`,
and a typo surfaced only after a round trip. Everything needed to render a
proper form was already in the constructor signatures.

## What is introspected and what is declared

Introspection reads `__init__` and gets the facts: parameter names, types,
which are required, and their defaults. Those cannot go stale, because they
*are* the signature.

Three things introspection cannot know, so they are declared:

* a **label** — `min_avg_daily_volume` is a field name, "Minimum ADV" is a label
* an **order** — a form has a designed reading order; a signature's order is
  whatever was convenient to write
* **choices** — a parameter annotated `str` may accept exactly three values,
  and the annotation cannot say so

The split is deliberate and the declaration is kept as small as possible: the
half that can drift from reality is then only the presentational half, where
drift means an awkward label rather than a wrong form.

## Registration is by decorator, and forgetting it is a test failure

A hand-kept list is the thing this module exists to delete — three of them
existed, and they had to agree with each other and with the classes. Here a
class registers itself where it is defined. A new rule that forgets to is
caught by a completeness test walking the base class's subclasses, because the
failure mode of a missed registration is silent: nothing breaks, the editor
simply never offers the rule and nobody notices.
"""
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin

logger = logging.getLogger(__name__)

# The groups a client asks for. Selection and weighting drive the methodology
# editor; constraints drive the optimiser's. They share this module so the two
# editors can share one component rather than converging by hand.
SELECTION = "selection"
WEIGHTING = "weighting"
CONSTRAINT = "constraint"

# Display types, not Python types. A client renders a control from these, so
# they name what the input should be rather than what the annotation says.
NUMBER = "number"
INTEGER = "integer"
BOOLEAN = "boolean"
STRING = "string"
JSON = "json"

_SCALAR_TYPES = {bool: BOOLEAN, int: INTEGER, float: NUMBER, str: STRING}

# Defaults are serialised into the response, so only values a client can
# actually receive are reported. Anything else becomes None rather than a
# repr that would render as a literal string in a form field.
_JSON_SCALARS = (bool, int, float, str, type(None))


@dataclass(frozen=True)
class Display:
    """What introspection cannot know about one parameter.

    Attributes:
        label: Human-readable field name.
        order: Position in the form. Defaults to signature order.
        choices: The values this parameter accepts, when it is a closed set.
        help: One line of guidance shown with the field.
    """
    label: str
    order: int | None = None
    choices: tuple[str, ...] | None = None
    help: str | None = None


@dataclass(frozen=True)
class Parameter:
    """One parameter of a configurable type, ready to render."""
    name: str
    type: str
    required: bool
    default: Any = None
    label: str = ""
    order: int = 0
    choices: tuple[str, ...] | None = None
    help: str | None = None


@dataclass(frozen=True)
class Entry:
    """One configurable type: what it is called, and what it takes."""
    name: str
    kind: str
    label: str
    summary: str
    parameters: tuple[Parameter, ...] = field(default_factory=tuple)


# kind -> name -> (class, entry). One structure, so the class that builds an
# object and the entry that describes it can never name different things.
_REGISTRY: dict[str, dict[str, tuple[type, Entry]]] = {}


def display_type(annotation: Any) -> tuple[str, bool]:
    """Map a constructor annotation to a display type.

    Args:
        annotation: The annotation as `inspect.signature` resolved it.

    Returns:
        tuple: The display type, and whether the annotation admits None. The
        second is what distinguishes "leave this blank" from "this is required"
        for a parameter that also carries a default.
    """
    if annotation is inspect.Parameter.empty:
        return JSON, False

    origin = get_origin(annotation)
    if origin is Union or origin is type(int | None):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        optional = len(args) != len(get_args(annotation))

        if len(args) == 1:
            inner, _ = display_type(args[0])

            return inner, optional

        return JSON, optional

    # bool before int: bool is a subclass of int, so an isinstance-style check
    # in the other order would render every checkbox as a number field.
    for python_type, name in _SCALAR_TYPES.items():
        if annotation is python_type:
            return name, False

    return JSON, False


def _default_label(name: str) -> str:
    """A readable label for a parameter nobody declared one for."""
    return name.replace("_", " ").capitalize()


def _summary_of(cls: type) -> str:
    """The first meaningful line of a class docstring."""
    doc = inspect.getdoc(cls) or ""

    for line in doc.splitlines():
        if line.strip():
            return line.strip()

    return ""


def parameters_of(cls: type,
                  displays: dict[str, Display] | None = None) -> tuple[Parameter, ...]:
    """Read a class's constructor into renderable parameters.

    Args:
        cls: The class to introspect.
        displays: Per-parameter presentation, keyed by parameter name.

    Returns:
        tuple: Parameters in declared order, then signature order.
    """
    declared = displays or {}

    # The signature of *calling the class*, which resolves to __init__ with
    # `self` already dropped. Asking for `cls.__init__` directly would work
    # too, but only by reaching through an instance attribute that a subclass
    # is free to replace — which mypy rejects, and rightly.
    #
    # eval_str resolves string annotations, so a module using postponed
    # evaluation still yields real types rather than the text of them.
    signature = inspect.signature(cls, eval_str=True)

    parameters = []
    for position, (name, parameter) in enumerate(signature.parameters.items()):
        # `self` is already gone — signature(cls) describes calling the class,
        # not the function. What is left to drop is *args/**kwargs, which name
        # no field a form could render.
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
            continue

        display = declared.get(name)
        kind, optional = display_type(parameter.annotation)
        has_default = parameter.default is not inspect.Parameter.empty
        default = parameter.default if has_default else None

        parameters.append(Parameter(
            name=name,
            type=kind,
            # Required means the constructor will not accept the call without
            # it. An annotation admitting None but carrying no default is
            # still required — the caller has to pass None explicitly.
            required=not has_default,
            default=default if isinstance(default, _JSON_SCALARS) else None,
            label=display.label if display else _default_label(name),
            order=(display.order if display and display.order is not None
                   else position),
            choices=display.choices if display else None,
            help=display.help if display else None))

        if optional and not has_default:
            logger.debug("%s.%s admits None but has no default.",
                         cls.__name__, name)

    return tuple(sorted(parameters, key=lambda item: item.order))


def register(kind: str,
             label: str,
             fields: dict[str, Display] | None = None) -> Any:
    """Register a class in the catalogue, as a decorator.

    Args:
        kind: SELECTION, WEIGHTING or CONSTRAINT.
        label: Human-readable name for the type itself.
        fields: Per-parameter presentation. Anything omitted falls back to the
            signature: a derived label and signature order.

    Returns:
        The class, unchanged. Registration is a side effect, so decorating
        never alters behaviour — a rule works identically registered or not,
        which is precisely why forgetting needs a test rather than showing up
        at runtime.
    """
    def decorate(cls: type) -> type:
        entry = Entry(name=cls.__name__,
                      kind=kind,
                      label=label,
                      summary=_summary_of(cls),
                      parameters=parameters_of(cls, fields))

        _REGISTRY.setdefault(kind, {})[cls.__name__] = (cls, entry)

        return cls

    return decorate


def entries(kind: str) -> list[Entry]:
    """Every registered type of one kind, by name."""
    return [registered[1]
            for _, registered in sorted(_REGISTRY.get(kind, {}).items())]


def entry_for(kind: str,
              name: str) -> Entry | None:
    """One entry, or None if nothing of that name is registered."""
    found = _REGISTRY.get(kind, {}).get(name)

    return found[1] if found else None


def classes(kind: str) -> dict[str, type]:
    """Name -> class for one kind.

    What replaces the hand-kept builder tables: the object that gets
    constructed and the entry that describes it come from one registration, so
    they cannot name different things.
    """
    return {name: cls for name, (cls, _) in _REGISTRY.get(kind, {}).items()}


def parameter_names(kind: str,
                    name: str) -> set[str]:
    """The parameters one type accepts, for validating a submitted spec."""
    entry = entry_for(kind, name)

    return {parameter.name for parameter in entry.parameters} if entry else set()


def registered_names(kind: str) -> set[str]:
    """Every registered name of one kind."""
    return set(_REGISTRY.get(kind, {}))
