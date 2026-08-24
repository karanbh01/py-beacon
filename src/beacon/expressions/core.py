# src/beacon/expressions/core.py
"""
The expression tree: fields, comparisons, and boolean composition.

A typed way to name a datapoint and say something about it:

    data.reference.sector == "Financials"
    data.market.adv_3m > 1_000_000
    (data.features.fundamentals.pe_ratio < 15) & (data.market.adv_3m > 1e6)

Nothing here evaluates. Building an expression produces a **tree**, which is
then either serialised into a stored index definition or resolved against one
instrument on one date. That indirection is the whole point: an index
definition is a document, and a screen that cannot be written down cannot be
saved, reloaded, or sent to a client.

## It compiles into the existing rule envelope

A rule in a stored pipeline is `{"id", "type", "params"}` and always has been.
An expression does not sit beside that — it becomes the `params` of a rule
type (`ExpressionRule`), so there is one representation of a pipeline rather
than two that drift. `to_dict` and `from_dict` here are what make that
possible, and the round-trip is exact rather than approximate: a definition
saved and reloaded must screen identically, or a backtest is not reproducible.

## Two hazards, which are most of why this module is written carefully

**`__eq__` does not return a bool.** That is what lets `sector == "Financials"`
build a tree, and it breaks three things Python assumes:

* `assert expression == x` in a test would pass silently, asserting nothing
* `expression in [...]` compares by `__eq__` and then takes a truth value
* an object defining `__eq__` loses `__hash__` unless it declares one

The first two are handled by `__bool__` raising rather than returning a value,
which converts a silent wrong answer into an error that names the problem.

The third is handled by declaring `__hash__` — but only partly, and the limit
is worth stating rather than glossing. **CPython checks `is` before `==`** when
looking up a set or dict entry, so reusing the *same* field object in a set or
a `in [...]` test works and never reaches `__eq__`; two *distinct* objects
naming the same datapoint hash alike, fall through to `__eq__`, get a tree back
and raise. A `Field` is therefore a safe dict key only when the same instance
is reused. `Field.key` is the plain tuple to use instead, and
`distinct_fields_in` deduplicates with it rather than with a set of fields.

**`and` and `or` cannot be overloaded.** Python evaluates `a and b` by taking
`bool(a)` and returning one operand or the other; there is no hook. So
`(a == 1) and (b > 2)` would quietly discard half the expression. This is the
most common pandas bug there is. The defence is `&` and `|` for composition
plus a `__bool__` that raises and names the fix — the error is the only place
a user finds out, so it says exactly what to type instead.
"""
from typing import Any

from ..exceptions import ExpressionError

# How a comparison is spelled once serialised. Words rather than symbols,
# because these live in a stored JSON document read back by a client: "ge"
# survives a round trip through JSON and a Python operator does not.
GT = "gt"
GE = "ge"
LT = "lt"
LE = "le"
EQ = "eq"
NE = "ne"
IN = "in"
BETWEEN = "between"

COMPARISONS = (GT, GE, LT, LE, EQ, NE, IN, BETWEEN)

# Node kinds, as they appear in `to_dict`.
FIELD = "field"
COMPARISON = "comparison"
ALL = "all"
ANY = "any"
NOT = "not"

TRUTH_VALUE_MESSAGE = (
    "an expression has no truth value. Use `&` and `|` rather than `and` "
    "and `or`, and parenthesise the operands: `(a > 1) & (b < 2)`.")


class Expression:
    """Base for everything that can be composed and serialised.

    Subclasses supply `to_dict`; composition and the truth-value guard are
    shared, so a new node type cannot forget either.
    """

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __bool__(self) -> bool:
        """Always raises.

        `and`, `or`, `not`, `if expr:` and `assert expr` all route through
        here. None of them can work — an expression is not true or false until
        it is resolved against an instrument and a date — and every one of them
        would otherwise fail silently, which is why this raises rather than
        returning a default.
        """
        raise ExpressionError(TRUTH_VALUE_MESSAGE)

    def __and__(self,
                other: "Expression") -> "Expression":
        return All((self, other))

    def __or__(self,
               other: "Expression") -> "Expression":
        return Any_((self, other))

    def __invert__(self) -> "Expression":
        return Not(self)


class Field(Expression):
    """A named datapoint in a namespace.

    `namespace` is the surface it came from — `reference`, `market`,
    `features` — and `dataset` narrows a feature to one `TYPE`, so two vendors
    may both ship a field called `revenue` without collision.

    A `Field` is an `Expression` so it composes, but on its own it says
    nothing: comparing it is what produces something screenable.
    """

    def __init__(self,
                 namespace: str,
                 name: str,
                 dataset: str | None = None):
        self.namespace = namespace
        self.name = name
        self.dataset = dataset

    # --- comparison, as methods -------------------------------------------
    #
    # The operators below are sugar over these. Somebody who finds operator
    # overloading surprising -- a reasonable position, given what `__eq__`
    # does here -- has a plain way through, and it is the same code path
    # rather than a second one that can drift.

    def gt(self,
           value: Any) -> "Comparison":
        """Greater than."""
        return Comparison(self, GT, value)

    def ge(self,
           value: Any) -> "Comparison":
        """Greater than or equal to."""
        return Comparison(self, GE, value)

    def lt(self,
           value: Any) -> "Comparison":
        """Less than."""
        return Comparison(self, LT, value)

    def le(self,
           value: Any) -> "Comparison":
        """Less than or equal to."""
        return Comparison(self, LE, value)

    def eq(self,
           value: Any) -> "Comparison":
        """Equal to."""
        return Comparison(self, EQ, value)

    def ne(self,
           value: Any) -> "Comparison":
        """Not equal to."""
        return Comparison(self, NE, value)

    def is_in(self,
              values: "list[Any] | tuple[Any, ...]") -> "Comparison":
        """One of a set of values.

        Named `is_in` rather than `in`, which is a keyword, and deliberately
        not spelled with `__contains__`: Python coerces that to a bool, so
        `x in field` could never build a tree.
        """
        return Comparison(self, IN, list(values))

    def between(self,
                low: Any,
                high: Any) -> "Comparison":
        """Inclusive on both ends."""
        if low > high:
            raise ExpressionError(
                f"between({low!r}, {high!r}) is empty: the low bound is above "
                "the high one.")

        return Comparison(self, BETWEEN, [low, high])

    # --- the operator sugar -----------------------------------------------

    def __gt__(self,
               value: Any) -> "Comparison":
        return self.gt(value)

    def __ge__(self,
               value: Any) -> "Comparison":
        return self.ge(value)

    def __lt__(self,
               value: Any) -> "Comparison":
        return self.lt(value)

    def __le__(self,
               value: Any) -> "Comparison":
        return self.le(value)

    def __eq__(self,  # type: ignore[override]
               value: Any) -> "Comparison":
        """Builds a comparison rather than answering a question.

        Deliberately incompatible with `object.__eq__`, hence the override
        marker. The cost is that two `Field`s cannot be compared for identity
        with `==`; use `same_as` for that.
        """
        return self.eq(value)

    def __ne__(self,  # type: ignore[override]
               value: Any) -> "Comparison":
        return self.ne(value)

    # Defining `__eq__` sets `__hash__` to None unless it is declared, which
    # would make a field unhashable outright.
    #
    # Declaring it gets hashing back, but *not* the full contract, and the
    # limit is worth stating exactly because it is easy to over-claim. CPython
    # checks `is` before `==` when looking up a set or dict entry, so:
    #
    #   {field, field}          works  -- identity short-circuits
    #   field in [field]        works  -- same reason, returns True
    #   {field, same_field}     RAISES -- distinct objects, hashes collide,
    #                                     `__eq__` runs and returns a tree
    #
    # So a field is a usable dict key only when the *same object* is reused.
    # To key by datapoint instead, use `.key`, which is a plain tuple and has
    # none of this trouble.
    def __hash__(self) -> int:
        return hash((FIELD, self.namespace, self.name, self.dataset))

    @property
    def key(self) -> tuple[str, str | None, str]:
        """The datapoint this field names, as a plain hashable tuple.

        What to use as a dict key or set member. A `Field` cannot safely serve
        as one for two distinct-but-equal instances (see `__hash__` above),
        and quietly raising deep inside a resolver is the worst place to find
        that out.
        """
        return (self.namespace, self.dataset, self.name)

    def same_as(self,
                other: object) -> bool:
        """Whether two fields name the same datapoint.

        The plain `==` a field cannot offer, because `==` builds a tree.
        """
        return (isinstance(other, Field)
                and (self.namespace, self.name, self.dataset)
                == (other.namespace, other.name, other.dataset))

    @property
    def path(self) -> str:
        """How the field is written, e.g. `features.fundamentals.revenue`."""
        parts = [self.namespace, self.dataset, self.name]

        return ".".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        node: dict[str, Any] = {"node": FIELD, "namespace": self.namespace,
                                "name": self.name}

        if self.dataset is not None:
            node["dataset"] = self.dataset

        return node

    def __repr__(self) -> str:
        return f"<{self.path}>"


class Comparison(Expression):
    """A field, an operator and a value."""

    def __init__(self,
                 field: Field,
                 comparison: str,
                 value: Any):
        if comparison not in COMPARISONS:
            raise ExpressionError(
                f"unknown comparison '{comparison}': expected one of "
                f"{', '.join(COMPARISONS)}.")

        self.field = field
        self.comparison = comparison
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return {"node": COMPARISON, "field": self.field.to_dict(),
                "comparison": self.comparison, "value": self.value}

    def __repr__(self) -> str:
        return f"({self.field.path} {self.comparison} {self.value!r})"


class _Group(Expression):
    """Shared behaviour for `All` and `Any_`."""

    node = ""
    operands: tuple[Expression, ...]

    def __init__(self,
                 operands: "tuple[Expression, ...] | list[Expression]"):
        flattened: list[Expression] = []

        # `(a & b) & c` should read as one group of three rather than a nest
        # of two, so that a stored document reflects what was written and two
        # ways of writing the same screen serialise identically. Matched on
        # `node` rather than on the class, so a group only ever absorbs one of
        # its own kind -- `(a & b) | c` has to keep its shape or it changes
        # meaning.
        for operand in operands:
            if isinstance(operand, _Group) and operand.node == self.node:
                flattened.extend(operand.operands)
            else:
                flattened.append(operand)

        if not flattened:
            raise ExpressionError(f"an '{self.node}' group needs at least one "
                                  "operand.")

        self.operands = tuple(flattened)

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node,
                "operands": [operand.to_dict() for operand in self.operands]}

    def __repr__(self) -> str:
        joiner = " & " if self.node == ALL else " | "

        return "(" + joiner.join(repr(o) for o in self.operands) + ")"


class All(_Group):
    """Every operand must pass."""

    node = ALL


class Any_(_Group):
    """At least one operand must pass.

    Named with a trailing underscore so it does not shadow `typing.Any`, which
    this module also uses.
    """

    node = ANY


class Not(Expression):
    """The negation of an expression."""

    def __init__(self,
                 operand: Expression):
        self.operand = operand

    def to_dict(self) -> dict[str, Any]:
        return {"node": NOT, "operand": self.operand.to_dict()}

    def __repr__(self) -> str:
        return f"~{self.operand!r}"


def from_dict(node: dict[str, Any]) -> Expression:
    """Rebuild an expression from `to_dict` output.

    The inverse has to be exact rather than close: a stored definition is
    reloaded and re-run, and a screen that resolves differently after a round
    trip makes a backtest irreproducible in a way nothing would flag.
    """
    if not isinstance(node, dict) or "node" not in node:
        raise ExpressionError(
            f"cannot rebuild an expression from {node!r}: expected a mapping "
            "with a 'node' key.")

    kind = node["node"]

    if kind == FIELD:
        return _field_from(node)

    if kind == COMPARISON:
        return Comparison(_field_from(node["field"]), node["comparison"],
                          node["value"])

    if kind in (ALL, ANY):
        operands = [from_dict(operand) for operand in node["operands"]]

        return All(operands) if kind == ALL else Any_(operands)

    if kind == NOT:
        return Not(from_dict(node["operand"]))

    raise ExpressionError(f"unknown expression node '{kind}'.")


def _field_from(node: dict[str, Any]) -> Field:
    """One field, from its serialised form."""
    if node.get("node") != FIELD:
        raise ExpressionError(f"expected a field node, got {node!r}.")

    return Field(node["namespace"], node["name"], node.get("dataset"))


def distinct_fields_in(expression: Expression) -> list[Field]:
    """Every field an expression mentions, deduplicated, in the order written.

    Deduplicated by `Field.key` rather than by putting the fields in a set:
    two distinct `Field` objects naming the same datapoint have equal hashes,
    so a set would compare them with `__eq__`, get a tree back, and raise.
    """
    seen: set[tuple[str, str | None, str]] = set()
    unique = []

    for field in fields_in(expression):
        if field.key not in seen:
            seen.add(field.key)
            unique.append(field)

    return unique


def fields_in(expression: Expression) -> list[Field]:
    """Every field an expression mentions, in the order written.

    What a caller needs to check coverage before running a screen, or to show
    which datapoints a saved definition depends on.
    """
    if isinstance(expression, Field):
        return [expression]

    if isinstance(expression, Comparison):
        return [expression.field]

    if isinstance(expression, _Group):
        return [field for operand in expression.operands
                for field in fields_in(operand)]

    if isinstance(expression, Not):
        return fields_in(expression.operand)

    return []
