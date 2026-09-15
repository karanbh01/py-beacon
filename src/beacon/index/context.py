# src/beacon/index/context.py
"""What a methodology rule knows about the index it is running inside."""
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexContext:
    """The index's own settings, handed to its rules and weighting scheme.

    Rules and schemes have always taken a `context` argument, documented as
    "additional context from the index or global settings" — and the
    calculator never passed one, so nothing that ran inside an index could see
    anything about it. That was invisible until a rule needed something: a
    market-cap weighting cannot compare a yen cap with a dollar one without
    knowing which currency to compare them in, and not knowing is what let it
    weight the two as though they were the same money (BN-188).

    A frozen dataclass rather than the `dict[str, Any]` the parameter used to
    be annotated as. The dict was already there and would have cost no
    signature churn, but a mypy-strict codebase gets nothing from it: every
    read comes back `Any`, a misspelled key is a silent `None`, and the next
    thing a rule needs would be added by convention rather than declared.
    This is the parameter the docstring always described, with a type — and it
    is the place to add the next such setting, one named field at a time.

    Attributes:
        currency: The currency the index reports in, upper-cased. What money
            amounts in a methodology — a market-cap bound, the caps a
            weighting compares — are denominated in.
    """
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", self.currency.upper())
