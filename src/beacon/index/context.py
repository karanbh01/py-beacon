# src/beacon/index/context.py
"""What a methodology rule knows about the index it is running inside."""
from dataclasses import dataclass


# Rules and schemes always took a `context` argument, documented as
# "additional context from the index or global settings", but the calculator
# never passed one until BN-188, so nothing inside an index could see anything
# about it. That was invisible until a market-cap weighting needed the index
# currency and, not having it, weighted a yen cap and a dollar cap as the same
# money.
#
# A frozen dataclass rather than the `dict[str, Any]` the parameter used to be
# annotated as. The dict would have cost no signature churn, but under mypy
# strict every read comes back `Any`, a misspelled key is a silent `None`, and
# the next setting would be added by convention rather than declared. Add the
# next such setting here, one named field at a time.
@dataclass(frozen=True)
class IndexContext:
    """The index's own settings, handed to its rules and weighting scheme.

    The calculator passes one to every eligibility rule and weighting scheme
    as their `context` argument, so a rule can see what it needs about the
    index it runs inside. For example, a market-cap weighting needs the index
    currency to compare a yen cap with a dollar one.

    Attributes:
        currency: The currency the index reports in, upper-cased. Money
            amounts in a methodology (a market-cap bound, the caps a
            weighting compares) are denominated in it.
    """
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", self.currency.upper())
