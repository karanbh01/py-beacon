# src/beacon/index/expression_rules.py
"""
The rule an expression compiles into.

    definition.add_rule(ExpressionRule.from_expression(
        (data.market.market_cap > 1e9)
        & (data.features.fundamentals.pe_ratio < 20)))

## One new rule type, not a replacement

`MarketCapRule`, `LiquidityRule`, `FeatureRule` and the rest keep working.
This sits beside them and stores its tree in `params`, so a definition written
in Python and one built in the client are the **same document** and neither
has to know which produced it:

    {"id": "r1", "type": "ExpressionRule",
     "params": {"expression": {"node": "all", "operands": [...]}}}

An expression that could not serialise could never reach a saved definition,
which is most of what a rule is for.

## Evaluated at the rebalance date

Every read goes through the point-in-time path (`expressions.resolve`). A new
authoring surface is exactly the sort of front door look-ahead walks back in
through: it is easy to write a resolver that reads the latest value because
that is the simpler query, and the resulting backtest looks better and is
wrong.

## Missing coverage is a stated behaviour

A name with no value for a field is **excluded** by default, matching
`FeatureRule` (BN-136) so the two do not disagree about the same situation.

The alternative — including it — means a screen for "revenue above a billion"
silently admits every company the dataset has never heard of, which is the
opposite of what the screen says. Excluding can be wrong too, so it is a
parameter; the default is the one whose failure is visible, since an index
that comes out too small prompts a question where one quietly full of
uncovered names does not.
"""
import logging
from typing import Any

import pandas as pd

from ..asset.base import Asset
from ..catalogue import SELECTION, Display, register
from ..data.features import MAX_AGE_DAYS
from ..data.fetcher import DataFetcher
from ..exceptions import ExpressionError, InvalidRuleError
from ..expressions.core import Expression, from_dict
from ..expressions.resolve import resolve
from .feature_rules import EXCLUDE, INCLUDE, ON_MISSING
from .methodology import EligibilityRuleBase

logger = logging.getLogger(__name__)


@register(SELECTION, "Expression",
          fields={
              "expression": Display("Expression", order=1,
                                    help="The screen, as a serialised "
                                         "expression tree."),
              "on_missing": Display("Names without a value", order=2,
                                    choices=ON_MISSING,
                                    help="Excluded by default: a screen that "
                                         "silently admits uncovered names is "
                                         "not the screen it claims to be."),
          })
class ExpressionRule(EligibilityRuleBase):
    """Select instruments that satisfy an expression."""

    def __init__(self,
                 expression: dict[str, Any],
                 on_missing: str = EXCLUDE,
                 max_age_days: int | None = MAX_AGE_DAYS):
        super().__init__(rule_name="ExpressionRule")

        if on_missing not in ON_MISSING:
            raise InvalidRuleError(
                f"ExpressionRule on_missing '{on_missing}'",
                f"expected one of {', '.join(ON_MISSING)}")

        # Rebuilt eagerly rather than at the first rebalance. A malformed tree
        # is a fact about the rule, and finding out at construction is the
        # difference between a rejected save and a run that dies partway
        # through with thousands of names already priced.
        try:
            self._tree = from_dict(expression)
        except ExpressionError as error:
            raise InvalidRuleError("ExpressionRule expression",
                                   str(error)) from error

        self.expression = expression
        self.on_missing = on_missing
        self.max_age_days = max_age_days

    @classmethod
    def from_expression(cls,
                        expression: Expression,
                        on_missing: str = EXCLUDE,
                        max_age_days: int | None = MAX_AGE_DAYS
                        ) -> "ExpressionRule":
        """Build from a live expression rather than from its serialised form.

        What a user writing Python calls. The stored `params` are identical
        either way, which is the point: one representation, two front doors.
        """
        return cls(expression.to_dict(), on_missing, max_age_days)

    @property
    def tree(self) -> Expression:
        """The rebuilt expression."""
        return self._tree

    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: dict[str, Any] | None = None) -> bool:
        """Whether the asset passes, as of `current_date`.

        The date is the rebalance date and is passed straight through to the
        point-in-time reads. A value published after it is invisible.
        """
        return resolve(self._tree, asset.asset_id, current_date,
                       market_data_provider,
                       on_missing=self.on_missing == INCLUDE,
                       max_age_days=self.max_age_days)

    def __repr__(self) -> str:
        return f"ExpressionRule({self._tree!r})"
