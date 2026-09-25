# src/beacon/index/expression_rules.py
"""
The rule an expression compiles into.

    rule = ExpressionRule.from_expression(
        (data.market.market_cap > 1e9)
        & (data.features.fundamentals.pe_ratio < 20))

    IndexDefinition(..., eligibility_rules=[rule])

## A rule type beside the others

`ExpressionRule` sits beside `MarketCapRule`, `LiquidityRule`, `FeatureRule`
and the rest, and stores its tree in `params`, so a definition written
in Python and one built in the client are the **same document** and neither
has to know which produced it:

    {"id": "r1", "type": "ExpressionRule",
     "params": {"expression": {"node": "all", "operands": [...]}}}

An expression that could not serialise could never reach a saved definition,
which is most of what a rule is for.

## Evaluated at the rebalance date

Every read goes through the point-in-time path (`beacon.expressions.resolve`),
so a value published after the rebalance date is invisible. Reading the latest
value instead would make the backtest look better and be wrong.

## Missing coverage is a stated behaviour

A name with no value for a field is **excluded** by default, matching
`FeatureRule` so the two do not disagree about the same situation.

The alternative (including it) means a screen for "revenue above a billion"
silently admits every company the dataset has never heard of, which is the
opposite of what the screen says. Excluding can be wrong too, so it is a
parameter; the default is the one whose failure is visible, since an index
that comes out too small prompts a question where one quietly full of
uncovered names does not.
"""
# Excluding missing coverage by default matches FeatureRule (BN-136).
import logging
from typing import Any, ClassVar

import pandas as pd

from ..asset.base import Asset
from ..catalogue import SELECTION, Display, register
from ..data.features import MAX_AGE_DAYS
from ..data.fetcher import DataFetcher
from ..exceptions import ExpressionError, InvalidRuleError
from ..expressions.core import Expression, fields_in, from_dict
from ..expressions.namespaces import market_columns_for
from ..expressions.resolve import resolve
from .context import IndexContext
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
    """Select instruments that satisfy an expression.

    Args:
        expression: The serialised expression tree (`Expression.to_dict()`
            output). Use :meth:`from_expression` to pass a live expression.
        on_missing: ``"exclude"`` (the default) or ``"include"``: what a
            comparison answers for a name with no value for its field.
        max_age_days: How old a feature value may be and still count. None
            means no limit.

    Raises:
        InvalidRuleError: If *on_missing* is not recognised, or *expression*
            is not a valid tree.
    """

    # Which published schema each parameter's value conforms to (BN-175).
    #
    # Declared here the way a constraint declares `UNIT`, and read by the
    # catalogue adapter that serves `/indices/rule-types`. `expression` is a
    # tree, and `ParameterSpec.type` can only say `json`: without this a client
    # meeting this rule learns that a parameter called `expression` exists and
    # is not a number, and has to special-case the NAME to render anything
    # better. The schema name is the one the server publishes the grammar
    # under — it is a contract, not an implementation detail, so it is written
    # where the parameter is defined rather than guessed at the edge.
    PARAM_SCHEMAS: ClassVar[dict[str, str]] = {"expression": "ExpressionNode"}

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

    def required_columns(self) -> frozenset[str]:
        """The market columns the expression reads, derived from its tree.

        An expression's needs are whatever it references, so they come from
        the fields in the tree. A derived field is expanded into what it is
        computed from: a screen on `market_cap` needs CLOSE and
        SHARES_OUTSTANDING, not a column called MARKET_CAP that no store has.
        Reference, action and feature fields read other tables and add
        nothing here.
        """
        # The up-front column check is BN-217 (`beacon.index.requirements`).
        return market_columns_for(fields_in(self._tree))

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
                    context: IndexContext | None = None) -> bool:
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
