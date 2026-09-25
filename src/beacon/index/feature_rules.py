# src/beacon/index/feature_rules.py
"""
Eligibility rules that screen on features.

A feature is any per-instrument datapoint that is not price, reference or
action data (`beacon.data.features`), so this is the rule that lets an index
select on a fundamental, an alternative dataset, or a value somebody derived
and imported, without a new rule class per datapoint.

## Resolved at the rebalance date, through the point-in-time accessor

An index rebalancing on 1 April screens on what was **published** by 1 April.
Q1 revenue announced in mid-May is invisible, however completely the quarter
had ended. The rule reads through `DataFetcher.fetch_feature`, which enforces
this; reading the table directly would put look-ahead back in, and the
resulting backtest would look better and be wrong. A value older than
`max_age_days` at the rebalance counts as missing.

## Missing coverage is a decision, not an accident

A name with no value for the field is **excluded** by default.

The alternative (including it) means a screen for "revenue above a billion"
silently admits every company the dataset has never heard of, which is the
opposite of what the screen says. Excluding can be wrong too: a universe with
patchy coverage shrinks to the names the vendor happened to cover. So the
behaviour is a parameter, `on_missing`, and the default is the one whose
failure is visible: an index that comes out too small prompts a question,
where one quietly containing uncovered names does not.

This is deliberately *not* the same as a name whose value is legitimately
zero. Zero fails a `> 0` test honestly; missing has no value to compare at
all.
"""
import logging

import pandas as pd

from ..asset.base import Asset
from ..catalogue import SELECTION, Display, register
from ..data.features import MAX_AGE_DAYS
from ..data.fetcher import DataFetcher
from ..exceptions import InvalidRuleError
from .context import IndexContext
from .methodology import EligibilityRuleBase

logger = logging.getLogger(__name__)

# How a value is compared against the threshold. Spelled rather than
# symbolic, because these are serialised into a stored document and read back
# by a client -- ">=" survives JSON, a Python operator does not.
COMPARISONS = {
    "gt": lambda value, threshold: value > threshold,
    "ge": lambda value, threshold: value >= threshold,
    "lt": lambda value, threshold: value < threshold,
    "le": lambda value, threshold: value <= threshold,
    "eq": lambda value, threshold: value == threshold,
    "ne": lambda value, threshold: value != threshold,
}

# What to do with a name the dataset does not cover.
EXCLUDE = "exclude"
INCLUDE = "include"
ON_MISSING = (EXCLUDE, INCLUDE)


@register(SELECTION, "Feature threshold",
          fields={
              "field": Display("Feature", order=1,
                               help="The datapoint, e.g. revenue."),
              "comparison": Display("Comparison", order=2,
                                    choices=tuple(COMPARISONS),
                                    help="How the value is tested."),
              "threshold": Display("Threshold", order=3),
              "feature_type": Display("Dataset", order=4,
                                      help="Which feature set to read from. "
                                           "Blank searches all, which picks "
                                           "arbitrarily between two carrying "
                                           "the same field name."),
              "on_missing": Display("Names without a value", order=5,
                                    choices=ON_MISSING,
                                    help="Excluded by default: a screen that "
                                         "silently admits uncovered names is "
                                         "not the screen it claims to be."),
          })
class FeatureRule(EligibilityRuleBase):
    """Select instruments whose feature value passes a threshold.

    Args:
        field: The feature to read, e.g. ``"revenue"``.
        comparison: How the value is tested against *threshold*: ``"gt"``,
            ``"ge"``, ``"lt"``, ``"le"``, ``"eq"`` or ``"ne"``.
        threshold: The value compared against.
        feature_type: Which feature dataset (`TYPE`) to read from. None
            searches all, which picks arbitrarily between two datasets
            carrying the same field name.
        on_missing: ``"exclude"`` (the default) or ``"include"``: what happens
            to a name with no value knowable at the rebalance.
        max_age_days: How old a value may be and still count. None means no
            limit.

    Raises:
        InvalidRuleError: If *comparison* or *on_missing* is not recognised,
            or *field* is empty.
    """

    def __init__(self,
                 field: str,
                 comparison: str = "gt",
                 threshold: float = 0.0,
                 feature_type: str | None = None,
                 on_missing: str = EXCLUDE,
                 max_age_days: int | None = MAX_AGE_DAYS):
        super().__init__(rule_name="FeatureRule")

        if comparison not in COMPARISONS:
            raise InvalidRuleError(
                f"FeatureRule comparison '{comparison}'",
                f"expected one of {', '.join(sorted(COMPARISONS))}")

        if on_missing not in ON_MISSING:
            raise InvalidRuleError(
                f"FeatureRule on_missing '{on_missing}'",
                f"expected one of {', '.join(ON_MISSING)}")

        if not field:
            raise InvalidRuleError("FeatureRule field",
                                   "a rule must name the datapoint it screens on")

        self.field = field
        self.comparison = comparison
        self.threshold = threshold
        self.feature_type = feature_type
        self.on_missing = on_missing
        self.max_age_days = max_age_days

    def required_columns(self) -> frozenset[str]:
        """No market columns: a feature is read from the features table.

        Stated rather than inherited, so the absence is visibly a decision.
        Whether the named feature exists is a real question with the same
        shape as a missing column, but it is asked of a different table, and
        the up-front column check covers market data only.
        """
        # The up-front column check is BN-217 (`beacon.index.requirements`).
        return frozenset()

    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: IndexContext | None = None) -> bool:
        """Whether the asset passes, as of `current_date`.

        The date is the rebalance date, and it is passed straight through to
        the point-in-time accessor. A value published after it is invisible.
        """
        value = market_data_provider.fetch_feature(
            asset.asset_id, self.field, current_date,
            self.feature_type, self.max_age_days)

        if value is None:
            logger.debug("FeatureRule: %s has no %s knowable on %s; %sd.",
                         asset.asset_id, self.field,
                         current_date.strftime("%Y-%m-%d"), self.on_missing)

            return self.on_missing == INCLUDE

        return bool(COMPARISONS[self.comparison](value, self.threshold))

    def __repr__(self) -> str:
        scope = f", type={self.feature_type!r}" if self.feature_type else ""

        return (f"FeatureRule({self.field!r} {self.comparison} "
                f"{self.threshold}{scope})")
