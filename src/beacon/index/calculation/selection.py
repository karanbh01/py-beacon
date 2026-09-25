# src/beacon/index/calculation/selection.py
"""
Constituent selection, and the record of how it happened.

One function, `select_with_provenance`, answers "which assets are eligible",
and it answers it by walking the rules in order and narrowing the universe a
rung at a time, keeping a note of which rule removed each name as it goes.

Before any rule runs, names with no trade within the data source's
`max_price_staleness_days` are dropped at their own rung
(`STALENESS_POSITION`, named `StalePrice`), so no rule evaluates a stale close.

## Why provenance is the general form

Survivors fall out of the provenance for free; provenance cannot be recovered
from a list of survivors. So the calculator and the preview waterfall both use
this one walk, and a preview cannot disagree with the run it is previewing.

The rule-outer loop also leaves room for a rule that ranks (*the largest
hundred by market capitalisation*, *at most ten per sector*), which needs to
see the set it is choosing from. `is_eligible` is a per-asset predicate, so no
rule ranks yet, but this loop structure can accommodate one where an
asset-outer loop could not, because it never has a set in hand.

## Rules are identified by position, not by name

A rule object carries a `rule_name`, which is its *type* (`"MarketCapRule"`),
and an index definition may hold several of the same type. Stable per-rule
identifiers exist only in the server's stored document, which is the server's
concern and not this layer's. So provenance here is keyed by position in the
rule list, and a caller that has its own identifiers maps position to them.
"""
# There used to be two implementations of selection: `IndexCalculator` looped
# assets on the outside and rules on the inside and returned the survivors,
# while the preview waterfall narrowed the set rule by rule and returned the
# survivors plus who removed what. Keeping only the general form is what stops
# the two disagreeing.
import logging
from dataclasses import dataclass, field

import pandas as pd

from ...asset.base import Asset
from ...data.fetcher import DataFetcher
from ..context import IndexContext
from ..methodology import EligibilityRuleBase

logger = logging.getLogger(__name__)

# The rung representing the universe before any rule has been applied.
UNIVERSE_POSITION = 0

# The rung a stale-price exclusion is recorded at (BN-211). Negative so it
# cannot collide with a rule's 1-based position, and so a consumer that maps
# positions onto the definition's rules -- the preview does, by index -- has to
# handle it deliberately rather than silently read the wrong rule's id.
#
# It sits before the rules because it is not one: the threshold is an
# installation-wide setting rather than part of any index's methodology, and a
# name nobody has priced for months is a data condition that every rule after
# it would otherwise evaluate against a stale close.
STALENESS_POSITION = -1

# What the funnel calls that rung, so a reader of the provenance record sees
# a reason rather than an unexplained drop.
STALENESS_RULE_NAME = "StalePrice"


@dataclass(frozen=True)
class SelectionStep:
    """One rung of the selection funnel.

    Attributes:
        position: 1-based index of the rule, UNIVERSE_POSITION (0) for the
            starting universe, or STALENESS_POSITION (-1) for the stale-price
            rung.
        rule_name: Type of the rule applied (``"StalePrice"`` for the
            stale-price rung), empty for the universe rung.
        remaining: How many assets survived this rung.
        excluded: Identifiers this rung removed, sorted. Empty for the
            universe rung.
    """
    position: int
    remaining: int
    rule_name: str = ""
    excluded: list[str] = field(default_factory=list)

    @property
    def is_universe(self) -> bool:
        """Whether this is the starting rung rather than a rule."""
        return self.position == UNIVERSE_POSITION


@dataclass(frozen=True)
class SelectionResult:
    """Which assets survived selection, and how each one fared.

    Attributes:
        survivors: Assets that passed every rule, in universe order.
        steps: One entry per rung, starting with the universe, then the
            stale-price rung when any name was stale, then one per rule.
        exclusions: Identifier to the position of the rule that removed it
            (STALENESS_POSITION for a name dropped as stale).
            Each excluded asset appears exactly once: an asset leaves the
            surviving set the moment it fails, so no later rule ever sees it
            and no name can be blamed on two rules. That single-owner property
            is what makes the funnel answer "why is this name missing" rather
            than only "how many are left".
    """
    survivors: list[Asset]
    steps: list[SelectionStep]
    exclusions: dict[str, int] = field(default_factory=dict)

    @property
    def survivor_ids(self) -> list[str]:
        """Identifiers of the surviving assets."""
        return [asset.asset_id for asset in self.survivors]

    @property
    def rule_steps(self) -> list[SelectionStep]:
        """Every rung after the universe, including any stale-price rung."""
        return [step for step in self.steps if not step.is_universe]

    def excluded_by(self,
                    asset_id: str) -> SelectionStep | None:
        """The rung that removed an asset.

        Args:
            asset_id: The identifier to look up.

        Returns:
            SelectionStep or None: The rung, or None if the asset survived or
            was never in the universe.
        """
        position = self.exclusions.get(asset_id)
        if position is None:
            return None

        # Matched on the recorded position rather than used as a list index:
        # the stale-price rung sits in `steps` at index 1 with position -1, so
        # indexing would attribute every rule's exclusion to the rung before
        # it, and a stale name to the last rule.
        return next((step for step in self.steps if step.position == position),
                    None)


def select_with_provenance(universe: list[Asset],
                           rules: list[EligibilityRuleBase],
                           current_date: pd.Timestamp,
                           data_fetcher: DataFetcher,
                           context: IndexContext | None = None) -> SelectionResult:
    """Narrow a universe to its eligible constituents, recording each step.

    Args:
        universe: Assets to select from.
        rules: Eligibility rules, applied in order. Each rule sees only what
            survived the ones before it.
        current_date: The date to evaluate at.
        data_fetcher: Data source the rules read from.
        context: What the index settles for its rules: its currency, so a
            bound stated in it is compared against a converted figure rather
            than a local one. None outside an index, and then a rule that
            needs it says so rather than assuming one.

    Returns:
        SelectionResult: Survivors, the funnel, and per-asset provenance.

    Raises:
        Exception: Whatever a rule raises, unchanged. A rule that could not
            run has not excluded anything, so its failure propagates rather
            than being recorded as an exclusion.
    """
    surviving = list(universe)
    steps = [SelectionStep(position=UNIVERSE_POSITION, remaining=len(surviving))]
    exclusions: dict[str, int] = {}

    stale = data_fetcher.stale_identifiers(
        [asset.asset_id for asset in surviving], current_date)

    if stale:
        surviving = [asset for asset in surviving
                     if asset.asset_id not in stale]

        for asset_id in stale:
            exclusions[asset_id] = STALENESS_POSITION

        steps.append(SelectionStep(position=STALENESS_POSITION,
                                   rule_name=STALENESS_RULE_NAME,
                                   remaining=len(surviving),
                                   excluded=sorted(stale)))

        logger.info(
            "[%s] %d name(s) dropped: no trade within %d days.",
            current_date.date(), len(stale),
            data_fetcher.max_price_staleness_days)

    for position, rule in enumerate(rules, start=1):
        surviving, removed = _apply_rule(rule, surviving, current_date,
                                        data_fetcher, context)

        for asset_id in removed:
            exclusions[asset_id] = position

        steps.append(SelectionStep(position=position,
                                   rule_name=rule.rule_name,
                                   remaining=len(surviving),
                                   excluded=sorted(removed)))

    for asset in surviving:
        logger.debug(f"Asset {asset.asset_id} passed all eligibility rules.")

    return SelectionResult(survivors=surviving, steps=steps, exclusions=exclusions)


def _apply_rule(rule: EligibilityRuleBase,
                candidates: list[Asset],
                current_date: pd.Timestamp,
                data_fetcher: DataFetcher,
                context: IndexContext | None = None) -> tuple[list[Asset], list[str]]:
    """Split candidates into those that pass a rule and those that do not.

    A rule that raises is left to raise (BN-182). This used to catch every
    exception and record the asset as excluded, which spelled "the rule
    evaluated this name and said no" and "the rule could not run" the same
    way — and once `MarketCapRule` began refusing dates outside the data's
    coverage, that swallow turned each refusal back into an exclusion and
    emptied the universe, which is the very failure the refusal exists to
    stop.

    The rule is shown the whole candidate set once before being asked about any
    of it (BN-190). Nothing it does there can change an answer — `prepare` is
    the rule reading in one slice what it would otherwise read a name at a time
    — so the loop below is unchanged and so is every verdict it records.
    """
    rule.prepare(candidates, current_date, data_fetcher, context)

    kept: list[Asset] = []
    removed: list[str] = []

    for asset in candidates:
        if rule.is_eligible(asset, current_date, data_fetcher, context):
            kept.append(asset)
        else:
            logger.debug(f"Asset {asset.asset_id} failed eligibility rule: {rule.rule_name}")
            removed.append(asset.asset_id)

    return kept, removed
