# src/beacon/index/calculation/selection.py
"""
Constituent selection, and the record of how it happened.

One function answers "which assets are eligible", and it answers it by walking
the rules in order and narrowing the universe a rung at a time — keeping a note
of which rule removed each name as it goes.

## Why provenance is the general form

There used to be two implementations of this. `IndexCalculator` looped assets
on the outside and rules on the inside, asking "does this asset clear every
rule?", and returned the survivors. The preview waterfall looped the other way
round, narrowing the set rule by rule, and returned the survivors *plus* who
removed what.

The two are not symmetric. Survivors fall out of the provenance for free;
provenance cannot be recovered from a list of survivors. So one of these is the
computation and the other is a projection of it, and keeping only the general
form is the only arrangement in which a preview and a real run cannot disagree
— which matters, because a preview that disagrees with the run it is previewing
is worse than no preview.

It also decides something the old calculator shape could not express. A rule
that ranks — *the largest hundred by market capitalisation*, *at most ten per
sector* — needs to see the set it is choosing from. `is_eligible` is a
per-asset predicate today, so the question does not arise; but the loop
structure here can accommodate one and an asset-outer loop structurally cannot,
because it never has a set in hand.

## Rules are identified by position, not by name

A rule object carries a `rule_name`, which is its *type* (`"MarketCapRule"`),
and an index definition may hold several of the same type. Stable per-rule
identifiers exist only in the server's stored document, which is the server's
concern and not this layer's. So provenance here is keyed by position in the
rule list, and a caller that has its own identifiers maps position to them.
"""
import logging
from dataclasses import dataclass, field

import pandas as pd

from ...asset.base import Asset
from ...data.fetcher import DataFetcher
from ..methodology import EligibilityRuleBase

logger = logging.getLogger(__name__)

# The rung representing the universe before any rule has been applied.
UNIVERSE_POSITION = 0


@dataclass(frozen=True)
class SelectionStep:
    """One rung of the selection funnel.

    Attributes:
        position: 1-based index of the rule, or UNIVERSE_POSITION for the
            starting universe.
        rule_name: Type of the rule applied, empty for the universe rung.
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
        steps: One entry per rung, starting with the universe.
        exclusions: Identifier to the position of the rule that removed it.
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
        """The rungs that are rules, excluding the universe."""
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

        return self.steps[position]


def select_with_provenance(universe: list[Asset],
                           rules: list[EligibilityRuleBase],
                           current_date: pd.Timestamp,
                           data_fetcher: DataFetcher) -> SelectionResult:
    """Narrow a universe to its eligible constituents, recording each step.

    Args:
        universe: Assets to select from.
        rules: Eligibility rules, applied in order. Each rule sees only what
            survived the ones before it.
        current_date: The date to evaluate at.
        data_fetcher: Data source the rules read from.

    Returns:
        SelectionResult: Survivors, the funnel, and per-asset provenance.
    """
    surviving = list(universe)
    steps = [SelectionStep(position=UNIVERSE_POSITION, remaining=len(surviving))]
    exclusions: dict[str, int] = {}

    for position, rule in enumerate(rules, start=1):
        surviving, removed = _apply_rule(rule, surviving, current_date, data_fetcher)

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
                data_fetcher: DataFetcher) -> tuple[list[Asset], list[str]]:
    """Split candidates into those that pass a rule and those that do not."""
    kept: list[Asset] = []
    removed: list[str] = []

    for asset in candidates:
        if _is_eligible(rule, asset, current_date, data_fetcher):
            kept.append(asset)
        else:
            removed.append(asset.asset_id)

    return kept, removed


def _is_eligible(rule: EligibilityRuleBase,
                 asset: Asset,
                 current_date: pd.Timestamp,
                 data_fetcher: DataFetcher) -> bool:
    """Apply one rule to one asset, treating a raised error as exclusion.

    A rule that throws has not said the asset is eligible, and defaulting to
    inclusion would put a name into a live index on the strength of a bug. It
    is logged at ERROR because it is a result-affecting failure rather than a
    routine exclusion — the asset would very likely have qualified.
    """
    try:
        if rule.is_eligible(asset, current_date, data_fetcher):
            return True
    except Exception as exc:
        logger.error(
            f"Error applying eligibility rule {rule.rule_name} to asset "
            f"{asset.asset_id}: {exc}")

        return False

    logger.debug(f"Asset {asset.asset_id} failed eligibility rule: {rule.rule_name}")

    return False
