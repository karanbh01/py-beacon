# src/beacon/server/stage_descriptions.py
"""
What each stage of a strategy's workflow is and does, as a reader sees it.

Each stage of a constituent preview or a holdings workflow carries its type,
one plain sentence of what it does with its settings, and its settings as
labelled values formatted with their units: money in the index's currency,
fractions as percentages, basis points and days as such. The labels and units
are the catalogue's own, so a stage reads the way its editor does.
"""
# BN-328. Karan's call on 2026-10-19: a column's header explains its stage on
# hover, from the engine rather than reconstructed by the app.
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import catalogue
from ..catalogue import BPS, COUNT, DAYS, FRACTION, MONEY
from .types import specs_for

# How a comparison reads in a sentence.
COMPARISONS = {"gt": "above", "ge": "at least", "lt": "below", "le": "at most",
               "eq": "equal to", "ne": "other than"}


@dataclass(frozen=True)
class Described:
    """A stage as a reader sees it.

    Attributes:
        type: Its class, or the kind for the steps that have none.
        description: One plain sentence of what it does with its settings.
        params: Its settings, each ``{label, value}`` formatted for reading.
    """
    type: str
    description: str
    params: list[dict[str, str]] = field(default_factory=list)


class Values:
    """A typed object's settings, each with its catalogue label and unit.

    Args:
        kind: Its catalogue family, such as ``catalogue.SCREEN``.
        type_name: Its class name.
        params: The settings given; the rest take their defaults.
        currency: What money is in.
    """

    def __init__(self,
                 kind: str,
                 type_name: str,
                 params: dict[str, Any] | None,
                 currency: str):
        spec = next((spec for spec in specs_for(kind) if spec.name == type_name), None)
        given = params or {}
        self.currency = currency
        self.parameters = spec.parameters if spec is not None else []
        self.values = {parameter.name: given.get(parameter.name, parameter.default)
                       for parameter in self.parameters}
        self.values.update({name: value for name, value in given.items()
                            if name not in self.values})
        self.units = {parameter.name: parameter.unit for parameter in self.parameters}

    def __getitem__(self,
                    name: str) -> Any:
        return self.values.get(name)

    def text(self,
             name: str) -> str:
        """One setting's value as a sentence writes it."""
        return formatted(self.values.get(name), self.units.get(name), self.currency)

    def rows(self) -> list[dict[str, str]]:
        """The settings that are set, labelled, in the editor's order."""
        # A setting left at a false or empty default says nothing a reader needs.
        return [{"label": parameter.label, "value": self.text(parameter.name)}
                for parameter in self.parameters
                if self.values.get(parameter.name) not in (None, False, [], "")]


def formatted(value: Any,
              unit: str | None,
              currency: str) -> str:
    """A setting's value as a reader reads it."""
    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, int | float):
        return _number(float(value), unit, currency)

    if isinstance(value, list | tuple):
        return _listed([str(item) for item in value])

    if isinstance(value, dict):
        return json.dumps(value, separators=(", ", ": "))

    return str(value)


def _listed(items: list[str]) -> str:
    """Items as a sentence lists them: "A, B and C"."""
    if len(items) < 2:
        return items[0] if items else ""

    return f"{', '.join(items[:-1])} and {items[-1]}"


def _number(value: float,
            unit: str | None,
            currency: str) -> str:
    """A number with its unit."""
    if unit == FRACTION:
        return f"{value * 100:,.4g}%"

    if unit == BPS:
        return f"{value:,.4g} bps"

    if unit == DAYS:
        return f"{value:,.0f} day{'s' if value != 1 else ''}"

    plain = f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"

    if unit == MONEY:
        return f"{plain} {currency}"

    return plain if unit == COUNT or value.is_integer() else f"{value:,.4g}"


def rule(type_name: str,
         params: dict[str, Any] | None,
         currency: str) -> Described:
    """An index's selection rule."""
    values = Values(catalogue.SELECTION, type_name, params, currency)
    sentence = _RULES.get(type_name, _generic("Keeps names that pass this rule"))(values)

    return Described(type=type_name, description=sentence, params=values.rows())


def screen(type_name: str,
           params: dict[str, Any] | None,
           currency: str) -> Described:
    """An implementation's screen."""
    values = Values(catalogue.SCREEN, type_name, params, currency)
    sentence = _SCREENS.get(type_name, _generic("Drops names that fail this screen"))(values)

    return Described(type=type_name, description=sentence, params=values.rows())


def stale_price() -> Described:
    """The stale-price screen every run applies."""
    return Described(type="StalePrice",
                     description="Drops names with no fresh price on the rebalance date.")


def universe(universe_id: str | None,
             name: str | None,
             count: int) -> Described:
    """The universe a rebalance starts from."""
    params = [{"label": "Universe", "value": name or universe_id or "Listed in the index"}]

    if universe_id is not None:
        params.append({"label": "Universe id", "value": universe_id})

    where = f"the {name or universe_id} universe" if (name or universe_id) else "its list"

    return Described(type="Universe", params=params,
                     description=f"Starts from the {count:,} names in {where}.")


def index_cap(max_weight: float) -> Described:
    """An index's cap on any one weight."""
    cap = formatted(max_weight, FRACTION, "")

    return Described(type="IndexCap",
                     description=f"Caps each name at {cap} of the index and spreads the "
                                 f"excess across the rest.",
                     params=[{"label": "Maximum weight", "value": cap}])


def strategy(settings: dict[str, Any],
             currency: str) -> Described:
    """What a tracking or active strategy chose."""
    if settings.get("type") == "index_tracking":
        replication = settings.get("replication") or {}
        values = Values(catalogue.REPLICATION, replication.get("type", ""),
                        replication.get("params"), currency)
        kind = replication.get("type", "FullReplication")

        return Described(type=kind, params=values.rows(),
                         description=_REPLICATIONS.get(kind, _generic(
                             "Holds the index through a replication"))(values))

    construction = settings.get("construction") or {}
    values = Values(catalogue.CONSTRUCTION, construction.get("type", ""),
                    construction.get("params"), currency)
    spec = _spec(catalogue.CONSTRUCTION, construction.get("type", ""))
    method = spec.label.lower() if spec is not None else "its own construction"

    return Described(type=construction.get("type", "ActiveStrategy"), params=values.rows(),
                     description=_sentence(f"Builds its own weights by {method}, measured "
                                           f"against the index"))


def cash_buffer(share: float) -> Described:
    """An implementation's cash buffer."""
    text = formatted(share, FRACTION, "")

    return Described(type="CashBuffer",
                     description=f"Keeps {text} of the book in cash and scales every other "
                                 f"weight down to match.",
                     params=[{"label": "Cash buffer", "value": text}])


def capacity(settings: dict[str, Any],
             currency: str) -> Described:
    """An implementation's caps and minimum position, as one stage."""
    limits: list[str] = []
    params: list[dict[str, str]] = []

    for cap in settings.get("caps") or []:
        values = Values(catalogue.CAP, cap.get("type", ""), cap.get("params"), currency)
        limits.append(_CAPS.get(cap.get("type", ""), _generic("its capacity"))(values))
        params += values.rows()

    minimum = settings.get("minimum_position") or {}

    for name, unit, label in (("value", MONEY, "Smallest position"),
                              ("weight", FRACTION, "Smallest weight")):
        if minimum.get(name) is not None:
            text = formatted(minimum[name], unit, currency)
            params.append({"label": label, "value": text})
            limits.append(f"no position smaller than {text}")

    cuts = [limit for limit in limits if not limit.startswith("no position")]
    drops = [limit for limit in limits if limit.startswith("no position")]
    clauses = ([f"cuts each position to {' and to '.join(cuts)}"] if cuts else []) + [
        f"drops any position smaller than {drop.removeprefix('no position smaller than ')}"
        for drop in drops]
    clauses = (clauses or ["holds each position within the vehicle's limits"]) + [
        "spreads what it moves across the rest"]
    sentence = (" and ".join(clauses) if len(clauses) == 2
                else f"{', '.join(clauses[:-1])}, and {clauses[-1]}")

    return Described(type="CapacityCaps", params=params, description=_sentence(sentence))


def _spec(kind: str,
          type_name: str) -> Any:
    return next((spec for spec in specs_for(kind) if spec.name == type_name), None)


def _sentence(text: str) -> str:
    """Text as one sentence: a capital first and a full stop last."""
    text = text.strip().rstrip(".")

    return f"{text[:1].upper()}{text[1:]}." if text else ""


def _generic(text: str) -> Callable[[Values], str]:
    return lambda values: _sentence(text)


def _market_cap_rule(values: Values) -> str:
    low, high = values["min_market_cap"], values["max_market_cap"]

    if low is not None and high is not None:
        return _sentence(f"Keeps names whose market cap is between "
                         f"{values.text('min_market_cap')} and {values.text('max_market_cap')}")

    if high is not None:
        return _sentence(f"Keeps names whose market cap is at most "
                         f"{values.text('max_market_cap')}")

    if low is not None:
        return _sentence(f"Keeps names whose market cap is at least "
                         f"{values.text('min_market_cap')}")

    return _sentence("Keeps every name with a market cap")


def _liquidity_rule(values: Values) -> str:
    floors = [f"{values.text('min_avg_daily_volume')} shares"
              if values["min_avg_daily_volume"] is not None else None,
              f"{values.text('min_avg_daily_value')}"
              if values["min_avg_daily_value"] is not None else None]
    floor = " and ".join(text for text in floors if text) or "anything"

    return _sentence(f"Keeps names that traded at least {floor} a day on average over the "
                     f"last {values.text('lookback_days')}")


def _feature_rule(values: Values) -> str:
    comparison = COMPARISONS.get(str(values["comparison"]), str(values["comparison"]))

    return _sentence(f"Keeps names whose {values['field']} is {comparison} "
                     f"{values.text('threshold')}")


def _exclusion(values: Values) -> str:
    parts = [f"{values.text('identifiers')}" if values["identifiers"] else None,
             f"any name in {values.text('sectors')}" if values["sectors"] else None,
             f"any name in {values.text('regions')}" if values["regions"] else None]
    named = [part for part in parts if part]

    return _sentence(f"Drops {', and '.join(named)}" if named else "Drops nothing")


def _held_exit(values: Values,
               name: str) -> str:
    """The clause for a held name's lower bar, when it has one."""
    return (f", though a name already held stays until it falls below {values.text(name)}"
            if values[name] is not None else "")


def _liquidity_screen(values: Values) -> str:
    # Its exit is in the floor's unit: money for traded value, else shares.
    values.units["exit"] = MONEY if values["min_traded_value"] is not None else COUNT

    floors = [values.text("min_traded_value") if values["min_traded_value"] is not None
              else None,
              f"{values.text('min_volume')} shares" if values["min_volume"] is not None
              else None]
    floor = " or ".join(text for text in floors if text) or "nothing"

    return _sentence(f"Drops names that traded less than {floor} a day on average over the "
                     f"last {values.text('lookback_days')}{_held_exit(values, 'exit')}")


def _market_cap_screen(values: Values) -> str:
    kind = "free-float market cap" if values["float_adjusted"] else "market cap"

    return _sentence(f"Drops names whose {kind} is below {values.text('min_cap')}"
                     f"{_held_exit(values, 'exit')}")


_RULES: dict[str, Callable[[Values], str]] = {
    "MarketCapRule": _market_cap_rule,
    "LiquidityRule": _liquidity_rule,
    "FeatureRule": _feature_rule,
    "ExpressionRule": _generic("Keeps names for which its expression holds"),
}
_SCREENS: dict[str, Callable[[Values], str]] = {
    "ExclusionScreen": _exclusion,
    "ExpressionScreen": _generic("Keeps names for which its expression holds"),
    "LiquidityScreen": _liquidity_screen,
    "ListingAgeScreen": lambda values: _sentence(
        f"Drops names with less than {values.text('min_days')} of prices"),
    "MarketCapScreen": _market_cap_screen,
    "MinimumPriceScreen": lambda values: _sentence(
        f"Drops names priced below {values.text('min_price')}{_held_exit(values, 'exit')}"),
}
_CAPS: dict[str, Callable[[Values], str]] = {
    "WeightCap": lambda values: f"at most {values.text('max_weight')} of the book",
    "LiquidityCap": lambda values: (f"what could be sold in {values.text('days')} at "
                                    f"{values.text('participation')} of the day's volume"),
    "OwnershipCap": lambda values: (f"at most {values.text('max_share')} of the company's "
                                    f"{'free-float ' if values['free_float'] else ''}"
                                    f"market cap"),
}
_REPLICATIONS: dict[str, Callable[[Values], str]] = {
    "FullReplication": _generic("Holds every constituent at its index weight"),
    "SampledReplication": lambda values: _sentence(
        f"Holds the largest names in each sector and size cell, {values.text('holdings')} "
        f"names in all, each cell at its index weight"),
    "OptimisedReplication": lambda values: _sentence(
        f"Holds {'at most ' + values.text('holdings') if values['holdings'] else 'the'} "
        f"names weighted to track the index as closely as possible"),
    "SyntheticReplication": _generic(
        "Holds a substitute basket and a swap that pays the index's return for the basket's"),
    "FuturesReplication": lambda values: _sentence(
        f"Holds cash and index futures, rolled {values.text('roll_days')} before expiry"),
}
