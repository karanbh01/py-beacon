# src/beacon/server/backtest_settings.py
"""
A backtest request's settings, built into the library's objects, and read
back out fully resolved.

Each part of the request (strategy, implementation, flows, vehicle and
modelling assumptions) is built from its `{type, params}` objects through the
catalogue, so the server accepts exactly the types `GET /backtest/options`
lists. Every problem is collected as a finding with the path of the field
that caused it, such as ``vehicle.settings.pricing.params.threshold``, and a
request with any is refused whole.

## A vehicle's settings over its preset

A vehicle is a preset (`generic` when none is named) with settings laid over
it: a typed object (`pricing`) or a list (`limits`) replaces the preset's
whole, a plain object (`market`) merges key by key, an omitted key keeps the
preset's value, and null means none, refused on a setting that cannot be
none. Settings for exchange-traded vehicles are refused on any other.
"""
# BN-272, phase 9 of decisions/0006. Wire shapes agreed with beacon-ui on
# 2026-10-08 (see #285).
import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import catalogue
from ..assumptions import ModellingAssumptions
from ..backtest.capacity import MinimumPosition
from ..backtest.costs import ExecutionLimit, MarketImpact
from ..backtest.dividends import DIVIDEND_POLICIES
from ..backtest.etf import EtfMarket, EtfVehicle
from ..backtest.flows import Flows
from ..backtest.implementation import Implementation
from ..backtest.presets import PRESETS
from ..backtest.vehicle import Vehicle
from ..data.fetcher import DataFetcher
from ..exceptions import BeaconError
from ..expressions.core import from_dict
from ..index.derived import AnyIndexDefinition
from ..optimise.config import constraint_from_payload
from ..strategy.active import FREQUENCIES, ActiveStrategy, MaxAlpha
from ..strategy.tracking import FullReplication, IndexTracking
from .schemas import BacktestRequest, Finding, TypedSpec, VehicleSpec

# The vehicle used when no preset is named, and the archetypes.
GENERIC = "generic"
GENERIC_NAME = "Generic vehicle"
OPEN_ENDED = "open_ended"
EXCHANGE_TRADED = "exchange_traded"

# Strategy types on the wire.
INDEX = "index"
INDEX_TRACKING = "index_tracking"
ACTIVE = "active"
STRATEGY_TYPES = (INDEX, INDEX_TRACKING, ACTIVE)

# A vehicle's settings, the exchange-traded ones among them, and the one that
# may be none.
COMMON_SETTINGS = ("name", "management_fee_bps", "launch_price", "pricing", "limits")
ETF_SETTINGS = ("creation_unit", "in_kind", "ap_fee", "creation_fee_bps", "market")
NULLABLE_SETTINGS = ("creation_fee_bps",)

# A structured parameter, by class, and how it is read off the wire.
STRUCTURED: dict[tuple[str, str], Callable[[Any], Any]] = {
    ("FieldSignal", "field"): from_dict,
    ("ExpressionScreen", "expression"): from_dict,
    ("OptimisedReplication", "constraints"): lambda rows: [constraint_from_payload(row)
                                                           for row in rows],
}

Strategy = AnyIndexDefinition | IndexTracking | ActiveStrategy


@dataclass
class Findings:
    """The problems found in a request, each at its field's path."""
    found: list[Finding] = field(default_factory=list)

    def add(self,
            path: str,
            code: str,
            message: str) -> None:
        self.found.append(Finding(path=path, severity="error", code=code,
                                  message=message))

    def __bool__(self) -> bool:
        return bool(self.found)


@dataclass
class BuiltSettings:
    """A request's settings as the library's objects.

    Attributes:
        strategy: Builds the strategy for the path's index definition.
        resolved: The settings, fully resolved, as the run reports them.
    """
    currency: str | None
    dividends: str
    modelling_assumptions: ModellingAssumptions | None
    strategy: Callable[[AnyIndexDefinition], Strategy]
    implementation: Implementation | None
    flows: list[Flows]
    vehicle: Vehicle | None
    resolved: dict[str, Any]

    @property
    def strategy_type(self) -> str:
        return str(self.resolved["strategy"]["type"])


def build(request: BacktestRequest,
          fetcher: DataFetcher | None = None) -> tuple[BuiltSettings | None, list[Finding]]:
    """Build a request's settings, or say everything wrong with them.

    Args:
        request: The backtest request.
        fetcher: The data the run would read, to resolve the modelling
            assumptions' data treatment; the library defaults when None.

    Returns:
        tuple: The built settings, or None when any finding was made, and
        the findings.
    """
    findings = Findings()

    if request.dividends not in DIVIDEND_POLICIES:
        findings.add("dividends", "invalid_value",
                     f"Unknown dividend policy '{request.dividends}'. "
                     f"Known: {', '.join(DIVIDEND_POLICIES)}.")

    assumptions = _assumptions(request, findings)
    strategy, strategy_settings = _strategy(request, findings)
    implementation = _implementation(request, findings)
    flows = [flow for position, spec in enumerate(request.flows)
             if (flow := typed(catalogue.FLOW, spec, f"flows.{position}", findings))
             is not None]
    vehicle, preset = _vehicle(request.vehicle, findings)

    if findings:
        return None, findings.found

    currency = request.currency.upper() if request.currency else None
    resolved: dict[str, Any] = {
        "currency": currency,
        "dividends": request.dividends,
        "modelling_assumptions": _resolved_assumptions(assumptions, fetcher),
        "strategy": strategy_settings,
        "implementation": (request.implementation.model_dump()
                           if request.implementation is not None else None),
        "flows": [spec.model_dump() for spec in request.flows],
        "vehicle": ({"preset": preset, "settings": vehicle_settings(vehicle)}
                    if vehicle is not None else None),
    }

    return BuiltSettings(currency=currency, dividends=request.dividends,
                         modelling_assumptions=assumptions, strategy=strategy,
                         implementation=implementation, flows=flows,
                         vehicle=vehicle, resolved=resolved), []


def typed(kind: str,
          spec: TypedSpec,
          path: str,
          findings: Findings) -> Any:
    """One `{type, params}` object, built through the catalogue; None, with
    findings, when it cannot be."""
    cls = catalogue.classes(kind).get(spec.type)

    if cls is None:
        findings.add(f"{path}.type", "unknown_type",
                     f"Unknown {kind.replace('_', ' ')} type '{spec.type}'. Known: "
                     f"{', '.join(sorted(catalogue.registered_names(kind)))}.")
        return None

    entry = catalogue.entry_for(kind, spec.type)
    parameters = {parameter.name: parameter for parameter in entry.parameters} if entry else {}
    problems = len(findings.found)

    for name in sorted(set(spec.params) - set(parameters)):
        findings.add(f"{path}.params.{name}", "unknown_parameter",
                     f"{spec.type} takes no '{name}'. It takes: "
                     f"{', '.join(parameters) or 'nothing'}.")

    for name, parameter in parameters.items():
        if parameter.required and name not in spec.params:
            findings.add(f"{path}.params.{name}", "missing_parameter",
                         f"{spec.type} needs '{name}'.")

    if len(findings.found) > problems:
        return None

    try:
        params = {name: STRUCTURED.get((spec.type, name), _same)(value)
                  for name, value in spec.params.items()}

        return cls(**params)
    except (ValueError, TypeError, BeaconError) as error:
        findings.add(f"{path}.params", "invalid_value", str(error))
        return None


def typed_payload(kind: str,
                  value: Any) -> dict[str, Any]:
    """An object as its `{type, params}` form, read off its attributes."""
    entry = catalogue.entry_for(kind, type(value).__name__)
    names = [parameter.name for parameter in entry.parameters] if entry else []

    return {"type": type(value).__name__,
            "params": {name: getattr(value, name) for name in names
                       if hasattr(value, name)}}


def vehicle_settings(vehicle: Vehicle) -> dict[str, Any]:
    """Every setting of *vehicle*, in the shape `vehicle.settings` takes."""
    settings: dict[str, Any] = {
        "name": vehicle.name,
        "management_fee_bps": vehicle.management_fee_bps,
        "launch_price": vehicle.launch_price,
        "pricing": typed_payload(catalogue.PRICING, vehicle.pricing),
        "limits": [typed_payload(catalogue.LIMIT, limit) for limit in vehicle.limits],
    }

    if isinstance(vehicle, EtfVehicle):
        settings.update(creation_unit=vehicle.creation_unit, in_kind=vehicle.in_kind,
                        ap_fee=vehicle.ap_fee, creation_fee_bps=vehicle.creation_fee_bps,
                        market=market_settings(vehicle.market))

    return settings


def market_settings(market: EtfMarket) -> dict[str, Any]:
    """An ETF market model's settings, by name."""
    return {parameter.name: getattr(market, parameter.name)
            for parameter in catalogue.parameters_of(EtfMarket)}


def preset_vehicle(key: str) -> Vehicle:
    """The vehicle a preset key names; the generic one for `generic`."""
    if key == GENERIC:
        return Vehicle(name=GENERIC_NAME)

    return PRESETS[key].build()


def archetype_of(key: str) -> str:
    """Which archetype a preset belongs to."""
    if key == GENERIC:
        return GENERIC

    return EXCHANGE_TRADED if isinstance(preset_vehicle(key), EtfVehicle) else OPEN_ENDED


def preset_keys() -> list[str]:
    """Every preset key, the generic vehicle's first."""
    return [GENERIC, *PRESETS]


def _vehicle(spec: VehicleSpec | None,
             findings: Findings) -> tuple[Vehicle | None, str | None]:
    """The vehicle: a preset with the settings laid over it."""
    if spec is None:
        return None, None

    key = spec.preset or GENERIC

    if key not in preset_keys():
        findings.add("vehicle.preset", "unknown_preset",
                     f"Unknown preset '{key}'. Known: {', '.join(preset_keys())}.")
        return None, None

    exchange_traded = archetype_of(key) == EXCHANGE_TRADED
    allowed = COMMON_SETTINGS + (ETF_SETTINGS if exchange_traded else ())

    for name, value in spec.settings.items():
        path = f"vehicle.settings.{name}"

        if name in ETF_SETTINGS and not exchange_traded:
            findings.add(path, "not_applicable",
                         f"'{name}' applies only to exchange-traded vehicles.")
        elif name not in allowed:
            findings.add(path, "unknown_setting",
                         f"No vehicle setting '{name}'. Settings: {', '.join(allowed)}.")
        elif value is None and name not in NULLABLE_SETTINGS:
            findings.add(path, "not_nullable", f"'{name}' cannot be none.")

    if findings:
        return None, key

    merged = _merged(vehicle_settings(preset_vehicle(key)), spec.settings)

    return _built_vehicle(merged, spec.settings, exchange_traded, findings), key


def _merged(base: dict[str, Any],
            changes: dict[str, Any]) -> dict[str, Any]:
    """*changes* over *base*: plain objects key by key, all else whole."""
    merged = copy.deepcopy(base)

    for name, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(name), dict) \
                and "type" not in value:
            merged[name] = {**merged[name], **value}
        else:
            merged[name] = value

    return merged


def _built_vehicle(settings: dict[str, Any],
                   changes: dict[str, Any],
                   exchange_traded: bool,
                   findings: Findings) -> Vehicle | None:
    """The vehicle from its merged settings."""
    pricing = _typed_setting(catalogue.PRICING, settings["pricing"],
                             "vehicle.settings.pricing", findings)
    limits = [_typed_setting(catalogue.LIMIT, limit, f"vehicle.settings.limits.{position}",
                             findings)
              for position, limit in enumerate(settings["limits"] or [])]

    if findings:
        return None

    common = {"name": settings["name"], "management_fee_bps": settings["management_fee_bps"],
              "launch_price": settings["launch_price"], "limits": limits}

    try:
        if not exchange_traded:
            return Vehicle(pricing=pricing, **common)

        # An ETF's pricing follows how it creates unless set: changing
        # `in_kind` alone moves it between a creation fee and none.
        if "pricing" in changes:
            common["pricing"] = pricing

        return EtfVehicle(creation_unit=settings["creation_unit"],
                          in_kind=settings["in_kind"], ap_fee=settings["ap_fee"],
                          creation_fee_bps=settings["creation_fee_bps"],
                          market=EtfMarket(**settings["market"]), **common)
    except (ValueError, TypeError) as error:
        findings.add("vehicle.settings", "invalid_value", str(error))
        return None


def _typed_setting(kind: str,
                   value: Any,
                   path: str,
                   findings: Findings) -> Any:
    """A typed setting from its `{type, params}` form."""
    try:
        spec = TypedSpec.model_validate(value)
    except ValueError as error:
        findings.add(path, "invalid_value", f"Expected {{type, params}}: {error}")
        return None

    return typed(kind, spec, path, findings)


def _strategy(request: BacktestRequest,
              findings: Findings) -> tuple[Callable[[AnyIndexDefinition], Strategy],
                                           dict[str, Any]]:
    """The strategy builder and its resolved settings."""
    spec = request.strategy

    if spec is None or spec.type == INDEX:
        return (lambda definition: definition), {"type": INDEX}

    if spec.type == INDEX_TRACKING:
        replication = (typed(catalogue.REPLICATION, spec.replication,
                             "strategy.replication", findings)
                       if spec.replication is not None else FullReplication())

        return ((lambda definition: IndexTracking(definition, replication)),
                {"type": INDEX_TRACKING,
                 "replication": typed_payload(catalogue.REPLICATION, replication)})

    if spec.type != ACTIVE:
        findings.add("strategy.type", "unknown_type",
                     f"Unknown strategy '{spec.type}'. Known: {', '.join(STRATEGY_TYPES)}.")
        return (lambda definition: definition), {"type": spec.type}

    return _active(spec, findings)


def _active(spec: Any,
            findings: Findings) -> tuple[Callable[[AnyIndexDefinition], Strategy],
                                         dict[str, Any]]:
    """An active strategy's builder and resolved settings."""
    if spec.signal is None:
        findings.add("strategy.signal", "missing_parameter",
                     "An active strategy needs a signal.")

    if spec.rebalancing not in FREQUENCIES:
        findings.add("strategy.rebalancing", "invalid_value",
                     f"Unknown rebalancing '{spec.rebalancing}'. Known: "
                     f"{', '.join(FREQUENCIES)}.")

    signal = (typed(catalogue.SIGNAL, spec.signal, "strategy.signal", findings)
              if spec.signal is not None else None)
    construction = (typed(catalogue.CONSTRUCTION, spec.construction,
                          "strategy.construction", findings)
                    if spec.construction is not None else MaxAlpha())
    constraints = [typed(catalogue.ACTIVE_CONSTRAINT, constraint,
                         f"strategy.constraints.{position}", findings)
                   for position, constraint in enumerate(spec.constraints)]
    screen = None

    if spec.screen is not None:
        try:
            screen = from_dict(spec.screen)
        except BeaconError as error:
            findings.add("strategy.screen", "invalid_value", str(error))

    def built(definition: AnyIndexDefinition) -> Strategy:
        # Only called once the request has no findings, so the signal exists.
        assert signal is not None

        windows = {name: value for name, value in
                   (("lookback_days", spec.lookback_days),
                    ("minimum_observations", spec.minimum_observations))
                   if value is not None}

        return ActiveStrategy(benchmark=definition, signal=signal,
                              construction=construction, constraints=constraints,
                              universe=spec.universe, screen=screen,
                              rebalancing=spec.rebalancing, **windows)

    resolved = {**spec.model_dump(),
                "construction": (typed_payload(catalogue.CONSTRUCTION, construction)
                                 if construction is not None else None)}

    return built, resolved


def _implementation(request: BacktestRequest,
                    findings: Findings) -> Implementation | None:
    """The implementation, from its spec."""
    spec = request.implementation

    if spec is None:
        return None

    screens = [typed(catalogue.SCREEN, screen, f"implementation.screens.{position}",
                     findings) for position, screen in enumerate(spec.screens)]
    caps = [typed(catalogue.CAP, cap, f"implementation.caps.{position}", findings)
            for position, cap in enumerate(spec.caps)]
    parts: dict[str, Any] = {}

    for name, cls in (("minimum_position", MinimumPosition), ("impact", MarketImpact),
                      ("execution", ExecutionLimit)):
        settings = getattr(spec, name)

        if settings is None:
            continue

        try:
            parts[name] = cls(**settings)
        except (ValueError, TypeError) as error:
            findings.add(f"implementation.{name}", "invalid_value", str(error))

    if findings:
        return None

    try:
        return Implementation(screens=screens, caps=caps, redistribution=spec.redistribution,
                              cash_buffer=spec.cash_buffer, invest_flows=spec.invest_flows,
                              **parts)
    except ValueError as error:
        findings.add("implementation", "invalid_value", str(error))
        return None


def _assumptions(request: BacktestRequest,
                 findings: Findings) -> ModellingAssumptions | None:
    """The run's modelling assumptions, from their spec."""
    if request.modelling_assumptions is None:
        return None

    try:
        return ModellingAssumptions(**request.modelling_assumptions.model_dump(
            exclude_none=True))
    except ValueError as error:
        findings.add("modelling_assumptions", "invalid_value", str(error))
        return None


def _resolved_assumptions(assumptions: ModellingAssumptions | None,
                          fetcher: DataFetcher | None) -> dict[str, Any]:
    """The assumptions the run would use, every field filled in."""
    effective = (assumptions or ModellingAssumptions()).effective()
    resolved = (effective.resolved(fetcher) if fetcher is not None
                else effective.with_defaults())

    return resolved.as_dict()


def _same(value: Any) -> Any:
    return value
