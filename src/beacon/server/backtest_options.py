# src/beacon/server/backtest_options.py
"""
`GET /backtest/options`: everything a backtest form needs, from the engine.

Every family's types come from the catalogue in the shape
`/indices/rule-types` uses, so one form component renders them all. The
presets carry their resolved settings, the vehicle's and the market model's
settings and the implementation's plain settings are described field by
field (with `applies_to` marking the exchange-traded ones), and the modelling
assumptions are listed with the engine's defaults. Nothing here reads data.
"""
# BN-272, phase 9 of decisions/0006, as agreed with beacon-ui on 2026-10-08.
from typing import Any

from .. import catalogue
from ..assumptions import ModellingAssumptions
from ..backtest.capacity import MinimumPosition
from ..backtest.costs import ExecutionLimit, MarketImpact
from ..backtest.dividends import DIVIDEND_POLICIES
from ..backtest.etf import EtfMarket, EtfVehicle
from ..backtest.implementation import FLOW_INVESTMENTS, REDISTRIBUTIONS
from ..backtest.vehicle import Vehicle
from ..catalogue import BPS, COUNT, DAYS, FRACTION, MONEY, RATIO, Display
from ..data.fetcher import DEFAULT_FX_POLICY, FX_POLICIES
from ..data.free_float import DEFAULT_FREE_FLOAT_BACKFILL_DAYS
from ..strategy.active import FREQUENCIES
from .backtest_settings import (
    EXCHANGE_TRADED,
    STRATEGY_TYPES,
    archetype_of,
    preset_keys,
    preset_vehicle,
    vehicle_settings,
)
from .schemas import BacktestOptions, ParameterSpec, VehiclePreset
from .types import parameter_spec, specs_for

VEHICLE_DISPLAYS = {
    "management_fee_bps": Display("Management fee", unit=BPS, minimum=0.0),
    "launch_price": Display("Launch price per unit", unit=MONEY, minimum=0.0),
    "pricing": Display("Pricing method", help="One of `pricing`, as {type, params}."),
    "limits": Display("Diversification limits", help="Any of `limits`, as {type, params}."),
    "name": Display("Name"),
}

ETF_DISPLAYS = {
    "creation_unit": Display("Shares per creation unit", unit=COUNT, minimum=1),
    "in_kind": Display("Create in kind"),
    "ap_fee": Display("AP fee per creation", unit=MONEY, minimum=0.0),
    "creation_fee_bps": Display("Creation fee", unit=BPS, minimum=0.0,
                                help="Null estimates it from the basket's trading cost."),
    "market": Display("Exchange quotes", help="See `market_settings`."),
}

MARKET_DISPLAYS = {
    "spread_floor_bps": Display("Narrowest spread", unit=BPS, minimum=0.0),
    "basket_sensitivity": Display("Basket cost passed on", unit=RATIO, minimum=0.0),
    "volatility_sensitivity": Display("Spread per unit of volatility", unit=RATIO, minimum=0.0),
    "tick": Display("Price tick", unit=MONEY, minimum=0.0),
    "persistence": Display("Premium persistence", unit=FRACTION, minimum=0.0, maximum=1.0),
    "flow_sensitivity": Display("Premium per unit of creations", unit=RATIO),
    "return_sensitivity": Display("Premium per unit of NAV return", unit=RATIO),
    "noise_bps": Display("Premium noise", unit=BPS, minimum=0.0),
    "volatility_days": Display("Volatility window", unit=DAYS, minimum=2),
    "seed": Display("Random seed"),
}

IMPLEMENTATION_DISPLAYS = {
    "minimum_position.value": Display("Smallest position", unit=MONEY, minimum=0.0),
    "minimum_position.weight": Display("Smallest weight", unit=FRACTION, minimum=0.0,
                                       maximum=1.0),
    "impact.coefficient": Display("Impact coefficient", unit=RATIO, minimum=0.0),
    "impact.lookback_days": Display("Impact window", unit=DAYS, minimum=1),
    "execution.participation": Display("Share of daily volume", unit=FRACTION,
                                       minimum=0.0, maximum=1.0),
    "execution.days": Display("Days per order", unit=DAYS, minimum=1),
    "execution.lookback_days": Display("Average volume window", unit=DAYS, minimum=1),
}

ASSUMPTION_DISPLAYS = {
    "fx_policy": Display("FX on a day with no rate", choices=tuple(FX_POLICIES)),
    "max_price_staleness_days": Display("Days before a price is stale", unit=DAYS,
                                        minimum=0, help="0 is no limit."),
    "free_float_backfill_days": Display("Free float carried for", unit=DAYS, minimum=0),
    "cash_rate": Display("Rate cash earns", unit=FRACTION),
    "risk_free_rate": Display("Risk-free rate", unit=FRACTION),
    "periods_per_year": Display("Periods a year", unit=COUNT, minimum=1),
    "withholding_tax_rate": Display("Dividend withholding", unit=FRACTION, minimum=0.0,
                                    maximum=1.0),
    "volume_backfill_days": Display("Volume carried for", unit=DAYS, minimum=0),
}

FAMILIES = {
    "screens": catalogue.SCREEN, "caps": catalogue.CAP, "flows": catalogue.FLOW,
    "pricing": catalogue.PRICING, "limits": catalogue.LIMIT,
    "replications": catalogue.REPLICATION, "signals": catalogue.SIGNAL,
    "constructions": catalogue.CONSTRUCTION,
    "active_constraints": catalogue.ACTIVE_CONSTRAINT,
}


def backtest_options() -> BacktestOptions:
    """The response of `GET /backtest/options`."""
    families: dict[str, Any] = {name: specs_for(kind) for name, kind in FAMILIES.items()}

    return BacktestOptions(
        **families,
        presets=[VehiclePreset(key=key, name=preset_vehicle(key).name,
                               archetype=archetype_of(key),
                               settings=vehicle_settings(preset_vehicle(key)))
                 for key in preset_keys()],
        vehicle_settings=_vehicle_settings(),
        market_settings=_described(EtfMarket, MARKET_DISPLAYS),
        implementation_settings=_implementation_settings(),
        modelling_assumptions=_assumptions(),
        choices={"dividends": list(DIVIDEND_POLICIES), "strategy": list(STRATEGY_TYPES),
                 "rebalancing": list(FREQUENCIES), "redistribution": list(REDISTRIBUTIONS),
                 "invest_flows": list(FLOW_INVESTMENTS), "fx_policy": list(FX_POLICIES)})


def _described(cls: type,
               displays: dict[str, Display],
               prefix: str = "",
               applies_to: list[str] | None = None) -> list[ParameterSpec]:
    """A class's constructor parameters, described, under *prefix*."""
    named = {name.removeprefix(prefix): display for name, display in displays.items()
             if name.startswith(prefix)}

    return [parameter_spec(parameter, {}).model_copy(
                update={"name": prefix + parameter.name, "applies_to": applies_to})
            for parameter in catalogue.parameters_of(cls, named)]


def _vehicle_settings() -> list[ParameterSpec]:
    """What `vehicle.settings` accepts."""
    common = _described(Vehicle, VEHICLE_DISPLAYS)
    etf = [spec for spec in _described(EtfVehicle, ETF_DISPLAYS,
                                       applies_to=[EXCHANGE_TRADED])
           if spec.name in ETF_DISPLAYS]

    return [*common, *etf]


def _implementation_settings() -> list[ParameterSpec]:
    """The implementation's plain settings."""
    specs = [*_described(MinimumPosition, IMPLEMENTATION_DISPLAYS, "minimum_position."),
             *_described(MarketImpact, IMPLEMENTATION_DISPLAYS, "impact."),
             *_described(ExecutionLimit, IMPLEMENTATION_DISPLAYS, "execution.")]

    return [*specs,
            ParameterSpec(name="redistribution", type=catalogue.STRING, required=False,
                          default="pro_rata", label="Removed weight goes",
                          order=len(specs), choices=list(REDISTRIBUTIONS)),
            ParameterSpec(name="cash_buffer", type=catalogue.NUMBER, required=False,
                          default=0.0, label="Cash buffer", order=len(specs) + 1,
                          unit=FRACTION, minimum=0.0, maximum=1.0),
            ParameterSpec(name="invest_flows", type=catalogue.STRING, required=False,
                          default="target", label="Inflows buy", order=len(specs) + 2,
                          choices=list(FLOW_INVESTMENTS))]


def _assumptions() -> list[ParameterSpec]:
    """Each modelling assumption, with the engine's default."""
    defaults = engine_defaults()

    return [spec.model_copy(update={"default": defaults.get(spec.name)})
            for spec in _described(ModellingAssumptions, ASSUMPTION_DISPLAYS)]


def engine_defaults() -> dict[str, Any]:
    """The modelling assumptions a run makes when none are set: the library's
    conventions, and the data treatment a data source has by default (no
    staleness limit is 0 here)."""
    defaults = ModellingAssumptions().with_defaults().as_dict()
    defaults.update(fx_policy=DEFAULT_FX_POLICY, max_price_staleness_days=0,
                    free_float_backfill_days=DEFAULT_FREE_FLOAT_BACKFILL_DAYS)

    return defaults
