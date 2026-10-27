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
from ..backtest.dividends import DIVIDEND_POLICIES, REINVESTMENT_TARGETS
from ..backtest.etf import EtfMarket, EtfVehicle
from ..backtest.implementation import FLOW_INVESTMENTS, REDISTRIBUTIONS
from ..backtest.share_classes import DISTRIBUTIONS, ShareClass
from ..backtest.vehicle import Vehicle
from ..catalogue import BPS, COUNT, DAYS, FRACTION, MONEY, RATIO, Display
from ..data.fetcher import DEFAULT_FX_POLICY, FX_POLICIES
from ..data.free_float import DEFAULT_FREE_FLOAT_BACKFILL_DAYS
from ..strategy.active import FREQUENCIES
from ..strategy.futures import IndexFutures
from .backtest_settings import (
    EXCHANGE_TRADED,
    STRATEGY_TYPES,
    archetype_of,
    preset_keys,
    preset_vehicle,
    vehicle_settings,
)
from .schemas import BENCHMARK_KINDS, BacktestOptions, ParameterSpec, VehiclePreset
from .types import parameter_spec, specs_for

VEHICLE_DISPLAYS = {
    "management_fee_bps": Display("Management fee", unit=BPS, minimum=0.0),
    "launch_price": Display("Launch price per unit", unit=MONEY, minimum=0.0),
    "pricing": Display("Pricing method", help="One of `pricing`, as {type, params}."),
    "limits": Display("Diversification limits", help="Any of `limits`, as {type, params}."),
    "name": Display("Name"),
    "domicile": Display("Domicile", help="A two-letter country code, such as 'IE'. "
                        "It sets the withholding tax on the fund's dividends."),
    "withholding_rates": Display("Withholding rates", help="The fund's own rates by "
                                 "the dividend's source country, such as {\"US\": 0.15}."),
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
    "equitise.roll_days": Display("Futures roll, days before expiry", unit=COUNT,
                                  minimum=0),
    "equitise.roll_cost_bps": Display("Cost of each futures roll", unit=BPS, minimum=0.0),
    "equitise.dividend_yield": Display("Index dividend yield", unit=FRACTION, minimum=0.0,
                                       help="Leave blank to estimate it."),
    # BN-358.
    "equitise.multiplier": Display("Value of an index point", unit=MONEY, minimum=0.0,
                                   help="Per futures contract, in the book's currency."),
    "equitise.whole_contracts": Display("Whole futures contracts"),
    "equitise.band": Display("Resize futures when exposure drifts by", unit=FRACTION,
                             minimum=0.0, maximum=1.0, help="A share of the net assets."),
}

# BN-314.
SHARE_CLASS_DISPLAYS = {
    "name": Display("Name"),
    "management_fee_bps": Display("Management fee", unit=BPS, minimum=0.0,
                                  help="Leave blank for the vehicle's."),
    "distribution": Display("Income", choices=tuple(DISTRIBUTIONS)),
    "currency": Display("Currency", help="A three-letter code, such as 'EUR'. Leave "
                        "blank for the book's."),
    "hedged": Display("Hedged into its currency"),
    "hedge_ratio": Display("Share hedged", unit=FRACTION, minimum=0.0, maximum=2.0),
    "hedge_band": Display("Hedge drift allowed", unit=FRACTION, minimum=0.0,
                          help="How far the hedge ratio may move before the forward "
                               "is resized. Leave blank to resize only at the rolls."),
    "hedge_cost_bps": Display("Hedge trading cost", unit=BPS, minimum=0.0,
                              help="On the notional each hedge trade deals."),
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
    # BN-330: what `OptimisedReplication.constraints` takes, the optimiser's own.
    "replication_constraints": catalogue.CONSTRAINT,
}

# Each kind by the family that lists it, for a parameter whose entries are
# themselves configurable types.
SERVED_AS = {kind: name for name, kind in FAMILIES.items()}


def backtest_options() -> BacktestOptions:
    """The response of `GET /backtest/options`."""
    families: dict[str, Any] = {name: specs_for(kind, SERVED_AS)
                                for name, kind in FAMILIES.items()}

    return BacktestOptions(
        **families,
        presets=[VehiclePreset(key=key, name=preset_vehicle(key).name,
                               archetype=archetype_of(key),
                               settings=vehicle_settings(preset_vehicle(key)))
                 for key in preset_keys()],
        vehicle_settings=_vehicle_settings(),
        market_settings=_described(EtfMarket, MARKET_DISPLAYS),
        run_settings=_run_settings(),
        implementation_settings=_implementation_settings(),
        share_class_settings=_described(ShareClass, SHARE_CLASS_DISPLAYS),
        modelling_assumptions=_assumptions(),
        choices={"dividends": list(DIVIDEND_POLICIES),
                 "reinvest_into": list(REINVESTMENT_TARGETS), "strategy": list(STRATEGY_TYPES),
                 "rebalancing": list(FREQUENCIES), "redistribution": list(REDISTRIBUTIONS),
                 "invest_flows": list(FLOW_INVESTMENTS), "fx_policy": list(FX_POLICIES),
                 "distribution": list(DISTRIBUTIONS)})


def _described(cls: type,
               displays: dict[str, Display],
               prefix: str = "",
               applies_to: list[str] | None = None) -> list[ParameterSpec]:
    """A class's constructor parameters, described, under *prefix*."""
    named = {name.removeprefix(prefix): display for name, display in displays.items()
             if name.startswith(prefix)}

    # Read off the class as `specs_for` reads it (BN-330).
    schemas = getattr(cls, "PARAM_SCHEMAS", {})
    options = {name: SERVED_AS[kind] for name, kind in getattr(cls, "PARAM_KINDS", {}).items()}

    return [parameter_spec(parameter, schemas, options=options).model_copy(
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
             *_described(ExecutionLimit, IMPLEMENTATION_DISPLAYS, "execution."),
             *_described(IndexFutures, IMPLEMENTATION_DISPLAYS, "equitise.")]

    return [*specs,
            ParameterSpec(name="redistribution", type=catalogue.STRING, required=False,
                          default="pro_rata", label="Removed weight goes",
                          order=len(specs), choices=list(REDISTRIBUTIONS)),
            ParameterSpec(name="cash_buffer", type=catalogue.NUMBER, required=False,
                          default=0.0, label="Cash buffer", order=len(specs) + 1,
                          unit=FRACTION, minimum=0.0, maximum=1.0),
            ParameterSpec(name="invest_flows", type=catalogue.STRING, required=False,
                          default="target", label="Inflows buy", order=len(specs) + 2,
                          choices=list(FLOW_INVESTMENTS)),
            # BN-358.
            ParameterSpec(name="cash_band.lower", type=catalogue.NUMBER, required=False,
                          default=None, label="Lowest cash", order=len(specs) + 3,
                          unit=FRACTION, minimum=0.0, maximum=1.0,
                          help="Below it, holdings are sold back to the cash buffer."),
            ParameterSpec(name="cash_band.upper", type=catalogue.NUMBER, required=False,
                          default=None, label="Highest cash", order=len(specs) + 4,
                          unit=FRACTION, minimum=0.0, maximum=1.0,
                          help="Above it, cash is invested back down to the cash buffer.")]


def _run_settings() -> list[ParameterSpec]:
    """The run's own settings: its dates, capital, cost, currency, dividend
    policy and benchmark."""
    # BN-321: the request always took these, but the catalogue never described
    # them, so a form built from it had no dates, capital or cost.
    rows: list[dict[str, Any]] = [
        {"name": "start", "type": catalogue.DATE, "label": "Start",
         "help": "Leave blank to start at the index's base date."},
        {"name": "end", "type": catalogue.DATE, "label": "End",
         "help": "Leave blank to run to the last date the data covers."},
        {"name": "initial_capital", "type": catalogue.NUMBER, "default": 1_000_000.0,
         "label": "Initial capital", "unit": MONEY, "minimum": 0.0,
         "help": "With share classes, give it by class name, each in its class's currency."},
        {"name": "transaction_cost_bps", "type": catalogue.NUMBER, "default": 0.0,
         "label": "Trading cost", "unit": BPS, "minimum": 0.0,
         "help": "Per trade, on its notional."},
        {"name": "currency", "type": catalogue.STRING, "label": "Book currency",
         "help": "A three-letter code, such as 'EUR'. Leave blank for the index's."},
        {"name": "dividends", "type": catalogue.STRING, "default": "reinvest",
         "label": "Dividends", "choices": list(DIVIDEND_POLICIES)},
        {"name": "reinvest_into", "type": catalogue.STRING, "default": "payer",
         "label": "Reinvest dividends in", "choices": list(REINVESTMENT_TARGETS),
         "help": "The name that paid each one, or the holdings in proportion to value."},
        {"name": "benchmark.kind", "type": catalogue.STRING, "label": "Benchmark",
         "choices": list(BENCHMARK_KINDS),
         "help": "A stored index, or a series in the market data. Leave blank for none."},
        {"name": "benchmark.id", "type": catalogue.STRING, "label": "Benchmark id"},
        {"name": "benchmark.price_column", "type": catalogue.STRING, "default": "CLOSE",
         "label": "Benchmark column",
         "help": "The market-data column read for an identifier benchmark."},
    ]

    return [ParameterSpec(required=False, order=order, **row)
            for order, row in enumerate(rows)]


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
