# src/beacon/server/share_class_settings.py
"""
A backtest request's share classes, and its capital and flows by class.

Without share classes a request's capital is one amount and its flows one
list, as before. With them, each class can have its own capital and flows,
keyed by its name, in its own currency, and a single amount or list is taken
for a fund of one class only, as `Backtest` takes them. Every problem is a
finding at its field's path, such as `share_classes.1.hedge_band` or
`flows.EUR H.0`.
"""
# BN-314.
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..backtest.flows import Flows
from ..backtest.share_classes import ShareClass
from .schemas import StrategyBacktestRequest, TypedSpec

# Records a finding: its path, code and message.
Report = Callable[[str, str, str], None]
# Builds one flow scenario from its {type, params}, at a path; None when it
# could not be built, having reported why.
BuildFlow = Callable[[TypedSpec, str], Flows | None]


@dataclass
class FundTerms:
    """A request's share classes, capital and flows as `Backtest` takes them.

    Attributes:
        share_classes: The classes; empty for a fund of one.
        initial_capital: One amount, or one per class name.
        flows: One list of scenarios, or one per class name.
    """
    share_classes: list[ShareClass]
    initial_capital: float | dict[str, float]
    flows: list[Flows] | dict[str, list[Flows]]

    def resolved(self) -> dict[str, Any]:
        """The share classes as the run reports them."""
        return {"share_classes": [_described(share_class)
                                  for share_class in self.share_classes]}


def fund_terms(request: StrategyBacktestRequest,
               report: Report,
               build_flow: BuildFlow) -> FundTerms:
    """The request's share classes, capital and flows, each problem reported."""
    classes = _share_classes(request, report)
    names = [spec.name for spec in request.share_classes]

    return FundTerms(share_classes=classes,
                     initial_capital=_capital(request.initial_capital, names, report),
                     flows=_flows(request.flows, names, report, build_flow))


def _share_classes(request: StrategyBacktestRequest,
                   report: Report) -> list[ShareClass]:
    """Each class built, its problems at its path."""
    built: list[ShareClass] = []
    seen: set[str] = set()
    book = request.currency.upper() if request.currency else None

    for position, spec in enumerate(request.share_classes):
        path = f"share_classes.{position}"

        if spec.name in seen:
            report(f"{path}.name", "duplicate", f"Another class is called '{spec.name}'.")
        seen.add(spec.name)

        if spec.hedged and spec.currency is None:
            report(f"{path}.currency", "missing_parameter",
                   "A hedged class needs a currency to hedge into.")

        if spec.hedged and spec.currency is not None and spec.currency.upper() == book:
            report(f"{path}.currency", "invalid_value",
                   f"A class cannot hedge into the book's own currency, {book}.")

        try:
            built.append(ShareClass(**spec.model_dump()))
        except ValueError as error:
            report(f"{path}.distribution" if "distribution" in str(error) else path,
                   "invalid_value", str(error))

    return built


def _capital(capital: float | dict[str, float],
             names: list[str],
             report: Report) -> float | dict[str, float]:
    """The capital, one amount or by class, checked against the classes."""
    if not isinstance(capital, dict):
        if capital <= 0:
            report("initial_capital", "invalid_value", "The capital must be above 0.")

        if len(names) > 1:
            report("initial_capital", "invalid_value",
                   f"With {len(names)} share classes, give the capital by class name.")

        return capital

    if not names:
        report("initial_capital", "invalid_value",
               "Capital by class name needs share classes.")

    for name, amount in capital.items():
        if names and name not in names:
            report(f"initial_capital.{name}", "unknown_class",
                   f"No share class is called '{name}'. Classes: {', '.join(names)}.")

        if amount <= 0:
            report(f"initial_capital.{name}", "invalid_value",
                   "The capital must be above 0.")

    return capital


def _flows(flows: list[TypedSpec] | dict[str, list[TypedSpec]],
           names: list[str],
           report: Report,
           build_flow: BuildFlow) -> list[Flows] | dict[str, list[Flows]]:
    """The flows, one list or by class, each built at its path."""
    if not isinstance(flows, dict):
        if flows and len(names) > 1:
            report("flows", "invalid_value",
                   f"With {len(names)} share classes, give the flows by class name.")

        return _built(flows, "flows", build_flow)

    if flows and not names:
        report("flows", "invalid_value", "Flows by class name need share classes.")

    for name in flows:
        if names and name not in names:
            report(f"flows.{name}", "unknown_class",
                   f"No share class is called '{name}'. Classes: {', '.join(names)}.")

    return {name: _built(specs, f"flows.{name}", build_flow) for name, specs in flows.items()}


def _built(specs: list[TypedSpec],
           path: str,
           build_flow: BuildFlow) -> list[Flows]:
    """Each scenario of a list built; those that could not be are left out,
    having been reported."""
    return [flow for position, spec in enumerate(specs)
            if (flow := build_flow(spec, f"{path}.{position}")) is not None]


def _described(share_class: ShareClass) -> dict[str, Any]:
    """A class's settings as the run reports them."""
    return {"name": share_class.name, "currency": share_class.currency,
            "hedged": share_class.hedged, "hedge_ratio": share_class.hedge_ratio,
            "hedge_band": share_class.hedge_band,
            "hedge_cost_bps": share_class.hedge_cost_bps,
            "management_fee_bps": share_class.management_fee_bps,
            "distribution": share_class.distribution}
