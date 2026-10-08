# src/beacon/fund/fund.py
"""
A fund product: what it is called, what it holds and how it is held.

    fund = Fund(name="Sample UCITS ETF", strategy=my_index,
                vehicle=ucits_etf(management_fee_bps=12),
                share_classes=[ShareClass("Acc"), ShareClass("Dist", distribution="distributing")])

    result = fund.backtest(end="2024-12-31", initial_capital=1e8,
                           share_class="Dist", data_provider=fetcher)

A `Fund` is a record of a real product, as a prospectus describes one: a
name, a currency, its share classes and documents, with one strategy (what
it holds) and one vehicle (how the money is held). It is not what the engine
runs: `fund.backtest(...)` is a `Backtest` with the fund's vehicle, run on
its strategy, so a saved product and a what-if run are the same computation.

## Share classes

Each share class has its own management fee and distribution policy. A
backtest of a class runs the fund once with that class's settings: its fee
in place of the vehicle's, and its dividends reinvested (accumulating) or
paid out (distributing). In a real fund every class shares one pool of
assets, so its flows trade one book and share its capacity and costs; a run
per class leaves that out, and per-class returns are otherwise the same.
"""
# BN-268, phase 6 of decisions/0006. Modelling the classes' shared pool of
# assets is BN-279 (#292).
import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..backtest.dividends import DISTRIBUTE, REINVEST
from ..backtest.flows import Flows
from ..backtest.implementation import Implementation
from ..backtest.main import Backtest
from ..backtest.result import BacktestResult
from ..backtest.vehicle import Vehicle
from ..data.fetcher import DataFetcher
from ..index.cache import IndexResultCache
from ..index.derived import AnyIndexDefinition
from ..index.result import IndexResult
from ..strategy.tracking import IndexTracking

ACCUMULATING = "accumulating"
DISTRIBUTING = "distributing"
DISTRIBUTIONS = (ACCUMULATING, DISTRIBUTING)


@dataclass(frozen=True)
class ShareClass:
    """One share class of a fund.

    Attributes:
        name: What the class is called, such as ``"Acc"`` or ``"I GBP Dist"``.
        management_fee_bps: The class's annual management fee in basis
            points. None uses the vehicle's.
        distribution: ``"accumulating"`` (the default) reinvests its income,
            ``"distributing"`` pays it out.

    Raises:
        ValueError: If the distribution is not one of the two, or the fee is
            negative.
    """
    name: str
    management_fee_bps: float | None = None
    distribution: str = ACCUMULATING

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution {self.distribution!r}. "
                             f"Supported: {', '.join(DISTRIBUTIONS)}.")

        if self.management_fee_bps is not None and self.management_fee_bps < 0:
            raise ValueError(f"management_fee_bps cannot be negative, got "
                             f"{self.management_fee_bps!r}.")


class Fund:
    """A fund product, with one strategy and one vehicle.

    Args:
        name: The fund's name.
        strategy: What it holds: an index definition, tracked in full, or
            an `IndexTracking` holding one through a replication.
        vehicle: How the money is held, such as a preset from
            `beacon.backtest.presets`.
        currency: The fund's base currency. None keeps the index's.
        share_classes: Its share classes. One accumulating class at the
            vehicle's fee when none are given.
        documents: Its documents by name (a prospectus, a factsheet), as
            paths or links. Kept with the record and not read.

    Raises:
        ValueError: If *name* is empty or two classes share a name.
    """

    def __init__(self,
                 name: str,
                 strategy: AnyIndexDefinition | IndexTracking,
                 vehicle: Vehicle,
                 currency: str | None = None,
                 share_classes: Iterable[ShareClass] = (),
                 documents: Mapping[str, str] | None = None):
        if not name:
            raise ValueError("A fund needs a name.")

        classes = tuple(share_classes) or (ShareClass(name="Default"),)
        names = [share_class.name for share_class in classes]

        if len(set(names)) != len(names):
            raise ValueError(f"Share class names must differ, got {names}.")

        self.name = name
        self.strategy = strategy
        self.vehicle = vehicle
        self.currency = currency.upper() if currency else None
        self.share_classes: tuple[ShareClass, ...] = classes
        self.documents: dict[str, str] = dict(documents or {})

    def share_class(self,
                    name: str | None = None) -> ShareClass:
        """A share class by name, or the first when *name* is None.

        Raises:
            KeyError: If there is no class by that name; the message lists
                them.
        """
        if name is None:
            return self.share_classes[0]

        for share_class in self.share_classes:
            if share_class.name == name:
                return share_class

        raise KeyError(f"{self.name} has no share class {name!r}. Classes: "
                       f"{', '.join(c.name for c in self.share_classes)}.")

    def vehicle_for(self,
                    share_class: ShareClass) -> Vehicle:
        """The fund's vehicle with *share_class*'s fee."""
        if share_class.management_fee_bps is None:
            return self.vehicle

        vehicle = copy.copy(self.vehicle)
        vehicle.management_fee_bps = share_class.management_fee_bps

        return vehicle

    def backtest(self,
                 end: str,
                 initial_capital: float,
                 start: str | None = None,
                 share_class: str | None = None,
                 flows: Flows | list[Flows] | None = None,
                 implementation: Implementation | None = None,
                 modelling_assumptions: ModellingAssumptions | None = None,
                 transaction_cost_bps: float = 0.0,
                 benchmark: IndexResult | pd.Series | None = None,
                 data_provider: DataFetcher | None = None,
                 cache: IndexResultCache | Literal[False] | None = None,
                 **backtest_settings: Any) -> BacktestResult:
        """Backtest one share class: a `Backtest` with the fund's vehicle,
        run on its strategy.

        Args:
            end: The last day, YYYY-MM-DD.
            initial_capital: The class's capital at launch.
            start: The first day; the strategy's base date when None.
            share_class: Which class, by name; the first when None.
            flows: Money arriving and leaving the class.
            implementation: Screens, caps, costs and execution.
            modelling_assumptions: What the run takes as given.
            transaction_cost_bps: The fixed cost of each trade.
            benchmark: The benchmark of record.
            data_provider: The data the run reads; the ambient source when
                None.
            cache: Where calculated indices are kept; see `Backtest`.
            **backtest_settings: Any other `Backtest` setting, such as
                `modifiers` or `price_column`.

        Returns:
            BacktestResult: The class's run.
        """
        chosen = self.share_class(share_class)
        dividends = DISTRIBUTE if chosen.distribution == DISTRIBUTING else REINVEST

        backtest = Backtest(initial_capital=initial_capital,
                            transaction_cost_bps=transaction_cost_bps,
                            currency=self.currency,
                            modelling_assumptions=modelling_assumptions,
                            dividends=dividends,
                            implementation=implementation,
                            benchmark=benchmark,
                            data_provider=data_provider,
                            cache=cache,
                            flows=flows,
                            vehicle=self.vehicle_for(chosen),
                            **backtest_settings)

        return backtest.run(self.strategy, start=start, end=end)

    def __repr__(self) -> str:
        return (f"Fund(name={self.name!r}, vehicle={self.vehicle.name!r}, "
                f"share_classes={[c.name for c in self.share_classes]!r})")
