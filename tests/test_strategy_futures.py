# tests/test_strategy_futures.py
"""BN-305: futures-based replication, and futures equitising a cash buffer.

Eight names on one market factor through 2024, with quarterly dividends in
2023 only, so the trailing yield at each roll is known but no dividend falls
inside the run. With no cash rate and no dividend yield a contract's fair
value is the index level itself, so a futures-based fund must end exactly
where the index does.
"""
import numpy as np
import pandas as pd
import pytest

from beacon import catalogue
from beacon.assumptions import ModellingAssumptions
from beacon.backtest import Backtest, DatedFlows, Implementation
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.index.schedule import sessions
from beacon.strategy import FuturesReplication, IndexFutures, IndexTracking
from beacon.strategy.futures import expiries

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
NAMES = [f"N{i}" for i in range(8)]
START, END = "2024-01-02", "2024-12-31"
FLOW_DAY = DAYS[120]
# A quarter's dividend per share, paid in 2023 only.
QUARTERLY_DIVIDEND = 0.35


def fetcher() -> DataFetcher:
    random = np.random.default_rng(3)
    market = random.normal(0.0004, 0.010, len(DAYS))
    rows, actions = [], []

    for position, name in enumerate(NAMES):
        prices = 40.0 * np.cumprod(1.0 + market + random.normal(0.0, 0.005, len(DAYS)))

        for day, price in zip(DAYS, prices, strict=True):
            rows.append({"IDENTIFIER": name, "DATE": day, "CLOSE": price,
                         "SHARES_OUTSTANDING": 1e9 / (1 + position)})

        actions.extend({"IDENTIFIER": name, "EX_DATE": ex, "TYPE": "DIVIDEND",
                        "VALUE": QUARTERLY_DIVIDEND * (1 + position % 3)}
                       for ex in ("2023-03-15", "2023-06-15", "2023-09-15", "2023-12-15"))

    reference = pd.DataFrame([{"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                               "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       CorporateActions.from_dataframe(pd.DataFrame(actions)))


DATA = fetcher()


def index(return_type: str = "PRICE") -> IndexDefinition:
    return IndexDefinition(index_id=f"MCW_{return_type}", index_name="Market cap",
                           base_date=START, base_value=1000.0, currency="USD",
                           eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
                           rebalancing_frequency="QUARTERLY", calendar="XNYS",
                           universe_identifiers=NAMES, return_type=return_type)


def run(strategy, cash_rate: float = 0.0, **settings):
    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    modelling_assumptions=ModellingAssumptions(cash_rate=cash_rate),
                    **settings).run(strategy, start=START, end=END)


def futures(**terms):
    return IndexTracking(index(), FuturesReplication(**terms))


def growth(series: pd.Series) -> float:
    return float(series.iloc[-1] / series.iloc[0])


NO_YIELD = {"dividend_yield": 0.0}


@pytest.fixture(scope="module")
def plain():
    return run(futures(**NO_YIELD))


class TestConstruction:

    @pytest.mark.parametrize(("terms", "message"), [
        ({"roll_days": -1}, "roll_days"),
        ({"roll_cost_bps": -1.0}, "roll_cost_bps"),
        ({"roll_cost_bps": float("nan")}, "roll_cost_bps"),
        ({"dividend_yield": 1.5}, "dividend_yield"),
    ])
    def test_settings_out_of_range_are_refused(self, terms, message):
        with pytest.raises(ValueError, match=message):
            FuturesReplication(**terms)

    def test_contracts_expire_on_the_third_friday_of_each_quarter(self):
        found = expiries(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

        assert [day.date().isoformat() for day in found[:4]] == [
            "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20"]
        assert all(day.weekday() == 4 for day in found)

    def test_the_fund_holds_no_securities(self):
        step = FuturesReplication().replicate({"N0": 1.0}, pd.Timestamp(START),
                                              None)  # type: ignore[arg-type]

        assert step.weights == {}


class TestTheFundEarnsTheIndex:

    def test_it_holds_cash_and_futures(self, plain):
        assert plain.portfolio.holdings == {}
        assert plain.futures_exposure.max() == pytest.approx(1.0)

    def test_with_no_carry_it_ends_exactly_where_the_index_does(self, plain):
        assert growth(plain.nav_per_unit) == pytest.approx(
            growth(plain.index.target.levels), rel=1e-9)

    def test_the_cash_rate_is_given_back_through_the_futures(self):
        result = run(futures(**NO_YIELD), cash_rate=0.04)

        # The cash earns 4%; the futures give it up as carry. Between resizes
        # the two differ a little, as a real position's do.
        assert growth(result.nav_per_unit) == pytest.approx(
            growth(result.index.target.levels), abs=0.003)

    def test_the_exposure_is_reset_to_net_assets_at_each_rebalance(self, plain):
        exposure = plain.futures_exposure

        for step in plain.replication:
            assert exposure[step.date] == pytest.approx(1.0)

    def test_a_flow_is_put_to_work_the_day_it_arrives(self):
        flowed = run(futures(**NO_YIELD), flows=DatedFlows({FLOW_DAY: 3e6}))

        assert flowed.futures_exposure[FLOW_DAY] == pytest.approx(1.0)
        assert growth(flowed.nav_per_unit) == pytest.approx(
            growth(flowed.index.target.levels), rel=1e-9)


class TestRolls:

    def test_it_rolls_five_sessions_before_each_expiry(self, plain):
        rolled = [(roll.date.date().isoformat(), roll.expiry.date().isoformat())
                  for roll in plain.futures_rolls]

        assert rolled == [("2024-03-08", "2024-06-21"), ("2024-06-14", "2024-09-20"),
                          ("2024-09-13", "2024-12-20"), ("2024-12-13", "2025-03-21")]

    def test_each_roll_costs_its_share_of_the_notional(self):
        result = run(futures(roll_cost_bps=3.0, **NO_YIELD))

        # Charged on the notional before the roll, then the new contracts
        # carry what is left.
        for roll in result.futures_rolls:
            assert roll.cost == pytest.approx((roll.notional + roll.cost) * 0.0003)

    def test_the_roll_cost_is_what_the_fund_gives_up(self, plain):
        costly = run(futures(roll_cost_bps=3.0, **NO_YIELD))
        levels = costly.index.target.levels

        # Each roll's cost, as it would have grown in the index since.
        expected = sum(roll.cost * levels.iloc[-1] / levels.asof(roll.date)
                       for roll in costly.futures_rolls) / 1e7
        drag = growth(plain.nav_per_unit) - growth(costly.nav_per_unit)

        assert drag == pytest.approx(expected, rel=1e-6)

    def test_margin_and_roll_costs_are_cash_flows(self):
        result = run(futures(roll_cost_bps=3.0, **NO_YIELD))
        kinds = {flow.kind for flow in result.portfolio.cash_flows}

        assert "FUTURES" in kinds


class TestTheDividendYield:

    def test_a_price_index_is_estimated_from_its_constituents(self):
        result = run(futures())
        first = result.futures_rolls[0]
        weights = result.index.target.source.weight_snapshots
        latest = weights[max(day for day in weights if day <= first.date)]
        expected = (sum(weight * DATA.fetch_trailing_dividend_yield(name, first.date)
                        for name, weight in latest.items()) / sum(latest.values()))

        assert first.dividend_yield == pytest.approx(expected)
        assert first.dividend_yield > 0.01

    def test_a_total_return_index_carries_none(self):
        result = run(IndexTracking(index("TOTAL_RETURN"), FuturesReplication()))

        assert {roll.dividend_yield for roll in result.futures_rolls} == {0.0}

    def test_the_futures_pay_the_yield_they_are_priced_with(self, plain):
        paid = run(futures(dividend_yield=0.02))
        excess = growth(paid.nav_per_unit) / growth(plain.nav_per_unit) - 1.0
        days = (DAYS[-1] - DAYS[0]).days

        # Roughly the yield over the year; the futures' carry is continuous.
        assert excess == pytest.approx(0.02 * days / 365, rel=0.1)


class TestEquitising:

    def test_the_buffer_is_held_in_futures(self):
        result = run(index(), implementation=Implementation(
            cash_buffer=0.10, equitise=IndexFutures(**NO_YIELD)))

        assert result.futures_exposure.mean() == pytest.approx(0.10, abs=0.01)
        assert set(result.portfolio.holdings) == set(NAMES)

    def test_it_brings_a_buffered_fund_back_to_the_index(self):
        buffered = run(index(), implementation=Implementation(cash_buffer=0.10))
        equitised = run(index(), implementation=Implementation(
            cash_buffer=0.10, equitise=IndexFutures(**NO_YIELD)))
        tracked = growth(equitised.index.target.levels)

        assert (abs(growth(equitised.nav_per_unit) - tracked)
                < abs(growth(buffered.nav_per_unit) - tracked) / 5)

    def test_the_implementation_given_is_left_as_it_was(self):
        implementation = Implementation(cash_buffer=0.05, equitise=IndexFutures())
        run(index("TOTAL_RETURN"), implementation=implementation)

        assert implementation.equitise is not None
        assert implementation.equitise.dividend_yield is None

    def test_a_fund_without_futures_records_none(self):
        result = run(index())

        assert result.futures_notional.empty
        assert result.futures_rolls == []
        assert result.futures_exposure.empty


class TestTheCatalogue:

    def test_the_editor_can_offer_it(self):
        entry = catalogue.entry_for(catalogue.REPLICATION, "FuturesReplication")

        assert {parameter.name for parameter in entry.parameters} == {
            "roll_days", "roll_cost_bps", "dividend_yield"}

    def test_its_settings_round_trip(self):
        from beacon.server.backtest_settings import typed_payload

        payload = typed_payload(catalogue.REPLICATION,
                                FuturesReplication(roll_days=3, roll_cost_bps=1.5))

        assert payload == {"type": "FuturesReplication",
                           "params": {"roll_days": 3, "roll_cost_bps": 1.5,
                                      "dividend_yield": None}}
