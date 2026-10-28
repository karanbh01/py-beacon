# tests/test_implementation_breakdown.py
"""BN-396: the tracking difference split into what caused it, checked two
ways: the parts add up to it, and each part is what switching its feature
off changes it by.

Runs from test_strategy_futures (an eight-name market-cap index through
2024, a cash rate of 2%), test_backtest_dividends (one name paying a
dividend, against a total return or a price index) and test_backtest_flows
(two names, one large flow).
"""
import pytest

from beacon.analysis.attribution import Attribution
from beacon.analysis.attribution import implementation as breakdown
from beacon.analysis.data import RunData
from beacon.assumptions import ModellingAssumptions
from beacon.backtest import Implementation
from beacon.strategy.futures import IndexFutures


def broken_down(result):
    return Attribution(RunData.from_result(result)).implementation()


def strategy_run(**settings):
    from test_strategy_futures import index, run

    return run(index(), cash_rate=0.02, **settings)


@pytest.fixture(scope="module")
def plain():
    return broken_down(strategy_run())


def change(found, base) -> float:
    """What a feature changed the tracking difference by."""
    return found.total_return - base.total_return


class TestThePartsAddUp:

    @pytest.mark.parametrize("settings", [
        {}, {"transaction_cost_bps": 10.0},
        {"implementation": Implementation(cash_buffer=0.05)},
        {"implementation": Implementation(cash_buffer=0.05, equitise=IndexFutures(
            dividend_yield=0.0, whole_contracts=False, roll_cost_bps=2.0,
            cost_per_contract=5.0))},
    ])
    def test_for_each_kind_of_fund(self,
                                   settings):
        found = broken_down(strategy_run(**settings))

        assert found.reconciles(1e-9)
        assert set(found.effects.index) == set(breakdown.PARTS)

    def test_with_dividends_and_withholding_exactly(self):
        from test_backtest_dividends import run

        for settings in ({}, {"modelling_assumptions": ModellingAssumptions(
                withholding_tax_rate=0.15)}):
            assert broken_down(run(**settings)).reconciles(1e-12)

    @pytest.mark.parametrize("flow", [500_000.0, -300_000.0])
    def test_with_a_flow_to_within_a_basis_point(self,
                                                 flow):
        """A flow is traded the day it arrives, at a cost the parts count;
        what the day's trading does beyond that is the residual."""
        from beacon.backtest import SinglePricing
        from test_backtest_dealing import protected

        assert abs(broken_down(protected(SinglePricing(), flow=flow)).residual) < 2e-4


class TestEachPartIsWhatItsFeatureCosts:

    def test_costs(self,
                   plain):
        found = broken_down(strategy_run(transaction_cost_bps=10.0))

        assert found.effects[breakdown.FIXED_COSTS] < 0.0
        assert found.effects[breakdown.FIXED_COSTS] == pytest.approx(change(found, plain),
                                                                     rel=0.02)

    def test_a_cash_buffer(self,
                           plain):
        found = broken_down(strategy_run(implementation=Implementation(cash_buffer=0.05)))

        assert found.effects[breakdown.CASH_DRAG] - plain.effects[breakdown.CASH_DRAG] == (
            pytest.approx(change(found, plain), rel=0.02))

    def test_equitising_it(self):
        cash = broken_down(strategy_run(implementation=Implementation(cash_buffer=0.05)))
        futures = broken_down(strategy_run(implementation=Implementation(
            cash_buffer=0.05, equitise=IndexFutures(dividend_yield=0.0, whole_contracts=False))))

        assert futures.effects[breakdown.EQUITISATION] > 0.0
        assert (futures.effects[breakdown.EQUITISATION] + futures.effects[breakdown.CASH_DRAG]
                - cash.effects[breakdown.CASH_DRAG]) == pytest.approx(change(futures, cash),
                                                                      rel=0.05)

    def test_futures_costs(self):
        free = broken_down(strategy_run(implementation=Implementation(
            cash_buffer=0.05, equitise=IndexFutures(dividend_yield=0.0, whole_contracts=False))))
        costly = broken_down(strategy_run(implementation=Implementation(
            cash_buffer=0.05, equitise=IndexFutures(dividend_yield=0.0, whole_contracts=False,
                                                    roll_cost_bps=20.0))))
        rolls = costly.effects[breakdown.FUTURES_ROLLS]

        assert rolls < 0.0
        assert rolls == pytest.approx(change(costly, free), rel=0.05)

    def test_withholding_tax(self):
        from test_backtest_dividends import run

        free = broken_down(run())
        taxed = broken_down(run(modelling_assumptions=ModellingAssumptions(
            withholding_tax_rate=0.15)))

        assert taxed.effects[breakdown.WITHHOLDING_TAX] < 0.0
        assert taxed.effects[breakdown.WITHHOLDING_TAX] == pytest.approx(change(taxed, free),
                                                                         rel=0.02)
        assert taxed.reconciles(1e-9)

    def test_a_price_index_leaves_its_dividends_out(self):
        from beacon.index.calculation import IndexCalculator
        from beacon.index.constructor import IndexDefinition
        from beacon.index.methodology import EqualWeighted
        from test_backtest_dividends import DAYS, fetcher, run

        data = fetcher()
        price = IndexCalculator(IndexDefinition(
            index_id="px", index_name="Price", base_date=str(DAYS[0].date()), base_value=100.0,
            currency="USD", eligibility_rules=[], weighting_scheme=EqualWeighted(),
            rebalancing_frequency="MONTHLY", calendar="XNYS", universe_identifiers=["AAA"]),
            data).run(end_date=str(DAYS[-1].date()))
        total = broken_down(run(data))
        found = broken_down(run(data, price))

        assert found.effects[breakdown.DIVIDEND_TREATMENT] > 0.0
        assert found.effects[breakdown.DIVIDEND_TREATMENT] == pytest.approx(
            change(found, total), rel=0.02)

    def test_creation_fees(self):
        from beacon.backtest import DilutionLevy, SinglePricing
        from test_backtest_dealing import protected

        single = broken_down(protected(SinglePricing()))
        levied = broken_down(protected(DilutionLevy(rate_bps=50.0)))

        assert levied.effects[breakdown.CREATION_FEES] > 0.0
        assert levied.effects[breakdown.CREATION_FEES] == pytest.approx(change(levied, single),
                                                                        rel=0.05)
