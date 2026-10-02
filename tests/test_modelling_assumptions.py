# tests/test_modelling_assumptions.py
"""BN-276: a backtest's modelling assumptions, in one object.

The object itself, the process-wide default and how the two combine, the
data treatment reaching both the index calculation and the simulation, the
simulation conventions (cash rate, risk-free rate, periods per year), and the
index cache keying on the data treatment.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from beacon import (
    ModellingAssumptions,
    current_modelling_assumptions,
    use_modelling_assumptions,
)
from beacon.backtest.engine import BacktestEngine
from beacon.backtest.main import Backtest
from beacon.data import store
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index import cache as index_cache
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.portfolio.cash_flows import INTEREST
from beacon.testing import dataset
from beacon.testing.weights import index_result_from_weights

DAYS = pd.bdate_range("2024-01-02", periods=40)
CAPITAL = 1_000_000.0


@pytest.fixture(autouse=True)
def _reset_the_process_default():
    """Each test starts and ends with nothing set process-wide."""
    use_modelling_assumptions(None)
    yield
    use_modelling_assumptions(None)


def fetcher(**settings) -> DataFetcher:
    """AAA rising 0.1% a day; the reference data marks it USD."""
    market = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE": day, "CLOSE": 100.0 * 1.001 ** step,
         "SHARES_OUTSTANDING": 1e6}
        for step, day in enumerate(DAYS)])
    reference = pd.DataFrame([{"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01",
                               "NAME": "A", "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"}])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference), **settings)


def definition() -> IndexDefinition:
    return IndexDefinition(index_id="one", index_name="One",
                           base_date=str(DAYS[0].date()), base_value=100.0,
                           currency="USD", eligibility_rules=[],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="MONTHLY", calendar="XNYS",
                           universe_identifiers=["AAA"])


def half_in_cash(**engine_args):
    """A run holding AAA at half weight, so half the book is cash."""
    data = fetcher()
    index = index_result_from_weights({DAYS[0]: {"AAA": 0.5}})

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=CAPITAL, data_provider=data,
                          index_result=index, **engine_args).run()


class TestTheObject:

    def test_every_field_is_unset_by_default(self):
        assert all(value is None
                   for value in ModellingAssumptions().as_dict().values())

    @pytest.mark.parametrize("settings", [
        {"fx_policy": "SOMETIMES"},
        {"max_price_staleness_days": -1},
        {"free_float_backfill_days": 1.5},
        {"periods_per_year": 0},
        {"cash_rate": 2.0},
    ])
    def test_an_impossible_value_is_refused(self,
                                            settings):
        with pytest.raises(ValueError):
            ModellingAssumptions(**settings)

    def test_a_backtests_own_fields_win_field_by_field(self):
        process = ModellingAssumptions(fx_policy="EXACT_DAY", cash_rate=0.05)
        own = ModellingAssumptions(cash_rate=0.01)

        merged = own.over(process)

        assert (merged.fx_policy, merged.cash_rate) == ("EXACT_DAY", 0.01)

    def test_resolving_fills_data_treatment_from_the_data_source(self):
        resolved = ModellingAssumptions().resolved(
            fetcher(fx_policy="EXACT_DAY", free_float_backfill_days=30))

        assert resolved.fx_policy == "EXACT_DAY"
        assert resolved.free_float_backfill_days == 30
        assert resolved.max_price_staleness_days == 0
        assert (resolved.cash_rate, resolved.risk_free_rate,
                resolved.periods_per_year) == (0.0, 0.0, 252)

    def test_nothing_set_leaves_the_data_source_itself(self):
        """So a run with no assumptions shares the source's caches."""
        data = fetcher()

        assert ModellingAssumptions(cash_rate=0.02).applied_to(data) is data

    def test_zero_staleness_means_no_limit(self):
        data = fetcher(max_price_staleness_days=5)

        applied = ModellingAssumptions(max_price_staleness_days=0).applied_to(data)

        assert applied.max_price_staleness_days is None


class TestTheProcessWideDefault:

    def test_it_can_be_set_and_reset(self):
        use_modelling_assumptions(ModellingAssumptions(cash_rate=0.03))

        assert current_modelling_assumptions().cash_rate == 0.03

        use_modelling_assumptions(None)

        assert current_modelling_assumptions() == ModellingAssumptions()

    def test_a_run_without_its_own_reads_it(self):
        use_modelling_assumptions(ModellingAssumptions(risk_free_rate=0.04))

        assert half_in_cash().modelling_assumptions.risk_free_rate == 0.04

    def test_a_runs_own_override_it_field_by_field(self):
        use_modelling_assumptions(ModellingAssumptions(risk_free_rate=0.04,
                                                       cash_rate=0.01))

        result = half_in_cash(
            modelling_assumptions=ModellingAssumptions(cash_rate=0.02))

        assert result.modelling_assumptions.cash_rate == 0.02
        assert result.modelling_assumptions.risk_free_rate == 0.04


class TestTheSimulationConventions:

    def test_the_defaults_change_nothing(self):
        plain = half_in_cash()
        explicit = half_in_cash(modelling_assumptions=ModellingAssumptions())

        pd.testing.assert_series_equal(plain.trading_nav, explicit.trading_nav)
        assert plain.portfolio.cash_flows == []

    def test_cash_earns_its_rate_act_365(self):
        """Half the book in cash at 5%. The whole capital earns one night
        before it is invested, from the eve of the first day."""
        plain = half_in_cash()
        earning = half_in_cash(
            modelling_assumptions=ModellingAssumptions(cash_rate=0.05))

        interest = [flow for flow in earning.portfolio.cash_flows
                    if flow.kind == INTEREST]
        eve = earning.portfolio.inception
        overnight = CAPITAL * 0.05 * (DAYS[0] - eve).days / 365.0
        invested_days = (DAYS[-1] - DAYS[0]).days

        assert interest[0].amount == pytest.approx(overnight)
        assert sum(flow.amount for flow in interest[1:]) == pytest.approx(
            CAPITAL * 0.5 * 0.05 * invested_days / 365.0, rel=5e-3)
        assert earning.trading_nav.iloc[-1] > plain.trading_nav.iloc[-1]

    def test_sharpe_is_measured_against_the_risk_free_rate(self):
        zero = half_in_cash().summary()
        four = half_in_cash(
            modelling_assumptions=ModellingAssumptions(risk_free_rate=0.04)
        ).summary()

        assert four["sharpe_ratio"] == pytest.approx(
            (zero["annualised_return"] - 0.04) / zero["volatility"])

    def test_annualising_uses_the_periods_per_year(self):
        """A calculated index, with costs so the book and index differ."""
        data = fetcher()
        index = IndexCalculator(definition(), data).run(
            end_date=str(DAYS[-1].date()))

        def run(assumptions=None):
            return BacktestEngine(start_date=str(DAYS[0].date()),
                                  end_date=str(DAYS[-1].date()),
                                  initial_capital=CAPITAL, data_provider=data,
                                  index_result=index, transaction_cost_bps=25.0,
                                  modelling_assumptions=assumptions).run()

        daily = run()
        weekly = run(ModellingAssumptions(periods_per_year=52))
        ratio = np.sqrt(52 / 252)

        assert daily.get_tracking_error() > 0
        assert weekly.get_tracking_error() == pytest.approx(
            daily.get_tracking_error() * ratio)
        assert weekly.summary()["volatility"] == pytest.approx(
            daily.summary()["volatility"] * ratio)


class TestTheIndexReadsDataTheSameWay:

    def test_a_calculation_records_its_data_treatment(self):
        result = IndexCalculator(
            definition(), fetcher(),
            modelling_assumptions=ModellingAssumptions(fx_policy="EXACT_DAY")
        ).run(end_date=str(DAYS[-1].date()))

        assert result.modelling_assumptions.fx_policy == "EXACT_DAY"
        assert result.modelling_assumptions.cash_rate is None

    def test_a_calculation_outside_a_backtest_reads_the_process_default(self):
        use_modelling_assumptions(ModellingAssumptions(free_float_backfill_days=7))

        result = IndexCalculator(definition(), fetcher()).run(
            end_date=str(DAYS[-1].date()))

        assert result.modelling_assumptions.free_float_backfill_days == 7

    def test_a_backtest_hands_its_assumptions_to_the_calculation(self):
        result = Backtest(initial_capital=CAPITAL, data_provider=fetcher(),
                          modelling_assumptions=ModellingAssumptions(
                              fx_policy="EXACT_DAY")).run(
            definition(), end=str(DAYS[-1].date()))

        assert result.index.target.source.modelling_assumptions.fx_policy == (
            "EXACT_DAY")
        assert result.modelling_assumptions.fx_policy == "EXACT_DAY"

    def test_an_index_calculated_differently_is_used_with_a_warning(self,
                                                                    caplog):
        data = fetcher()
        index = IndexCalculator(
            definition(), data,
            modelling_assumptions=ModellingAssumptions(fx_policy="EXACT_DAY")
        ).run(end_date=str(DAYS[-1].date()))

        with caplog.at_level(logging.WARNING):
            result = BacktestEngine(start_date=str(DAYS[0].date()),
                                    end_date=str(DAYS[-1].date()),
                                    initial_capital=CAPITAL,
                                    data_provider=data,
                                    index_result=index).run()

        assert "fx_policy 'EXACT_DAY' for the index" in caplog.text
        assert result.modelling_assumptions.fx_policy == "CARRY_FORWARD"


class TestTheCacheKeysOnTheDataTreatment:

    def test_another_fx_policy_is_another_key(self,
                                              tmp_path):
        folder = store.save(dataset.data_fetcher(), tmp_path / "store")
        sample = IndexDefinition(index_id="s", index_name="S",
                                 base_date="2023-01-03", base_value=100.0,
                                 currency="USD", eligibility_rules=[],
                                 weighting_scheme=EqualWeighted(),
                                 rebalancing_frequency="QUARTERLY",
                                 calendar="XNYS",
                                 universe_identifiers=["AAA", "BBB"])

        carry = index_cache.fingerprint(sample, store.load(folder), None,
                                        "2023-06-30")
        exact = index_cache.fingerprint(
            sample, store.load(folder, fx_policy="EXACT_DAY"), None,
            "2023-06-30")

        assert carry is not None and exact is not None
        assert carry != exact

    def test_the_assumptions_survive_the_cache(self,
                                               tmp_path):
        result = IndexCalculator(
            definition(), fetcher(),
            modelling_assumptions=ModellingAssumptions(fx_policy="EXACT_DAY")
        ).run(end_date=str(DAYS[-1].date()))
        cache = index_cache.IndexResultCache(tmp_path)
        cache.put("c" * 64, result)

        assert cache.get("c" * 64).modelling_assumptions == (
            result.modelling_assumptions)
