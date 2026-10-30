# tests/test_analysis_rerun.py
"""BN-399: a stored run's analysis asked for again, and its factor work kept
with it.

test_strategy_futures' eight-name index held through 2024 with a benchmark,
analysed with a factsheet's and the risk report's cores, stored, and asked
again: as it was, with a statistic added, with other choices, and as a run
stored before a table was kept.
"""
import dataclasses
import json

import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.options import AnalysisOptions
from beacon.analysis.reports.rerun import rerun, reusable
from beacon.analysis.reports.statistics import RUN_PREDATES
from beacon.analysis.request import AnalysisRequest, keep_analysis_work
from beacon.analysis.risk import factor_store
from beacon.analysis.risk.factors import FactorAnalytics
from beacon.backtest import Backtest
from beacon.strategy import Factor
from beacon.strategy.factors import default_factors
from beacon.strategy.signals import FunctionSignal

ASKED = AnalysisRequest(reports={"factsheet": "core", "risk": "core"})


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """The run, its live block, and the folder it is stored in with its
    factor work."""
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)
    result = Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                      transaction_cost_bps=5.0, benchmark=benchmark).run(
        index(), start=START, end=END, analysis=ASKED)
    folder = tmp_path_factory.mktemp("rerun") / "run"
    result.run_data.write(folder)
    assert keep_analysis_work(result, folder)

    return result, folder, DATA


def statistics_of(block: dict) -> str:
    return json.dumps(block["statistics"], sort_keys=True)


class TestTheFactorWorkKept:

    def test_it_is_kept_beside_the_run(self,
                                       stored):
        _, folder, _ = stored

        assert (folder / factor_store.FOLDER / factor_store.KEY_FILE).exists()

    def test_a_stored_run_reads_it_rather_than_working_it_out(self,
                                                              stored,
                                                              monkeypatch):
        result, folder, source = stored
        live = result.__dict__["_analysis_context"].factor_analytics
        monkeypatch.setattr(FactorAnalytics, "_scored",
                            lambda self, day: pytest.fail("worked out again"))

        restored = Context(RunData.from_record(folder), None, source, None, folder).factor_analytics

        for day in live.dates:
            pd.testing.assert_frame_equal(restored.exposures(day), live.exposures(day),
                                          check_exact=True)

        pd.testing.assert_frame_equal(restored.factor_returns(), live.factor_returns(),
                                      check_exact=True, check_freq=False)

    def test_other_factors_work_them_out_again(self,
                                               stored):
        _, folder, source = stored
        fewer = FactorAnalytics(RunData.from_record(folder), source, default_factors()[:2])

        assert not factor_store.restore(fewer, folder)

    def test_a_factor_read_from_a_function_is_never_kept(self):
        constant = Factor("constant", FunctionSignal(lambda name, date, fetched: 1.0))

        assert factor_store.fingerprint([constant], None) is None
        assert factor_store.fingerprint(default_factors(), None) is not None


class TestRerun:

    def test_the_run_s_own_analysis_comes_back_as_it_was(self,
                                                         stored):
        result, folder, source = stored

        again = rerun(RunData.from_record(folder), ASKED, source, folder)

        assert statistics_of(again) == statistics_of(result.analysis_block)

    def test_a_statistic_added_is_the_only_one_worked_out(self,
                                                         stored,
                                                         monkeypatch):
        result, folder, source = stored
        added = AnalysisRequest(reports={"factsheet": "core", "risk": "core"},
                                add={"factsheet": ["sortino_ratio"]})
        worked = []
        from beacon.analysis import request as module
        original = module._work_out
        monkeypatch.setattr(module, "_work_out",
                            lambda statistic, *rest: worked.append(statistic.id)
                            or original(statistic, *rest))

        merged = rerun(RunData.from_record(folder), added, source, folder,
                       previous=result.analysis_block)

        assert worked == ["sortino_ratio"]
        assert set(result.analysis_block["statistics"]) < set(merged["statistics"])
        assert "sortino_ratio" in merged["statistics"]

    def test_a_view_now_rolling_is_worked_out_again(self,
                                                    stored):
        result, _, _ = stored
        rolled = AnalysisRequest(reports={"factsheet": "core", "risk": "core"},
                                 time_views={"factsheet": {"sharpe_ratio": "both"}})

        assert "sharpe_ratio" not in reusable(result.analysis_block, rolled)
        assert "total_return" in reusable(result.analysis_block, rolled)

    def test_other_choices_start_afresh(self,
                                        stored):
        result, folder, source = stored
        later = AnalysisRequest(reports={"performance": "core"}, options=AnalysisOptions(
            window=AnalysisSettings(start=pd.Timestamp("2024-07-01"))))

        fresh = rerun(RunData.from_record(folder), later, source, folder,
                      previous=result.analysis_block)

        assert set(fresh["reports"]) == {"performance"}
        assert fresh["statistics"]["total_return"] != result.analysis_block["statistics"][
            "total_return"]
        assert "factor_betas" not in fresh["statistics"]


class TestARunStoredBeforeATable:

    def test_a_statistic_reading_it_says_the_run_predates_it(self,
                                                             stored):
        _, folder, source = stored
        run = RunData.from_record(folder)
        older = dataclasses.replace(run, tables={name: run.tables[name] for name in run.tables
                                                 if name != "characteristics"},
                                    missing={"characteristics": "kept before 0.10.0"})

        block = rerun(older, AnalysisRequest(reports={}, statistics=["characteristics"]), source)

        assert block["unavailable"]["characteristics"]["code"] == RUN_PREDATES
