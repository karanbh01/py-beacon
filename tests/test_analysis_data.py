# tests/test_analysis_data.py
"""BN-369: RunData, everything a run's analysis reads, from a finished run or
a stored one.

A small market-cap run with costs and a benchmark index, written to a
temporary folder and read back; and records as runs stored before this kept
them.
"""
import json

import numpy as np
import pandas as pd
import pytest

from beacon.analysis import RunData
from beacon.analysis.data import BENCHMARK, INDEX, RUN_FILE, RUN_VERSION
from beacon.analysis.panels import (
    CHARACTERISTICS,
    COSTS,
    FUND_WEIGHTS,
    NAME_RETURNS,
    write_panels,
)
from beacon.exceptions import ConfigurationError


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index, run

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return run(index(), transaction_cost_bps=5.0, benchmark=benchmark)


@pytest.fixture(scope="module")
def data(result):
    return RunData.from_result(result)


def payload(series: pd.Series) -> dict:
    """A series as a backtest record keeps it."""
    return {"index": [str(day.date()) for day in series.index],
            "data": [float(value) for value in series]}


class TestFromAResult:

    def test_its_returns_are_the_metrics_own(self,
                                             result,
                                             data):
        assert data.returns.equals(result.get_returns())
        assert data.nav.equals(result.trading_nav)

    def test_it_has_the_comparators_levels(self,
                                           result,
                                           data):
        assert set(data.comparators) == {INDEX, BENCHMARK}
        assert data.comparators[BENCHMARK].equals(result.benchmark.levels)

    def test_it_has_every_table_and_misses_none(self,
                                                data):
        assert {FUND_WEIGHTS, NAME_RETURNS, COSTS, CHARACTERISTICS} <= set(data.tables)
        assert data.missing == {}

    def test_it_carries_the_run_s_assumptions(self,
                                              result,
                                              data):
        assert data.currency == result.currency
        assert data.initial_capital == result.portfolio.initial_capital
        assert data.periods_per_year == 252


class TestTheRoundTrip:

    def test_a_stored_run_reads_back_exactly(self,
                                             data,
                                             tmp_path):
        data.write(tmp_path / "run")
        again = RunData.from_record(tmp_path / "run")

        for name in ("currency", "initial_capital", "periods_per_year", "risk_free_rate"):
            assert getattr(again, name) == getattr(data, name), name
        for name in ("returns", "nav", "units"):
            assert getattr(again, name).equals(getattr(data, name)), name
        assert set(again.comparators) == set(data.comparators)
        assert all(again.comparators[name].equals(data.comparators[name])
                   for name in data.comparators)
        assert set(again.tables) == set(data.tables)
        assert all(again.tables[name].equals(data.tables[name]) for name in data.tables)
        assert again.missing == data.missing

    def test_writing_again_replaces_the_run(self,
                                            data,
                                            tmp_path):
        data.write(tmp_path / "run")
        data.write(tmp_path / "run")

        assert RunData.from_record(tmp_path / "run").nav.equals(data.nav)


class TestRecordsKeptBefore:

    def record(self,
               result) -> dict:
        return {"currency": result.currency,
                "portfolio": {"initial_capital": result.portfolio.initial_capital,
                              "nav": payload(result.trading_nav)},
                "index": {"target": {"levels": payload(result.index.target.levels)},
                          "optimised": None},
                "benchmark": None,
                "settings": {"modelling_assumptions": {"periods_per_year": 252,
                                                       "risk_free_rate": 0.01}}}

    def test_one_without_a_folder_loads_its_series_from_the_record(self,
                                                                  result,
                                                                  tmp_path):
        old = RunData.from_record(tmp_path / "never", self.record(result))

        assert np.allclose(old.nav.to_numpy(), result.trading_nav.to_numpy())
        assert set(old.comparators) == {INDEX}
        assert old.tables == {}
        assert old.risk_free_rate == 0.01

    def test_it_says_what_it_lacks_and_why(self,
                                           result,
                                           tmp_path):
        old = RunData.from_record(tmp_path / "never", self.record(result))

        assert "0.10.0" in old.missing[NAME_RETURNS]
        assert "worked out from the NAV" in old.missing["returns"]
        assert old.units.empty and "units" in old.missing

    def test_its_returns_are_worked_out_from_the_nav(self,
                                                     result,
                                                     tmp_path):
        old = RunData.from_record(tmp_path / "never", self.record(result))
        nav = result.trading_nav

        assert old.returns.iloc[0] == pytest.approx(nav.iloc[0] / result.portfolio.initial_capital
                                                    - 1.0)
        assert old.returns.iloc[5] == pytest.approx(nav.iloc[5] / nav.iloc[4] - 1.0)

    def test_one_with_the_first_tables_keeps_them(self,
                                                  result,
                                                  tmp_path):
        write_panels(tmp_path / "run", {FUND_WEIGHTS: result.portfolio.weights})
        old = RunData.from_record(tmp_path / "run", self.record(result))

        assert set(old.tables) == {FUND_WEIGHTS}
        assert FUND_WEIGHTS not in old.missing and COSTS in old.missing


class TestRefusals:

    def test_no_folder_and_no_record(self,
                                     tmp_path):
        with pytest.raises(ConfigurationError, match="no record"):
            RunData.from_record(tmp_path / "never")

    def test_a_run_kept_by_a_newer_version(self,
                                           data,
                                           tmp_path):
        data.write(tmp_path / "run")
        path = tmp_path / "run" / RUN_FILE
        stored = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**stored, "version": RUN_VERSION + 1}), encoding="utf-8")

        with pytest.raises(ConfigurationError, match="newer"):
            RunData.from_record(tmp_path / "run")


class TestPackedTables:
    """BN-403: a stored run keeps its closes packed and works its name
    returns out again when first asked for."""

    def test_the_name_returns_are_not_written(self,
                                              data,
                                              tmp_path):
        data.write(tmp_path / "run")

        assert not (tmp_path / "run" / f"{NAME_RETURNS}.parquet").exists()
        assert (tmp_path / "run" / "closes_packed.parquet").exists()

    def test_they_are_worked_out_only_when_asked_for(self,
                                                     data,
                                                     tmp_path,
                                                     monkeypatch):
        from beacon.analysis import data as module

        data.write(tmp_path / "run")
        calls = []
        original = module.name_returns_from
        monkeypatch.setattr(module, "name_returns_from",
                            lambda *args: calls.append(1) or original(*args))

        again = RunData.from_record(tmp_path / "run")
        assert calls == [] and NAME_RETURNS in again.tables

        first = again.tables[NAME_RETURNS]
        second = again.tables[NAME_RETURNS]

        assert calls == [1] and first is second
        assert first.equals(data.tables[NAME_RETURNS])

    def test_the_weights_are_not_written_but_their_holdings_are(self,
                                                                data,
                                                                tmp_path):
        data.write(tmp_path / "run")
        folder = tmp_path / "run"

        for name in (FUND_WEIGHTS, "index_weights", "benchmark_weights"):
            assert not (folder / f"{name}.parquet").exists(), name
            assert (folder / f"{name}_holdings.parquet").exists(), name

    def test_the_fund_s_weights_need_no_patch(self,
                                              data):
        assert data.tables[FUND_WEIGHTS + "_patch"].empty

    def test_rebuilt_weights_are_the_recorded_ones_exactly(self,
                                                           result,
                                                           data,
                                                           tmp_path):
        data.write(tmp_path / "run")
        again = RunData.from_record(tmp_path / "run")

        fund = again.tables[FUND_WEIGHTS]
        index = again.tables["index_weights"]

        np.testing.assert_array_equal(fund.to_numpy(), result.portfolio.weights.to_numpy())
        np.testing.assert_array_equal(index.to_numpy(), result.index.target.weights.to_numpy())
        assert index.dtypes.iloc[0] == result.index.target.weights.dtypes.iloc[0]

    def test_the_latest_weights_alone_are_those_rows_of_the_whole_table(self,
                                                                       data,
                                                                       tmp_path):
        data.write(tmp_path / "run")
        again = RunData.from_record(tmp_path / "run")

        for name in (FUND_WEIGHTS, "index_weights"):
            for rows in (1, 7, 40, 10_000):
                latest = again.latest(name, rows)
                whole = RunData.from_record(tmp_path / "run").tables[name].tail(rows)

                np.testing.assert_array_equal(latest.to_numpy(), whole.to_numpy())
                assert latest.index.equals(whole.index)
                assert list(latest.columns) == list(whole.columns)
