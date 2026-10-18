# tests/test_server_comparisons.py
"""BN-323: a run's comparisons, against its index and its benchmark.

Reuses test_server_backtest's two-name index and data: a plain run, a run
with one of the names as an external benchmark, and an active one.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from conftest import submit_backtest
from test_server_backtest import (
    TOKEN,
    auth,
    build_fetcher,
    index_document,
    run_backtest,
)

BENCHMARK = {"kind": "identifier", "id": "AAA"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("comparisons"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=index_document(),
                            headers=auth()).status_code == 200
        yield entered


@pytest.fixture(scope="module")
def plain(client) -> dict:
    return run_backtest(client, transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def benchmarked(client) -> dict:
    return run_backtest(client, transaction_cost_bps=5.0, benchmark=BENCHMARK)


class TestAgainstTheIndex:

    def test_it_agrees_with_the_run_s_tracking(self,
                                               plain):
        relative = plain["comparisons"]["index"]["relative"]
        metrics = plain["metrics"]

        assert relative["tracking_error"] == pytest.approx(metrics["tracking_error"])
        assert relative["relative_return"] == pytest.approx(metrics["tracking_difference"])

    def test_its_window_is_the_run_s(self,
                                     plain):
        index = plain["comparisons"]["index"]

        assert (index["id"], index["kind"]) == ("BT", "index")
        assert index["level"]["index"] == plain["level"]["index"]
        assert index["level"]["data"][0] == pytest.approx(100.0)
        assert index["start"] == plain["level"]["index"][0][:10]
        assert index["observations"] == len(plain["level"]["index"])

    def test_its_own_figures_are_the_index_s(self,
                                             plain):
        index = plain["comparisons"]["index"]
        level = index["level"]["data"]

        assert index["own"]["total_return"] == pytest.approx(level[-1] / level[0] - 1)
        assert 0.0 <= index["own"]["hit_rate_monthly"] <= 1.0

    def test_the_information_ratio_is_given(self,
                                            plain):
        relative = plain["comparisons"]["index"]["relative"]

        assert relative["information_ratio"] == pytest.approx(
            relative["annualised_relative_return"] / relative["tracking_error"])


class TestAgainstTheBenchmark:

    def test_without_one_there_is_none(self,
                                       plain):
        assert plain["comparisons"]["benchmark"] is None

    def test_with_one_it_is_compared_too(self,
                                         benchmarked):
        benchmark = benchmarked["comparisons"]["benchmark"]

        assert (benchmark["id"], benchmark["kind"]) == ("AAA", "identifier")
        assert benchmark["relative"]["information_ratio"] is not None
        assert benchmark["relative"]["relative_return"] == pytest.approx(
            benchmarked["metrics"]["total_return"] - benchmark["own"]["total_return"])

    def test_the_index_is_still_compared(self,
                                         benchmarked):
        assert benchmarked["comparisons"]["index"]["id"] == "BT"


class TestTheRunsOwnFigures:

    def test_its_metrics_carry_a_monthly_hit_rate(self,
                                                  plain):
        assert 0.0 <= plain["metrics"]["hit_rate_monthly"] <= 1.0

    def test_the_result_route_carries_them(self,
                                           client,
                                           plain):
        result = client.get("/strategies/BT/result", headers=auth()).json()

        assert result["comparisons"]["index"]["id"] == "BT"


class TestAnActiveStrategy:

    def test_its_index_comparison_is_its_benchmark_index(self,
                                                         client):
        """An active strategy's index is the one it is built against."""
        strategy = {"type": "active", "construction": {"type": "MinimumVariance", "params": {}},
                    "lookback_days": 20, "minimum_observations": 10}
        response = submit_backtest(client, "BT", {"strategy": strategy}, auth(),
                                   strategy_id="active")
        assert response.status_code == 202, response.text
        client.portal.call(client.app.state.jobs.drain)
        result = client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()["result"]

        index = result["comparisons"]["index"]
        assert index["id"] == "BT"
        assert index["relative"]["information_ratio"] is not None
