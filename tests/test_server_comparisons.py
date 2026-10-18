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



@pytest.fixture(scope="module")
def quarters(client,
             benchmarked) -> dict:
    """The benchmarked run by quarter."""
    response = client.get("/strategies/BT/periods", params={"frequency": "quarterly"},
                          headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


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


class TestPeriods:
    """BN-324: the latest run's figures by year, quarter and month."""

    def test_each_period_is_named_and_dated(self,
                                            quarters,
                                            benchmarked):
        periods = quarters["periods"]

        assert quarters["frequency"] == "quarterly" and quarters["strategy_id"] == "BT"
        assert periods[0]["period"].endswith(("Q1", "Q2", "Q3", "Q4"))
        assert periods[0]["start"] == benchmarked["level"]["index"][0][:10]
        assert periods[-1]["end"] == benchmarked["level"]["index"][-1][:10]

    def test_the_periods_compound_to_the_run(self,
                                             quarters,
                                             benchmarked):
        compounded = 1.0
        for period in quarters["periods"]:
            compounded *= 1 + period["strategy"]["total_return"]

        assert compounded - 1 == pytest.approx(benchmarked["metrics"]["total_return"])

    def test_each_period_compares_with_the_index_and_the_benchmark(self,
                                                                   quarters):
        for period in quarters["periods"]:
            assert period["index"]["relative"]["relative_return"] is not None
            assert period["benchmark"]["own"]["total_return"] is not None

    def test_the_index_s_periods_compound_to_its_whole(self,
                                                       quarters,
                                                       benchmarked):
        compounded = 1.0
        for period in quarters["periods"]:
            compounded *= 1 + period["index"]["own"]["total_return"]

        assert compounded - 1 == pytest.approx(
            benchmarked["comparisons"]["index"]["own"]["total_return"])

    @pytest.mark.parametrize("frequency", ["annual", "monthly"])
    def test_every_frequency_answers(self,
                                     client,
                                     benchmarked,
                                     frequency):
        response = client.get("/strategies/BT/periods", params={"frequency": frequency},
                              headers=auth())

        assert response.status_code == 200 and response.json()["periods"]

    def test_an_unknown_frequency_is_refused(self,
                                            client):
        response = client.get("/strategies/BT/periods", params={"frequency": "weekly"},
                              headers=auth())

        assert response.status_code == 422

    def test_without_a_benchmark_there_is_none(self,
                                               client):
        run_backtest(client)
        periods = client.get("/strategies/BT/periods", headers=auth()).json()["periods"]

        assert all(period["benchmark"] is None for period in periods)
        assert all(period["index"] is not None for period in periods)
