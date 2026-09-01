# tests/test_server_backtest.py
"""Contract tests for the backtest endpoint and its result consistency."""
import copy
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.backtests import annual_returns
from beacon.server.jobs import SUCCEEDED

TOKEN = "test-token-value"
# Spans a calendar-year boundary so the annual-return identity is exercised,
# but kept short: the calculator walks every business day and fetches prices
# per name per day, so a multi-year window makes this file dominate the suite.
START = "2023-10-02"
END = "2024-03-29"
DATES = pd.bdate_range(START, END)

# Two names on deterministic geometric paths over two calendar years, so the
# result spans a year boundary and the annual-return identity is exercised.
GROWTH = {"AAA": 1.30, "BBB": 0.85}
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0}
SHARES = 1_000


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data with no randomness."""
    span = len(DATES) - 1
    rows = []
    for name, total_growth in GROWTH.items():
        for index, date in enumerate(DATES):
            fraction = index / span
            rows.append({"IDENTIFIER": name,
                         "DATE": date,
                         "CLOSE": BASE_PRICE[name] * (total_growth ** fraction),
                         "VOLUME": 1_000_000,
                         "SHARES_OUTSTANDING": SHARES})

    reference = pd.DataFrame([
        {"IDENTIFIER": name,
         "DATE_FROM": "2020-01-01",
         "NAME": name,
         "CURRENCY": "USD",
         "EXCHANGE": "NYSE"}
        for name in GROWTH
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def index_document() -> dict:
    """An equal-weighted, quarterly-rebalanced index over the two names."""
    return copy.deepcopy({
        "id": "BT",
        "name": "Backtest Index",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "universe": {"universe_id": None, "identifiers": list(GROWTH)},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting",
                          "scheme": "EqualWeighted",
                          "params": {},
                          "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    })


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path):
    """Entered client holding the index definition and a data source."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path)
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=index_document(), headers=auth())
        assert created.status_code == 200, created.json()
        yield entered


@pytest.fixture(scope="module")
def module_client(tmp_path_factory):
    """A client shared across the read-only consistency tests.

    Those tests all interrogate the same completed backtest, and a backtest is
    expensive enough that running one per test would dominate the suite.
    """
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("backtest"))
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=index_document(), headers=auth())
        assert created.status_code == 200, created.json()
        yield entered


def run_backtest(client,
                 **body) -> dict:
    """Submit a backtest, wait for it, and return the job's result."""
    response = client.post("/beacon/BT/backtest", json=body or {}, headers=auth())
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]

    client.portal.call(client.app.state.jobs.drain)

    job = client.get(f"/jobs/{job_id}", headers=auth()).json()
    assert job["status"] == SUCCEEDED, job

    return job["result"]


@pytest.fixture(scope="module")
def result(module_client) -> dict:
    """One completed backtest, shared by every test that only reads it."""
    return run_backtest(module_client)


class TestSubmission:

    def test_returns_a_job_immediately(self,
                                       client):
        response = client.post("/beacon/BT/backtest", json={}, headers=auth())

        assert response.status_code == 202
        assert response.json()["kind"] == "backtest:BT"
        assert response.json()["job_id"]

    def test_unknown_index_fails_before_a_job_is_created(self,
                                                         client):
        """Better a clean 404 than a job that fails a moment later."""
        response = client.post("/beacon/ABSENT/backtest", json={}, headers=auth())

        assert response.status_code == 404
        assert client.get("/jobs", headers=auth()).json()["jobs"] == []

    def test_without_a_data_source_is_a_configuration_error(self,
                                                            tmp_path):
        config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
        with TestClient(create_app(config), raise_server_exceptions=False) as bare:
            bare.post("/indices", json=index_document(), headers=auth())

            response = bare.post("/beacon/BT/backtest", json={}, headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "CONFIGURATION_ERROR"

    def test_requires_authentication(self,
                                     client):
        assert client.post("/beacon/BT/backtest", json={}).status_code == 401

    def test_rejects_non_positive_capital(self,
                                          client):
        response = client.post("/beacon/BT/backtest",
                               json={"initial_capital": 0},
                               headers=auth())

        assert response.status_code == 422


class TestResultConsistency:
    """The acceptance criteria: every series reconciles against the others."""

    def test_level_compounds_exactly_from_the_returns(self,
                                                      result):
        level = result["level"]["data"]
        returns = result["returns"]["data"]

        rebuilt = [level[0]]
        for period_return in returns:
            rebuilt.append(rebuilt[-1] * (1 + period_return))

        assert len(rebuilt) == len(level)
        for expected, actual in zip(level, rebuilt, strict=True):
            assert math.isclose(actual, expected, rel_tol=1e-9)

    def test_drawdown_derives_from_the_level_series(self,
                                                    result):
        level = result["level"]["data"]
        drawdown = result["drawdown"]["data"]

        peak = level[0]
        for value, reported in zip(level, drawdown, strict=True):
            peak = max(peak, value)
            assert math.isclose(reported, value / peak - 1.0, abs_tol=1e-12)

    def test_drawdown_is_never_positive(self,
                                        result):
        assert all(value <= 1e-12 for value in result["drawdown"]["data"])

    def test_annual_returns_compound_to_the_total(self,
                                                  result):
        compounded = 1.0
        for annual in result["annual_returns"].values():
            compounded *= 1 + annual

        assert math.isclose(compounded - 1.0,
                            result["metrics"]["total_return"],
                            rel_tol=1e-9)

    def test_annual_returns_cover_every_year_in_the_series(self,
                                                           result):
        years = {label[:4] for label in result["level"]["index"]}

        assert set(result["annual_returns"]) == years

    def test_metrics_match_the_returned_series(self,
                                               result):
        """Recomputing from the payload must reproduce the reported metrics."""
        level = result["level"]["data"]
        returns = pd.Series(result["returns"]["data"])
        metrics = result["metrics"]

        total = level[-1] / level[0] - 1.0
        assert math.isclose(total, metrics["total_return"], rel_tol=1e-9)

        volatility = float(returns.std() * math.sqrt(252))
        assert math.isclose(volatility, metrics["volatility"], rel_tol=1e-9)

        assert math.isclose(min(result["drawdown"]["data"]),
                            metrics["max_drawdown"],
                            rel_tol=1e-9)

    def test_sharpe_is_consistent_with_the_other_metrics(self,
                                                         result):
        metrics = result["metrics"]

        if metrics["volatility"] > 0:
            expected = metrics["annualised_return"] / metrics["volatility"]
            assert math.isclose(expected, metrics["sharpe_ratio"], rel_tol=1e-9)


class TestSeriesShape:

    def test_level_starts_at_one_hundred(self,
                                         result):
        assert result["level"]["data"][0] == pytest.approx(100.0)

    def test_benchmark_starts_at_one_hundred(self,
                                             result):
        assert result["index_level"]["data"][0] == pytest.approx(100.0)

    def test_returns_is_one_shorter_than_level(self,
                                               result):
        assert len(result["returns"]["data"]) == len(result["level"]["data"]) - 1

    def test_index_is_iso_dates(self,
                                result):
        assert result["level"]["index"][0].startswith("2023-10")

    def test_tracking_metrics_are_present(self,
                                          result):
        """The backtest tracked an index, so tracking figures are meaningful."""
        assert result["metrics"]["tracking_error"] is not None
        assert result["metrics"]["tracking_difference"] is not None

    def test_costs_reduce_the_total_return(self,
                                           module_client):
        """Tightened once BN-85 (#104) landed.

        This previously asserted only that costs *changed* the result: the
        engine dropped a buy entirely when cash fell short, so a cost could
        remove one leg of a rebalance and leave the portfolio concentrated in
        whichever name performed better — making a costlier run look better.
        Orders are now sized down instead, so the intuitive direction holds.
        """
        free = run_backtest(module_client, transaction_cost_bps=0.0)
        costly = run_backtest(module_client, transaction_cost_bps=100.0)

        assert (costly["metrics"]["total_return"]
                < free["metrics"]["total_return"])


class TestProgressOverTheSocket:
    """The acceptance criterion: progress is visible over the WebSocket."""

    def test_progress_frames_reach_a_subscriber(self,
                                                client):
        with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
            client.post("/beacon/BT/backtest", json={}, headers=auth())
            # Let the job finish before reading. The socket subscribed before
            # submission, so every frame was published to its queue and is
            # waiting — this proves delivery without a receive that could
            # block forever if the job never ran.
            client.portal.call(client.app.state.jobs.drain)

            statuses = []
            progresses = []
            for _ in range(40):
                event = socket.receive_json()
                if event["type"] != "job":
                    continue
                statuses.append(event["status"])
                progresses.append(event["progress"])
                # Break on ANY terminal state, not just success: a failed job
                # would otherwise leave this blocking on a frame that never
                # arrives, turning a clear assertion failure into a hang.
                if event["status"] in {"succeeded", "failed", "cancelled"}:
                    break

        assert statuses[-1] == SUCCEEDED
        assert progresses[-1] == 1.0
        assert any(0.0 < value < 1.0 for value in progresses), progresses


class TestAnnualReturnsHelper:
    """The telescoping definition, checked directly."""

    def test_compounds_to_the_total_across_years(self):
        dates = pd.to_datetime(["2022-06-01", "2022-12-30",
                                "2023-06-01", "2023-12-29"])
        level = pd.Series([100.0, 110.0, 120.0, 132.0], index=dates)

        returns = annual_returns(level)
        compounded = math.prod(1 + value for value in returns.values())

        assert set(returns) == {"2022", "2023"}
        assert math.isclose(compounded - 1.0, 132.0 / 100.0 - 1.0, rel_tol=1e-12)

    def test_empty_series_has_no_years(self):
        assert annual_returns(pd.Series(dtype=float)) == {}

    def test_single_year_return_is_the_whole_move(self):
        dates = pd.to_datetime(["2023-01-03", "2023-12-29"])
        level = pd.Series([100.0, 125.0], index=dates)

        assert annual_returns(level)["2023"] == pytest.approx(0.25)


@pytest.fixture(scope="module")
def record(module_client,
           result):
    """The record of the run the module fixture already paid for.

    Depends on `result` so the backtest has definitely completed before the
    record is read.
    """
    response = module_client.get("/beacon/BT/record", headers=auth())
    assert response.status_code == 200, response.text
    return response.json()


class TestTheRecord:
    """BN-158: `GET /beacon/{index_id}/record` — the books, finally served.

    Found by the beacon-ui session after its BN-155 migration: the nested
    `BacktestResultSummary` was defined, exported and tested, but referenced
    by no route — the OpenAPI spec was byte-identical before and after the
    reshape. The record is captured at job completion, because the library
    BacktestResult exists only inside the job.
    """


    def test_the_portfolio_book_arrives(self,
                                        record):
        book = record["portfolio"]

        assert book["portfolio_id"]
        assert book["positions_total"] > 0
        assert len(book["positions"]["data"]) > 0
        assert book["weights_dates_total"] > 0

    def test_nav_opens_with_the_capital(self,
                                        record):
        """The day-zero row survives serialisation: the first NAV value is
        the initial capital, dated before the first trading day."""
        book = record["portfolio"]

        assert book["nav"]["data"][0] == pytest.approx(
            book["initial_capital"])

    def test_the_index_book_is_present_and_the_benchmark_null(self,
                                                              record):
        """This run tracked an index and was given no benchmark. Null and
        present must both survive the wire — a client tells "not measured"
        from "measured and empty" by exactly this."""
        assert record["index"] is not None
        assert len(record["index"]["levels"]["data"]) > 0
        assert record["benchmark"] is None

    def test_a_never_backtested_index_404s_with_the_pointer(self,
                                                            module_client):
        created = module_client.post(
            "/indices", json={**index_document(), "id": "NEVER-RUN",
                              "name": "Never run"}, headers=auth())
        assert created.status_code == 200

        response = module_client.get("/beacon/NEVER-RUN/record",
                                     headers=auth())

        assert response.status_code == 404
        assert "backtest" in response.json()["error"]["message"]

    def test_it_requires_authentication(self,
                                        module_client):
        assert module_client.get("/beacon/BT/record").status_code == 401

    def test_the_route_is_in_the_spec(self,
                                      module_client):
        """The finding that started this: the spec did not change when the
        payload did. Now it must contain the path."""
        paths = module_client.get("/openapi.json").json()["paths"]

        assert "/beacon/{index_id}/record" in paths

    def test_deleting_the_index_deletes_the_record(self,
                                                   module_client):
        """BN-157's cascade extends to the record store, or the delete leaves
        exactly the orphan it exists to prevent.

        A dedicated index rather than the shared BT one: consuming the module
        fixture's index would break sibling tests under randomised ordering.
        The record is seeded through the store the job writes to, which is
        the layer the cascade must clean.
        """
        created = module_client.post(
            "/indices", json={**index_document(), "id": "DOOMED",
                              "name": "Doomed"}, headers=auth())
        assert created.status_code == 200

        # A skeletal record: enough for the store, deliberately not enough
        # for the route's response model -- this test is about the cascade,
        # and the pre-check that guards against a vacuous pass reads the
        # store directly.
        records = module_client.app.state.backtest_record_store
        records.write("DOOMED", {"portfolio": {"portfolio_id": "DOOMED"}})
        assert records.read("DOOMED") is not None

        deleted = module_client.delete("/indices/DOOMED", headers=auth())
        assert deleted.status_code == 204

        assert records.read("DOOMED") is None
        response = module_client.get("/beacon/DOOMED/record", headers=auth())

        assert response.status_code == 404
