# tests/test_server_strategy_runs.py
"""BN-313: backtests of stored strategies, with runs kept per strategy.

Every backtest runs a stored strategy, its runs are filed under the strategy's
id, and every view of a run is read by strategy. The optimiser and risk models
take an index or a strategy as their target, and a factsheet describes a
strategy's run.
"""
import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.analysis.panels import FUND_WEIGHTS, read_panels, write_panels
from beacon.server import ServerConfig, create_app
from beacon.server.jobs import SUCCEEDED
from beacon.server.store import DocumentStore
from beacon.server.strategy_runs import latest_run
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "canon"
FRESH_ID = "fresh"
SET_ID = "loose"
FACTSHEET = "FACTSHEET-A4"

START = "2023-01-02"
END = "2023-12-29"

# Three ways of holding the one index: as defined, through a replication, and
# as an active strategy measured against it.
PLAIN = {"type": "index"}
SAMPLED = {"type": "index_tracking",
           "replication": {"type": "SampledReplication", "params": {"holdings": 3}}}
ACTIVE = {"type": "active",
          "construction": {"type": "MinimumVariance", "params": {}},
          "lookback_days": 40, "minimum_observations": 20}
STRATEGIES = {"plain": PLAIN, "sampled": SAMPLED, "active": ACTIVE}


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def index_document(index_id: str = INDEX_ID) -> dict:
    """An equal-weighted, quarterly-rebalanced index over the canonical universe."""
    return {
        "id": index_id,
        "name": f"Index {index_id}",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "calendar": "XNYS",
        "rebalancing_frequency": "QUARTERLY",
        "universe": {"universe_id": None, "identifiers": list(dataset.UNIVERSE)},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting", "scheme": "EqualWeighted", "params": {}},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }


def strategy_document(strategy_id: str,
                      strategy: dict | None = None,
                      index_id: str = INDEX_ID) -> dict:
    """A stored strategy holding *index_id*."""
    return {"id": strategy_id,
            "name": f"Strategy {strategy_id}",
            "index_id": index_id,
            "strategy": strategy or PLAIN}


def constraint_set() -> dict:
    """Full investment and a position cap loose enough for any target here."""
    return {"id": SET_ID,
            "name": "Loose",
            "constraints": [
                {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}},
                {"id": "cap", "type": "PositionBounds",
                 "params": {"minimum": 0.0, "maximum": 0.6}}]}


def finished(client,
             response) -> dict:
    """The job a submission started, once it has run."""
    assert response.status_code == 202, response.text

    client.portal.call(client.app.state.jobs.drain)

    return client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()


def persist_result(client,
                   kind: str,
                   result: dict) -> None:
    """Write a succeeded result the way a finished job would."""
    job_id = str(uuid.uuid4())
    client.app.state.jobs._results.write(job_id, {
        "job_id": job_id, "kind": kind, "status": SUCCEEDED, "progress": 1.0,
        "message": "done", "result": result, "error": None,
        "completed_at": datetime.now(UTC).isoformat()})


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Three strategies on one index, each backtested once, and an index no
    strategy holds."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=dataset.data_fetcher(),
                          storage_root=tmp_path_factory.mktemp("strategy-runs"))

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        for index_id in (INDEX_ID, FRESH_ID):
            assert started.put(f"/indices/{index_id}", json=index_document(index_id),
                               headers=auth()).status_code == 200
        assert started.put(f"/optimise/constraint-sets/{SET_ID}", json=constraint_set(),
                           headers=auth()).status_code == 200

        for strategy_id, strategy in STRATEGIES.items():
            assert started.post("/strategies", json=strategy_document(strategy_id, strategy),
                                headers=auth()).status_code == 200
            job = finished(started, started.post(f"/strategies/{strategy_id}/backtest",
                                                 json={"start": START, "end": END},
                                                 headers=auth()))
            assert job["status"] == SUCCEEDED, job

        yield started


@pytest.fixture
def bare(tmp_path):
    """A server with no data: document work only."""
    config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        assert started.put(f"/indices/{INDEX_ID}", json=index_document(),
                           headers=auth()).status_code == 200
        yield started


def run_of(client,
           strategy_id: str) -> dict:
    """A strategy's latest run, as the server serves it."""
    run = latest_run(client.app.state.jobs, strategy_id)
    assert run is not None

    return run


class TestRunsArePerStrategy:

    def test_two_strategies_on_one_index_keep_their_own_runs(self,
                                                             client):
        plain = run_of(client, "plain")
        sampled = run_of(client, "sampled")

        assert plain["settings"]["strategy"]["type"] == "index"
        assert sampled["settings"]["strategy"]["type"] == "index_tracking"
        assert sampled["replication"] and not plain["replication"]

    def test_each_run_says_which_strategy_it_is_of(self,
                                                   client):
        settings = run_of(client, "sampled")["settings"]

        assert (settings["strategy_id"], settings["strategy_name"], settings["index_id"]) == (
            "sampled", "Strategy sampled", INDEX_ID)

    def test_each_keeps_its_own_record(self,
                                       client):
        for strategy_id in STRATEGIES:
            record = client.get(f"/strategies/{strategy_id}/record", headers=auth())

            assert record.status_code == 200, record.text
            assert record.json()["settings"]["strategy_id"] == strategy_id

    def test_the_views_name_the_strategy_and_its_index(self,
                                                       client):
        for path in ("overview", "weights", "attribution", "assets/AAA"):
            payload = client.get(f"/strategies/sampled/{path}", headers=auth()).json()

            assert (payload["strategy_id"], payload["index_id"]) == ("sampled", INDEX_ID), path

    def test_the_overview_is_named_after_the_strategy(self,
                                                      client):
        payload = client.get("/strategies/active/overview", headers=auth()).json()

        assert payload["name"] == "Strategy active"

    def test_a_strategy_nobody_has_backtested_has_no_views(self,
                                                           bare):
        bare.post("/strategies", json=strategy_document("idle"), headers=auth())

        response = bare.get("/strategies/idle/overview", headers=auth())

        assert response.status_code == 404
        assert "POST /strategies/idle/backtest" in response.json()["error"]["message"]


class TestTheRunsTables:
    """BN-361: the tables too large for the record, beside it."""

    def test_each_record_keeps_its_fund_s_daily_weights_beside_it(self,
                                                                 client):
        store = client.app.state.backtest_record_store

        for strategy_id in STRATEGIES:
            tables = read_panels(store.panels_directory(strategy_id))
            record = client.get(f"/strategies/{strategy_id}/record", headers=auth()).json()
            weights = tables[FUND_WEIGHTS]

            assert len(weights) == record["portfolio"]["weights_dates_total"]
            assert weights.sum(axis=1).max() <= 1.0 + 1e-9

    def test_a_record_kept_before_them_still_opens(self,
                                                  client):
        store = client.app.state.backtest_record_store
        record = store.read("plain")
        store.write("before-tables", record)

        assert read_panels(store.panels_directory("before-tables")) == {}
        assert store.read("before-tables") == {**record}


class TestTheLatestResult:
    """BN-320: the run payload by strategy, so a client need not hold the job's id."""

    def test_it_is_the_latest_run_s_payload(self,
                                            client):
        response = client.get("/strategies/sampled/result", headers=auth())

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["settings"]["strategy_id"] == "sampled"
        assert payload["level"] == run_of(client, "sampled")["level"]
        assert payload["replication"]

    def test_it_is_not_found_until_the_strategy_has_run(self,
                                                        bare):
        bare.post("/strategies", json=strategy_document("idle"), headers=auth())

        response = bare.get("/strategies/idle/result", headers=auth())

        assert response.status_code == 404
        assert "POST /strategies/idle/backtest" in response.json()["error"]["message"]

    def test_an_unknown_strategy_is_not_found(self,
                                              client):
        assert client.get("/strategies/nope/result", headers=auth()).status_code == 404

    def test_a_run_filed_under_an_index_is_not_served(self,
                                                     bare):
        bare.post("/strategies", json=strategy_document(INDEX_ID), headers=auth())
        persist_result(bare, f"backtest:{INDEX_ID}",
                       {"level": {"index": [], "data": []},
                        "settings": {"strategy": {"type": "index"}}})

        assert bare.get(f"/strategies/{INDEX_ID}/result", headers=auth()).status_code == 404


class TestRunsFromBeforeTheChange:

    def test_a_run_filed_under_an_index_is_not_a_strategy_s(self,
                                                            bare):
        """A run saved before BN-313 carries no strategy id, and is not served
        for a strategy that happens to share the index's id."""
        bare.post("/strategies", json=strategy_document(INDEX_ID), headers=auth())
        persist_result(bare, f"backtest:{INDEX_ID}",
                       {"level": {"index": [], "data": []},
                        "settings": {"strategy": {"type": "index"}}})

        assert latest_run(bare.app.state.jobs, INDEX_ID) is None
        assert bare.get(f"/strategies/{INDEX_ID}/overview",
                        headers=auth()).status_code == 404

    def test_records_saved_under_index_ids_are_not_read(self,
                                                        bare):
        """They sit in the collection runs used before BN-313, which nothing
        reads any more."""
        old = DocumentStore("backtest_records", root=bare.app.state.config.storage_root)
        old.write(INDEX_ID, {"run_at": "2025-01-01T00:00:00+00:00"})

        body = bare.get("/strategies/runs", headers=auth()).json()

        assert body["backtests"] == [] and body["skipped"] == 0


class TestTheRunsListing:

    def test_every_strategy_s_run_is_listed_with_what_ran(self,
                                                          client):
        rows = client.get("/strategies/runs", headers=auth()).json()["backtests"]
        by_id = {row["strategy_id"]: row for row in rows}

        assert set(by_id) == set(STRATEGIES)
        assert by_id["active"]["strategy"] == "active"
        assert by_id["sampled"]["strategy_name"] == "Strategy sampled"
        assert {row["index_id"] for row in rows} == {INDEX_ID}

    def test_the_newest_run_comes_first(self,
                                        client):
        rows = client.get("/strategies/runs", headers=auth()).json()["backtests"]

        assert [row["run_at"] for row in rows] == sorted((row["run_at"] for row in rows),
                                                         reverse=True)


class TestDeletingAStrategy:

    def test_its_runs_and_record_go_with_it(self,
                                            bare):
        bare.post("/strategies", json=strategy_document("doomed"), headers=auth())
        persist_result(bare, "backtest:doomed", {"settings": {"strategy_id": "doomed"}})
        records = bare.app.state.backtest_record_store
        records.write("doomed", {"run_at": None})
        write_panels(records.panels_directory("doomed"), {FUND_WEIGHTS: pd.DataFrame(
            {"AAA": [1.0]}, index=pd.DatetimeIndex(["2023-01-03"]))})

        deleted = bare.delete("/strategies/doomed", headers=auth())

        assert deleted.json() == {"strategy_id": "doomed",
                                  "backtest_record_deleted": True,
                                  "backtest_results_deleted": 1}
        assert bare.app.state.jobs.latest_result("backtest:doomed") is None
        assert bare.get("/strategies/doomed/overview", headers=auth()).status_code == 404
        assert not records.panels_directory("doomed").exists()

    def test_another_strategy_s_runs_survive(self,
                                             bare):
        """The forget is exact-kind, not prefix: deleting `core` must not take
        `core-hedged`'s results with it."""
        for strategy_id in ("core", "core-hedged"):
            bare.post("/strategies", json=strategy_document(strategy_id), headers=auth())
            persist_result(bare, f"backtest:{strategy_id}",
                           {"settings": {"strategy_id": strategy_id}})

        bare.delete("/strategies/core", headers=auth())

        assert bare.app.state.jobs.latest_result("backtest:core") is None
        assert bare.app.state.jobs.latest_result("backtest:core-hedged") is not None

    def test_a_strategy_never_run_deletes_cleanly(self,
                                                  bare):
        bare.post("/strategies", json=strategy_document("idle"), headers=auth())

        assert bare.delete("/strategies/idle", headers=auth()).json() == {
            "strategy_id": "idle", "backtest_record_deleted": False,
            "backtest_results_deleted": 0}


class TestTheBacktestBody:

    @pytest.mark.parametrize("field", ["strategy", "implementation"])
    def test_what_the_strategy_holds_is_refused_in_the_body(self,
                                                            client,
                                                            field):
        """It belongs to the stored strategy, so sending it here is a mistake
        worth hearing about rather than a setting silently ignored."""
        response = client.post("/strategies/plain/backtest",
                               json={field: {"type": "index"} if field == "strategy" else {}},
                               headers=auth())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_an_unknown_strategy_cannot_be_backtested(self,
                                                      client):
        assert client.post("/strategies/nope/backtest", json={},
                           headers=auth()).status_code == 404

    def test_a_strategy_whose_index_has_gone_cannot_be_backtested(self,
                                                                  client):
        client.put(f"/indices/{FRESH_ID}2", json=index_document(f"{FRESH_ID}2"),
                   headers=auth())
        client.post("/strategies", json=strategy_document("orphan", index_id=f"{FRESH_ID}2"),
                    headers=auth())
        client.app.state.index_store.delete(f"{FRESH_ID}2")

        response = client.post("/strategies/orphan/backtest", json={}, headers=auth())

        assert response.status_code == 404
        assert f"{FRESH_ID}2" in response.json()["error"]["message"]

    def test_validate_takes_the_strategy_s_own_settings(self,
                                                        client):
        settings = client.post("/strategies/sampled/backtest/validate", json={},
                               headers=auth()).json()["settings"]

        assert settings["strategy"]["replication"]["type"] == "SampledReplication"


class TestCompare:

    def test_it_compares_strategies_by_id(self,
                                          client):
        payload = client.get("/strategies/compare", params={"ids": ["plain", "sampled"]},
                             headers=auth()).json()

        assert payload["strategy_ids"] == ["plain", "sampled"]
        assert [entry["strategy_id"] for entry in payload["entries"]] == ["plain", "sampled"]

    def test_a_strategy_without_a_run_is_a_404(self,
                                               client):
        client.post("/strategies", json=strategy_document("unrun"), headers=auth())

        response = client.get("/strategies/compare", params={"ids": ["plain", "unrun"]},
                              headers=auth())

        assert response.status_code == 404


class TestOptimisingATarget:

    def solved(self,
               client,
               **target) -> dict:
        """A finished optimisation against *target*."""
        job = finished(client, client.post("/optimise/runs",
                                           json={"constraint_set_id": SET_ID,
                                                 "start": START, "end": END, **target},
                                           headers=auth()))
        assert job["status"] == SUCCEEDED, job

        return job["result"]

    def test_an_index_needs_no_backtest(self,
                                        client):
        """No strategy holds this index, so nothing has been run on it."""
        result = self.solved(client, index_id=FRESH_ID)
        targets = {row["asset_id"]: row["target_weight"] for row in result["weights"]}

        assert result["index_id"] == FRESH_ID and result["strategy_id"] is None
        assert targets == pytest.approx(dict.fromkeys(dataset.UNIVERSE,
                                                      1 / len(dataset.UNIVERSE)))

    def test_an_index_s_target_date_is_its_latest_rebalance(self,
                                                            client):
        """Without `as_of`, the index as it stands in the data; `end` bounds
        only the window the risk model is estimated over."""
        level = client.get(f"/indices/{FRESH_ID}/level", headers=auth()).json()

        result = self.solved(client, index_id=FRESH_ID)

        assert result["target_date"] == level["rebalances"][-1]["date"]

    def test_a_strategy_s_target_is_what_its_run_chose(self,
                                                       client):
        run = run_of(client, "sampled")
        result = self.solved(client, strategy_id="sampled")
        targets = {row["asset_id"]: row["target_weight"] for row in result["weights"]}

        assert result["strategy_id"] == "sampled" and result["index_id"] == INDEX_ID
        assert result["target_date"] == run["replication"][-1]["date"]
        assert {name: weight for name, weight in targets.items() if weight} == pytest.approx(
            {name: weight for name, weight in run["replication"][-1]["weights"].items()
             if weight})

    def test_an_active_strategy_s_target_is_its_construction_s(self,
                                                               client):
        """How close a portfolio can get to the active one under the
        constraints."""
        run = run_of(client, "active")
        result = self.solved(client, strategy_id="active")
        targets = {row["asset_id"]: row["target_weight"] for row in result["weights"]}
        chosen = run["active"][-1]

        assert result["target_date"] == chosen["date"]
        assert targets == pytest.approx({name: chosen["weights"].get(name, 0.0)
                                         for name in targets})

    def test_as_of_picks_the_rebalance_in_force(self,
                                                client):
        run = run_of(client, "plain")
        first, second = run["rebalances"][0]["date"], run["rebalances"][1]["date"]

        result = self.solved(client, strategy_id="plain", as_of=first)

        assert first <= result["target_date"] < second


class TestTheRiskOfATarget:

    def estimated(self,
                  client,
                  model_id: str,
                  **source) -> dict:
        """A finished risk model over a target's names."""
        job = finished(client, client.post(f"/risk-models/{model_id}/estimate",
                                           json={"start": START, "end": END, **source},
                                           headers=auth()))
        assert job["status"] == SUCCEEDED, job

        return client.get(f"/risk-models/{model_id}", headers=auth()).json()

    def test_an_index_s_names_are_its_constituents(self,
                                                   client):
        model = self.estimated(client, "of-index", index_id=FRESH_ID)

        assert set(model["asset_ids"]) == set(dataset.UNIVERSE)

    def test_a_strategy_s_names_are_what_it_held(self,
                                                 client):
        run = run_of(client, "sampled")
        model = self.estimated(client, "of-strategy", strategy_id="sampled")

        assert set(model["asset_ids"]) == set(run["replication"][-1]["weights"])


class TestTheFactsheet:

    def test_it_describes_a_strategy_s_run_and_is_named_after_it(self,
                                                                 client):
        job = finished(client, client.post("/reports/render",
                                           json={"template_id": FACTSHEET,
                                                 "strategy_id": "active"},
                                           headers=auth()))

        assert job["status"] == SUCCEEDED, job
        assert "Strategy active" in job["result"]["name"]
        assert job["result"]["strategy_id"] == "active"


class TestReservedIds:

    @pytest.mark.parametrize("strategy_id", ["compare", "runs"])
    def test_a_name_beside_the_strategy_routes_is_refused(self,
                                                          bare,
                                                          strategy_id):
        """`/strategies/compare` and `/strategies/runs` are routes, so no
        strategy may take their names."""
        response = bare.post("/strategies", json=strategy_document(strategy_id),
                             headers=auth())

        assert response.status_code == 422
        assert bare.get("/strategies", headers=auth()).json()["strategies"] == []


class TestARunsSettings:
    """BN-322: a run's settings carry everything a backtest request does, so a
    client can start a new one from them."""

    def test_they_carry_the_dates_capital_cost_and_benchmark(self,
                                                             client):
        settings = run_of(client, "plain")["settings"]

        assert settings["start"] == START and settings["end"] == END
        assert settings["initial_capital"] > 0
        assert "transaction_cost_bps" in settings and "benchmark" in settings

    def test_a_run_fills_in_the_dates_it_defaulted(self,
                                                   client):
        """Asked for no dates, a run reports the ones it ran over."""
        finished(client, client.post("/strategies/plain/backtest", json={}, headers=auth()))
        settings = run_of(client, "plain")["settings"]

        assert settings["start"] is not None and settings["end"] is not None

    def test_they_rerun_as_a_request(self,
                                     client):
        """The request fields read back from a run make a valid request."""
        from beacon.server.schemas import StrategyBacktestRequest

        settings = run_of(client, "sampled")["settings"]
        body = {name: settings[name] for name in StrategyBacktestRequest.model_fields
                if name in settings and settings[name] is not None}
        validation = client.post("/strategies/sampled/backtest/validate", json=body,
                                 headers=auth()).json()

        assert validation["valid"] is True, validation["findings"]
