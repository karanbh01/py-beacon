# tests/test_server_analysis.py
"""BN-398/BN-400: a run's analysis through the API.

The catalogue of what an analysis can be asked for; a strategy backtested
with an analysis, whose block the job answers with and the strategy serves
later; its trades whole, by column and as Arrow; requests refused; and large
responses compressed only for a client on another machine.
"""
import gzip
import io

import pyarrow.ipc as ipc
import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from beacon.analysis import RunData
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.definitions import REPORTS
from beacon.server import ServerConfig, create_app
from beacon.server.analysis import ARROW
from beacon.server.compression import RemoteGZip
from beacon.server.jobs import SUCCEEDED
from test_server_strategy_runs import (
    END,
    START,
    TOKEN,
    auth,
    finished,
    index_document,
    strategy_document,
)

ANALYSIS = {"reports": {"factsheet": "core", "risk": "extended"},
            "add": {"factsheet": ["sortino_ratio"]},
            "time_views": {"factsheet": {"sharpe_ratio": "both"}},
            "options": {"rolling": {"window": 63}, "value_at_risk": {"levels": [0.95]}}}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A strategy backtested with an analysis, one without, and one never
    backtested."""
    from beacon.testing import dataset

    config = ServerConfig(auth_token=TOKEN, data_fetcher=dataset.data_fetcher(),
                          storage_root=tmp_path_factory.mktemp("analysis"))

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        assert started.put("/indices/canon", json=index_document(),
                           headers=auth()).status_code == 200

        for strategy_id in ("analysed", "plain", "unrun"):
            assert started.post("/strategies", json=strategy_document(strategy_id),
                                headers=auth()).status_code == 200

        for strategy_id, analysis in (("analysed", ANALYSIS), ("plain", None)):
            body = {"start": START, "end": END, **({"analysis": analysis} if analysis else {})}
            job = finished(started, started.post(f"/strategies/{strategy_id}/backtest",
                                                 json=body, headers=auth()))
            assert job["status"] == SUCCEEDED, job
            started.app.state.__dict__.setdefault("jobs_by_strategy", {})[strategy_id] = job

        yield started


def job_of(client,
           strategy_id: str) -> dict:
    return client.app.state.jobs_by_strategy[strategy_id]


class TestTheCatalogue:

    @pytest.fixture(scope="class")
    def listed(self,
               client):
        response = client.get("/analysis/options", headers=auth())
        assert response.status_code == 200, response.text

        return response.json()

    def test_it_lists_every_statistic_and_report(self,
                                                 listed):
        assert [statistic["id"] for statistic in listed["statistics"]] == list(
            CATALOGUE.statistics)
        assert [report["id"] for report in listed["reports"]] == list(REPORTS)

    def test_each_level_lists_its_sections_in_full(self,
                                                   listed):
        factsheet = listed["reports"][0]
        core = [entry["id"] for section in factsheet["levels"]["core"]["sections"]
                for entry in section["statistics"] if entry["state"] == "included"]

        assert core == REPORTS["factsheet"].level("core").included
        assert {"core", "extended"} == set(factsheet["levels"])

    def test_settings_are_drawn_as_the_backtest_options_are(self,
                                                            listed):
        rolling = next(section for section in listed["settings"]
                       if section["name"] == "rolling")
        window = next(parameter for parameter in rolling["parameters"]
                      if parameter["name"] == "window")

        assert window["type"] == "integer" and window["suggestions"]
        assert {"label", "order", "help", "unit"} <= set(window)

    def test_figure_names_and_codes_are_labelled(self,
                                                 listed):
        assert listed["names"]["method"]["cornish_fisher"] == "Cornish-Fisher"
        assert {reason["code"] for reason in listed["unavailable"]} >= {"no_index",
                                                                         "no_futures"}
        trades = next(table for table in listed["tables"] if table["name"] == "trades")
        assert [column["name"] for column in trades["columns"]][:3] == ["date", "name", "side"]


class TestTheBlock:

    def test_the_job_answers_with_it(self,
                                     client):
        block = job_of(client, "analysed")["result"]["analytics"]

        assert set(block["reports"]) == {"factsheet", "risk"}
        assert "sortino_ratio" in block["statistics"]
        assert block["statistics"]["sharpe_ratio"]["rolling"]["window"] == 63
        assert block["tables"]["trades"]["path"] == "/strategies/analysed/tables/trades"

    def test_the_strategy_serves_it_later(self,
                                          client):
        response = client.get("/strategies/analysed/analytics", headers=auth())

        assert response.status_code == 200
        assert response.json()["status"] == "computed"
        assert response.json()["analytics"] == job_of(client, "analysed")["result"]["analytics"]

    def test_a_run_without_one_says_so(self,
                                       client):
        response = client.get("/strategies/plain/analytics", headers=auth())

        assert response.json()["status"] == "not_computed"
        assert response.json()["analytics"] is None and response.json()["reason"]
        assert job_of(client, "plain")["result"]["analytics"] is None

    def test_a_strategy_never_backtested_is_not_found(self,
                                                      client):
        assert client.get("/strategies/unrun/analytics", headers=auth()).status_code == 404
        assert client.get("/strategies/nobody/analytics", headers=auth()).status_code == 404


class TestTheTrades:

    @pytest.fixture(scope="class")
    def kept(self,
             client):
        folder = client.app.state.backtest_record_store.panels_directory("analysed")

        return RunData.from_record(folder).tables["trades"]

    def test_by_column_every_trade_comes(self,
                                         client,
                                         kept):
        table = client.get("/strategies/analysed/tables/trades", headers=auth()).json()
        names = table["columns"]["name"]

        assert table["length"] == len(kept)
        assert [names["values"][code] for code in names["codes"]] == kept["IDENTIFIER"].tolist()
        assert table["columns"]["units"] == kept["QUANTITY"].round(6).tolist()
        assert table["decimals"]["cost"] == 4

    def test_as_arrow_every_trade_comes(self,
                                        client,
                                        kept):
        response = client.get("/strategies/analysed/tables/trades",
                              headers={**auth(), "Accept": ARROW})
        table = ipc.open_stream(io.BytesIO(response.content)).read_all()

        assert response.headers["content-type"].startswith(ARROW)
        assert table.num_rows == len(kept)
        assert table.column("name").to_pylist() == kept["IDENTIFIER"].tolist()
        assert table.column("price").to_pylist() == kept["PRICE"].round(4).tolist()

    def test_an_unknown_table_is_not_found(self,
                                           client):
        assert client.get("/strategies/analysed/tables/secrets",
                          headers=auth()).status_code == 404


class TestRefusals:

    @pytest.mark.parametrize("analysis", [
        {"reports": {"nope": "core"}},
        {"reports": {"factsheet": "deep"}},
        {"options": {"nowhere": {}}},
        {"options": {"rolling": {"length": 5}}},
        {"time_views": {"factsheet": {"trailing_returns": "rolling"}}},
    ])
    def test_an_analysis_that_cannot_be_given_is_refused_before_the_run(self,
                                                                         client,
                                                                         analysis):
        response = client.post("/strategies/analysed/backtest",
                               json={"start": START, "end": END, "analysis": analysis},
                               headers=auth())

        assert response.status_code == 422, response.text


class TestCompression:

    @staticmethod
    def app():
        large = PlainTextResponse("x" * 2_000_000)
        return RemoteGZip(Starlette(routes=[Route("/", lambda request: large)]))

    def test_a_large_response_to_another_machine_is_gzipped(self):
        with TestClient(self.app(), client=("203.0.113.5", 50000)) as remote:
            response = remote.get("/", headers={"Accept-Encoding": "gzip"})

        assert response.headers.get("content-encoding") == "gzip"
        assert response.text == "x" * 2_000_000

    def test_one_to_the_same_machine_is_not(self):
        with TestClient(self.app(), client=("127.0.0.1", 50000)) as local:
            response = local.get("/", headers={"Accept-Encoding": "gzip"})

        assert "content-encoding" not in response.headers

    def test_a_small_one_is_never_gzipped(self):
        small = RemoteGZip(Starlette(routes=[Route("/", lambda request: PlainTextResponse(
            "small"))]))

        with TestClient(small, client=("203.0.113.5", 50000)) as remote:
            assert "content-encoding" not in remote.get(
                "/", headers={"Accept-Encoding": "gzip"}).headers


def test_gzip_is_what_a_client_reads_back():
    assert gzip.decompress(gzip.compress(b"ok")) == b"ok"
