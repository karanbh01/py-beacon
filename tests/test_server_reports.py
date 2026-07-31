# tests/test_server_reports.py
"""BN-75: report templates and the render job."""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "canon"
INDEX_NAME = "Canonical Index"
FACTSHEET = "FACTSHEET-A4"

START = "2023-01-02"
END = "2024-06-28"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def index_document() -> dict:
    return {
        "id": INDEX_ID,
        "name": INDEX_NAME,
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "universe": {"universe_id": None, "identifiers": list(dataset.UNIVERSE)},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting", "scheme": "EqualWeighted",
                          "params": {}},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }


def template_document(template_id: str = "custom") -> dict:
    """A small stored template using several block kinds."""
    return {
        "template_id": template_id,
        "name": "Custom page",
        "page": {"size": "A4", "orientation": "portrait", "margin": 48.0},
        "blocks": [
            {"kind": "header", "title": "A custom report", "subtitle": "Sub",
             "as_of": "2024-06-28"},
            {"kind": "stat_grid",
             "stats": [{"label": "Return", "value": "12.3%", "change": ""}],
             "columns": 4},
            {"kind": "text", "body": "Some prose.", "size": 9.0, "muted": True},
            {"kind": "table", "columns": ["A", "B"], "rows": [["1", "2"]],
             "title": "T", "align_right": [1]},
        ],
    }


@pytest.fixture(scope="module")
def storage():
    with tempfile.TemporaryDirectory() as path:
        yield Path(path)


@pytest.fixture(scope="module")
def client(storage):
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=dataset.data_fetcher(),
                          storage_root=storage)

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        started.put(f"/indices/{INDEX_ID}", json=index_document(), headers=auth())
        started.post(f"/beacon/{INDEX_ID}/backtest",
                     json={"start": START, "end": END}, headers=auth())
        started.portal.call(started.app.state.jobs.drain)

        yield started


def render(client,
           body: dict) -> dict:
    """Submit a render and return the finished job's result."""
    submitted = client.post("/reports/render", json=body, headers=auth())
    assert submitted.status_code == 202, submitted.text

    client.portal.call(client.app.state.jobs.drain)
    job = client.get(f"/jobs/{submitted.json()['job_id']}", headers=auth()).json()
    assert job["status"] == "succeeded", job.get("error")

    return job["result"]


class TestTemplateCrud:

    def test_a_template_round_trips(self,
                                    client):
        """The acceptance criterion: templates round-trip."""
        client.put("/reports/templates/custom", json=template_document(),
                   headers=auth())

        stored = client.get("/reports/templates/custom", headers=auth()).json()

        assert stored["template_id"] == "custom"
        assert len(stored["blocks"]) == 4
        assert stored["blocks"][0]["kind"] == "header"

    def test_the_blocks_survive_unchanged(self,
                                          client):
        client.put("/reports/templates/roundtrip", json=template_document("roundtrip"),
                   headers=auth())

        stored = client.get("/reports/templates/roundtrip", headers=auth()).json()
        table = next(b for b in stored["blocks"] if b["kind"] == "table")

        assert table["rows"] == [["1", "2"]]
        assert table["align_right"] == [1]

    def test_the_path_id_wins_over_the_body(self,
                                            client):
        saved = client.put("/reports/templates/actual",
                           json=template_document("claims-otherwise"),
                           headers=auth()).json()

        assert saved["template_id"] == "actual"

    def test_a_malformed_block_is_refused_at_save(self,
                                                  client):
        """Refused when it is written, not when it is rendered — by which point
        whoever wrote it has moved on."""
        broken = template_document("broken")
        broken["blocks"] = [{"kind": "hologram"}]

        response = client.put("/reports/templates/broken", json=broken,
                              headers=auth())

        assert response.status_code == 500
        assert "unknown block kind" in response.json()["error"]["message"]

    def test_templates_appear_in_the_listing(self,
                                             client):
        client.put("/reports/templates/listed", json=template_document("listed"),
                   headers=auth())

        listing = client.get("/reports/templates", headers=auth()).json()

        assert "listed" in {entry["template_id"] for entry in listing["templates"]}

    def test_the_listing_names_the_built_ins(self,
                                             client):
        """They can be rendered but not edited: they are code, not documents."""
        listing = client.get("/reports/templates", headers=auth()).json()

        assert FACTSHEET in listing["built_in"]

    def test_an_unknown_template_is_a_404(self,
                                          client):
        assert client.get("/reports/templates/nope",
                          headers=auth()).status_code == 404

    def test_deleting_removes_it(self,
                                 client):
        client.put("/reports/templates/temporary",
                   json=template_document("temporary"), headers=auth())

        assert client.delete("/reports/templates/temporary",
                             headers=auth()).status_code == 204
        assert client.get("/reports/templates/temporary",
                          headers=auth()).status_code == 404

    def test_deleting_something_absent_is_a_404(self,
                                                client):
        assert client.delete("/reports/templates/never",
                             headers=auth()).status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.get("/reports/templates").status_code == 401


class TestFactsheet:
    """The acceptance criterion: FACTSHEET-A4 renders a one-page PDF."""

    @pytest.fixture(scope="class")
    def rendered(self,
                 client):
        return render(client, {"template_id": FACTSHEET, "index_id": INDEX_ID})

    def test_it_renders(self,
                        rendered):
        assert rendered["template_id"] == FACTSHEET
        assert rendered["bytes"] > 0

    def test_it_names_the_index(self,
                                rendered):
        assert INDEX_NAME in rendered["name"]

    def test_it_has_the_mock_s_structure(self,
                                         rendered):
        """Header, headline figures, prose, a chart, a table and a chart
        placeholder — six blocks, the shape the mock lays out."""
        assert rendered["blocks"] == 6

    def test_the_document_is_a_single_page(self,
                                           client,
                                           rendered):
        content = client.get(f"/reports/renders/{rendered['render_id']}",
                             headers=auth()).content
        pages = content.count(b"/Type /Page") - content.count(b"/Type /Pages")

        assert pages == 1

    def test_the_download_is_a_pdf(self,
                                   client,
                                   rendered):
        response = client.get(f"/reports/renders/{rendered['render_id']}",
                              headers=auth())

        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")
        assert response.headers["content-type"] == "application/pdf"

    def test_two_renders_of_one_run_are_identical(self,
                                                  client):
        """Determinism carries through the endpoint, not only the renderer."""
        first = render(client, {"template_id": FACTSHEET, "index_id": INDEX_ID})
        second = render(client, {"template_id": FACTSHEET, "index_id": INDEX_ID})

        first_bytes = client.get(f"/reports/renders/{first['render_id']}",
                                 headers=auth()).content
        second_bytes = client.get(f"/reports/renders/{second['render_id']}",
                                  headers=auth()).content

        assert first_bytes == second_bytes

    def test_a_built_in_needs_an_index(self,
                                       client):
        """It is generated from a run, so there is nothing to generate without
        one."""
        response = client.post("/reports/render",
                               json={"template_id": FACTSHEET}, headers=auth())

        assert response.status_code == 404

    def test_an_index_without_a_backtest_is_a_404(self,
                                                  client):
        response = client.post("/reports/render",
                               json={"template_id": FACTSHEET,
                                     "index_id": "never-run"},
                               headers=auth())

        assert response.status_code == 404

    def test_a_run_without_composition_is_refused(self,
                                                  client):
        """A factsheet without holdings is not a factsheet."""
        client.app.state.jobs._results.write(
            "legacy",
            {"job_id": "legacy", "kind": "backtest:legacy-index",
             "status": "succeeded", "completed_at": "2020-01-01T00:00:00+00:00",
             "result": {"level": {"index": [], "data": []},
                        "metrics": {"total_return": 0.1}}})

        response = client.post("/reports/render",
                               json={"template_id": FACTSHEET,
                                     "index_id": "legacy-index"},
                               headers=auth())

        assert response.status_code == 404
        assert "rebalance snapshots" in response.json()["error"]["message"]


class TestFactsheetContent:
    """What the generated blocks actually say."""

    @pytest.fixture(scope="class")
    def blocks(self,
               client):
        from beacon.server.reports import build_factsheet

        run = client.app.state.jobs.latest_result(f"backtest:{INDEX_ID}")

        return build_factsheet(INDEX_NAME, run).blocks

    def test_the_header_carries_the_as_of_date(self,
                                               blocks):
        assert blocks[0].as_of.startswith("20")

    def test_the_headline_figures_are_percentages(self,
                                                  blocks):
        values = [stat.value for stat in blocks[1].stats[:3]]

        assert all(value.endswith("%") for value in values)

    def test_a_missing_metric_shows_a_dash_not_a_zero(self):
        """A metric that could not be computed and one that came out at zero
        are different statements, and a factsheet is read by people who will
        not check which."""
        from beacon.server.reports import _percent

        assert _percent(None) == "—"
        assert _percent(0.0) == "0.00%"

    def test_the_holdings_table_matches_the_chart(self,
                                                  blocks):
        chart = blocks[3]
        table = blocks[4]

        assert chart.categories == [row[0] for row in table.rows]

    def test_the_weights_are_ordered_largest_first(self,
                                                   blocks):
        assert blocks[3].values == sorted(blocks[3].values, reverse=True)

    def test_the_summary_mentions_the_rebalances(self,
                                                 blocks):
        assert "Rebalanced across" in blocks[2].body


class TestRenderingAStoredTemplate:

    def test_it_renders_exactly_what_was_saved(self,
                                               client):
        """Nothing is substituted in: that is what "I designed this page"
        means."""
        client.put("/reports/templates/literal", json=template_document("literal"),
                   headers=auth())

        result = render(client, {"template_id": "literal"})

        assert result["blocks"] == 4
        assert result["name"] == "Custom page"

    def test_an_index_id_is_ignored_for_a_stored_template(self,
                                                          client):
        client.put("/reports/templates/ignores", json=template_document("ignores"),
                   headers=auth())

        result = render(client, {"template_id": "ignores", "index_id": INDEX_ID})

        assert result["name"] == "Custom page"

    def test_an_empty_template_is_refused(self,
                                          client):
        """It would render a blank page, which is not what anyone asked for."""
        empty = template_document("empty")
        empty["blocks"] = []
        client.put("/reports/templates/empty", json=empty, headers=auth())

        response = client.post("/reports/render",
                               json={"template_id": "empty"}, headers=auth())

        assert response.status_code == 500
        assert "no blocks" in response.json()["error"]["message"]

    def test_content_that_does_not_fit_fails_the_job(self,
                                                     client):
        """The renderer refuses rather than paginating, and names the block."""
        crowded = template_document("crowded")
        crowded["blocks"] = [{"kind": "chart", "title": f"Chart {n}",
                              "height": 200.0, "image_path": None}
                             for n in range(8)]
        client.put("/reports/templates/crowded", json=crowded, headers=auth())

        submitted = client.post("/reports/render",
                                json={"template_id": "crowded"}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        job = client.get(f"/jobs/{submitted.json()['job_id']}",
                         headers=auth()).json()

        assert job["status"] == "failed"
        assert "of the page is left" in job["error"]


class TestDownload:

    def test_an_unrendered_id_is_a_404(self,
                                       client):
        response = client.get("/reports/renders/never-rendered", headers=auth())

        assert response.status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.get("/reports/renders/anything").status_code == 401


class TestOpenApi:

    def test_every_endpoint_is_documented(self,
                                          client):
        paths = client.app.openapi()["paths"]

        for path in ("/reports/templates",
                     "/reports/templates/{template_id}",
                     "/reports/render",
                     "/reports/renders/{render_id}"):
            assert path in paths, path
