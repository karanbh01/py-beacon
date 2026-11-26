# tests/test_portfolio_dimension.py
"""BN-463: the dimension a figure or table row is the portfolio of is named
"portfolio", in the catalogue, the tables and the server's names, and a
block kept before 0.11.0, which named it "book", is read with the new name."""
import json

from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.server.analysis import NAMES, read_analytics, read_saved
from beacon.server.block_names import portfolio_named
from beacon.server.store import SCHEMA_VERSION_KEY, DocumentStore

TABLES = {"risk_rating", "portfolio_risk_return", "crisis_paths", "style_classification"}


def old_block() -> dict:
    """A block as one kept before 0.11.0: a table with a book column, and a
    figure keyed by its portfolio."""
    return {"statistics": {
        "risk_rating": {"kind": "table",
                        "point": {"index": [0, 1], "columns": ["book", "rating"],
                                  "data": [["fund", 4], ["index", 4]]}},
        "volatility": {"kind": "figures", "point": {"fund": 0.15, "index": 0.14}}}}


class TestTheCatalogue:

    def test_no_statistic_names_a_book(self):
        assert not [statistic.id for statistic in CATALOGUE
                    if "book" in statistic.figure_names
                    or any(column.name == "book" for column in statistic.columns)]

    def test_the_tables_name_their_portfolio(self):
        named = {statistic.id for statistic in CATALOGUE
                 if any(column.name == "portfolio" for column in statistic.columns)}

        assert named >= TABLES

    def test_the_server_labels_the_portfolio(self):
        assert "book" not in NAMES
        assert NAMES["portfolio"]["fund"] == "Fund"


class TestAnOldBlock:

    def test_its_table_s_column_is_renamed(self):
        found = portfolio_named(old_block())["statistics"]

        assert found["risk_rating"]["point"]["columns"] == ["portfolio", "rating"]
        assert found["volatility"]["point"] == {"fund": 0.15, "index": 0.14}

    def test_a_run_s_block_is_read_renamed(self,
                                           tmp_path):
        (tmp_path / "analytics.json").write_text(json.dumps(old_block()), encoding="utf-8")

        found = read_analytics(tmp_path)

        assert found is not None
        assert found["statistics"]["risk_rating"]["point"]["columns"][0] == "portfolio"

    def test_a_saved_analysis_is_read_renamed(self,
                                              tmp_path):
        (tmp_path / "analyses").mkdir()
        (tmp_path / "analyses" / "kept.json").write_text(
            json.dumps({"name": "kept", "analytics": {**old_block(), "request": {}}}),
            encoding="utf-8")

        found = read_saved(tmp_path, "kept")

        assert found["analytics"]["statistics"]["risk_rating"]["point"]["columns"][0] == (
            "portfolio")

    def test_a_stored_job_s_block_is_migrated(self,
                                              tmp_path):
        store = DocumentStore("jobs", root=tmp_path)
        document = {"job_id": "old", "status": "succeeded",
                    "result": {"analytics": old_block()}, SCHEMA_VERSION_KEY: 3}
        (store.directory / "old.json").write_text(json.dumps(document), encoding="utf-8")

        found = store.read("old")

        assert found is not None
        assert found["result"]["analytics"]["statistics"]["risk_rating"]["point"][
            "columns"][0] == "portfolio"

    def test_a_job_without_a_block_is_untouched(self,
                                                tmp_path):
        store = DocumentStore("jobs", root=tmp_path)
        document = {"job_id": "plain", "status": "succeeded", "result": {"nav": []},
                    SCHEMA_VERSION_KEY: 3}
        (store.directory / "plain.json").write_text(json.dumps(document), encoding="utf-8")

        found = store.read("plain")

        assert found is not None
        assert found["result"] == {"nav": []}
