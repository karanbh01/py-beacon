# tests/test_server_import.py
"""BN-239: importing CSV files or an Excel workbook through the engine."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from beacon.data import importing
from beacon.server import ServerConfig, create_app

TOKEN = "import-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
MARKET = "IDENTIFIER,DATE,CLOSE\nAAA,2024-01-02,100\nAAA,2024-01-03,101\n"
REFERENCE = ("IDENTIFIER,NAME,CURRENCY,EXCHANGE,DATE_FROM\n"
             "AAA,Alpha,USD,XNYS,2020-01-01\n")


@pytest.fixture
def client(tmp_path):
    app = create_app(ServerConfig(auth_token=TOKEN,
                                  storage_root=tmp_path / "documents"))

    with TestClient(app, raise_server_exceptions=False) as started:
        yield started


def csv_files(folder,
              market: str = MARKET,
              reference: str = REFERENCE) -> list[str]:
    (folder / "market.csv").write_text(market)
    (folder / "reference.csv").write_text(reference)

    return [str(folder / "market.csv"), str(folder / "reference.csv")]


def served(client) -> dict:
    return client.get("/health", headers=HEADERS).json()["data_source"]


class TestImporting:

    def test_valid_files_become_a_store_that_is_loaded(self,
                                                       client,
                                                       tmp_path):
        before = served(client)["data_version"]

        response = client.post("/data/import", headers=HEADERS,
                               json={"name": "My Data",
                                     "paths": csv_files(tmp_path)})
        client.portal.call(client.app.state.jobs.drain)
        body = response.json()

        assert response.status_code == 201
        assert body["store"]["id"] == "my-data"
        assert body["store"]["managed"] is True
        assert body["store"]["source"] == "imported"
        assert body["load_job"]["kind"] == "load:my-data"
        assert served(client)["store_id"] == "my-data"
        assert served(client)["data_version"] != before

    def test_it_can_be_saved_without_loading(self,
                                             client,
                                             tmp_path):
        response = client.post("/data/import", headers=HEADERS,
                               json={"paths": csv_files(tmp_path),
                                     "activate": False})

        assert response.json()["load_job"] is None
        assert response.json()["store"]["name"] == "Imported data"
        assert served(client)["configured"] is False

    def test_an_excel_workbook_works_the_same_way(self,
                                                  client,
                                                  tmp_path):
        workbook = tmp_path / "data.xlsx"
        workbook.write_bytes(importing.template("xlsx"))

        response = client.post("/data/import", headers=HEADERS,
                               json={"paths": [str(workbook)]})

        assert response.status_code == 201


class TestRefusing:

    def test_bad_rows_are_refused_with_one_finding_each(self,
                                                        client,
                                                        tmp_path):
        market = MARKET + "AAA,not a date,102\n"
        response = client.post("/data/import", headers=HEADERS,
                               json={"paths": csv_files(tmp_path, market=market)})
        error = response.json()["error"]

        assert response.status_code == 422
        assert error["code"] == "INVALID_RULE"
        assert error["detail"]["total"] == 1
        assert error["detail"]["findings"][0] | {"message": ""} == {
            "path": "market, row 4, DATE", "rule_id": None,
            "severity": "error", "code": "BAD_DATE", "message": "",
            "sheet": "market", "row": 4, "column": "DATE"}

    def test_nothing_is_saved_when_refused(self,
                                           client,
                                           tmp_path):
        client.post("/data/import", headers=HEADERS,
                    json={"paths": csv_files(tmp_path, reference="X\n1\n")})

        assert client.get("/data/stores", headers=HEADERS).json()["stores"] == []
        assert list(client.app.state.managed_store_root.iterdir()) == []

    def test_a_path_that_does_not_exist_is_a_finding(self,
                                                     client,
                                                     tmp_path):
        response = client.post("/data/import", headers=HEADERS,
                               json={"paths": [str(tmp_path / "missing.csv")]})
        codes = [finding["code"] for finding in response.json()["error"]["detail"]["findings"]]

        assert response.status_code == 422
        assert "UNREADABLE_FILE" in codes


class TestTheTemplateDownload:

    def test_the_excel_template(self,
                                client):
        response = client.get("/data/import/template", headers=HEADERS)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "application/vnd.openxmlformats")
        assert response.content == importing.template("xlsx")

    def test_the_csv_template_is_a_zip_of_every_sheet(self,
                                                      client):
        response = client.get("/data/import/template", headers=HEADERS,
                              params={"format": "csv"})
        names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()

        assert response.headers["content-type"] == "application/zip"
        assert sorted(names) == sorted(["market.csv", "reference.csv", "fx.csv",
                                        "corporate_actions.csv", "features.csv"])

    def test_another_format_is_refused(self,
                                       client):
        response = client.get("/data/import/template", headers=HEADERS,
                              params={"format": "pdf"})

        assert response.status_code == 422
