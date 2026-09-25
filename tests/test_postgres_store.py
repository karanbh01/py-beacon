# tests/test_postgres_store.py
"""BN-241: a data store can be a Postgres database, read-only.

Most tests stand SQLite in for Postgres: both speak the same Python database
interface, and the reading and checking do not depend on which database is
behind it. The tests at the end run against a real Postgres, and only when
`BEACON_TEST_POSTGRES_URL` names a disposable database to use. They create a
schema of their own and drop it afterwards; point them at nothing that holds
data you care about.
"""
import os
import sqlite3
import uuid
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

from beacon.data import layout, postgres
from beacon.server import ServerConfig, create_app

TOKEN = "postgres-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
# SQLite's own schema name, standing in for Postgres's "public".
SCHEMA = "main"


def database(path,
             *statements: str) -> str:
    """A SQLite file holding a valid store, plus any extra statements."""
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE market (identifier TEXT, date TEXT, close REAL);
        INSERT INTO market VALUES ('AAA', '2024-01-02', 100), ('AAA', '2024-01-03', 101);
        CREATE TABLE reference (identifier TEXT, name TEXT, currency TEXT,
                                exchange TEXT, date_from TEXT);
        INSERT INTO reference VALUES ('AAA', 'Alpha', 'USD', 'XNYS', '2020-01-01');
    """)

    for statement in statements:
        connection.execute(statement)

    connection.commit()
    connection.close()

    return str(path)


def source(**overrides) -> postgres.PostgresSource:
    return postgres.PostgresSource(**{"host": "localhost", "database": "main",
                                      "user": "reader", "schema": SCHEMA,
                                      **overrides})


def sqlite_connector(path: str):
    return lambda _source: sqlite3.connect(path)


class TestReading:

    def test_the_layout_tables_load(self,
                                    tmp_path):
        path = database(tmp_path / "db.sqlite")

        data = postgres.load(source(), connector=sqlite_connector(path))

        assert data.identifiers == ["AAA"]
        assert data.fetch_price("AAA", "2024-01-03") == 101

    def test_a_view_over_your_own_tables_works(self,
                                               tmp_path):
        """The intended use: views with the layout's names and columns over
        tables that have their own."""
        path = database(tmp_path / "db.sqlite",
                        "CREATE TABLE my_rates (pair TEXT, day TEXT, px REAL)",
                        "INSERT INTO my_rates VALUES ('GBPUSD', '2024-01-02', 1.27)",
                        "CREATE VIEW fx AS SELECT pair, day AS date, px AS rate "
                        "FROM my_rates")

        data = postgres.load(source(), connector=sqlite_connector(path))

        assert data.fx_rate_on("GBP", "USD", "2024-01-02") == pytest.approx(1.27)

    def test_a_missing_required_table_is_named(self,
                                               tmp_path):
        path = tmp_path / "db.sqlite"
        sqlite3.connect(path).executescript(
            "CREATE TABLE market (identifier TEXT, date TEXT, close REAL);")

        with pytest.raises(layout.DataImportError) as raised:
            postgres.load(source(), connector=sqlite_connector(str(path)))

        assert [(problem.sheet, problem.code) for problem in raised.value.problems] == [
            ("market", "EMPTY_SHEET"), ("reference", "MISSING_SHEET")]

    def test_a_table_that_exists_but_cannot_be_read_is_named(self,
                                                             tmp_path):
        """Not the same as a missing optional table, which is simply absent."""
        path = database(tmp_path / "db.sqlite",
                        "CREATE VIEW features AS SELECT no_such_column FROM market")

        with pytest.raises(layout.DataImportError) as raised:
            postgres.load(source(), connector=sqlite_connector(path))

        assert [(problem.sheet, problem.code) for problem in raised.value.problems] == [
            ("features", "UNREADABLE_TABLE")]

    def test_bad_rows_are_found_at_their_rows(self,
                                              tmp_path):
        path = database(tmp_path / "db.sqlite",
                        "INSERT INTO market VALUES ('AAA', 'soon', 102)")

        with pytest.raises(layout.DataImportError) as raised:
            postgres.load(source(), connector=sqlite_connector(path))

        assert [(problem.sheet, problem.row, problem.code)
                for problem in raised.value.problems] == [("market", 4, "BAD_DATE")]

    def test_a_database_that_cannot_be_reached_says_so(self):
        def refuse(_source):
            raise OSError("connection refused")

        with pytest.raises(layout.DataImportError) as raised:
            postgres.load(source(), connector=refuse)

        # Only the connection: the tables it could not read are not also
        # reported missing.
        assert [problem.code for problem in raised.value.problems] == [
            "CONNECTION_FAILED"]

    def test_a_password_variable_that_is_not_set_is_named(self,
                                                          monkeypatch):
        monkeypatch.delenv("BEACON_TEST_NO_SUCH_PASSWORD", raising=False)

        with pytest.raises(layout.DataImportError) as raised:
            postgres.load(source(password_env="BEACON_TEST_NO_SUCH_PASSWORD"))

        assert raised.value.problems[0].code == "NO_PASSWORD"

    def test_a_schema_name_cannot_carry_sql(self):
        with pytest.raises(ValueError, match="schema name"):
            source(schema='public"; DROP TABLE market; --')


class TestThroughTheEngine:

    @pytest.fixture
    def client(self,
               tmp_path,
               monkeypatch):
        path = database(tmp_path / "db.sqlite")
        monkeypatch.setattr(postgres, "connect",
                            lambda _source: sqlite3.connect(path))
        app = create_app(ServerConfig(auth_token=TOKEN,
                                      storage_root=tmp_path / "documents"))

        with TestClient(app, raise_server_exceptions=False) as started:
            yield started

    @staticmethod
    def register(client,
                 **connection):
        return client.post("/data/stores", headers=HEADERS, json={
            "name": "Warehouse", "kind": "postgres",
            "connection": {"host": "db.internal", "database": "markets",
                           "user": "reader", "schema": SCHEMA, **connection}})

    def test_a_database_is_checked_then_registered(self,
                                                   client):
        response = self.register(client)
        body = response.json()

        assert response.status_code == 201
        assert body["kind"] == "postgres"
        assert body["source"] == "database"
        assert body["managed"] is False
        assert body["path"] == ("postgresql://reader@db.internal:5432/markets "
                                "(schema main)")
        assert body["connection"]["schema"] == SCHEMA
        assert "password" not in body["connection"]

    def test_it_loads_like_any_store(self,
                                     client):
        self.register(client)

        client.post("/data/stores/warehouse/activate", headers=HEADERS)
        client.portal.call(client.app.state.jobs.drain)
        served = client.get("/health", headers=HEADERS).json()["data_source"]

        assert served["store_id"] == "warehouse"
        assert served["identifiers"] == 1

    def test_the_same_database_twice_is_a_conflict(self,
                                                   client):
        self.register(client)

        assert self.register(client).status_code == 409

    def test_a_database_with_problems_is_refused_and_not_registered(self,
                                                                    client,
                                                                    monkeypatch):
        def refuse(_source):
            raise OSError("connection refused")

        monkeypatch.setattr(postgres, "connect", refuse)
        response = self.register(client)

        assert response.status_code == 422
        assert response.json()["error"]["detail"]["findings"][0]["code"] == (
            "CONNECTION_FAILED")
        assert client.get("/data/stores", headers=HEADERS).json()["stores"] == []

    def test_a_missing_password_variable_blocks_loading(self,
                                                        client,
                                                        monkeypatch):
        monkeypatch.setenv("BEACON_TEST_PASSWORD", "secret")
        self.register(client, password_env="BEACON_TEST_PASSWORD")
        monkeypatch.delenv("BEACON_TEST_PASSWORD")

        listing = client.get("/data/stores", headers=HEADERS).json()["stores"]
        response = client.post("/data/stores/warehouse/activate", headers=HEADERS)

        assert listing[0]["readable"] is False
        assert response.status_code == 409
        assert "password" in response.json()["error"]["message"]

    def test_both_a_path_and_nothing_are_refused_by_kind(self,
                                                         client):
        folder_without_path = client.post("/data/stores", headers=HEADERS,
                                          json={"name": "x", "kind": "folder"})
        database_without_connection = client.post(
            "/data/stores", headers=HEADERS, json={"name": "x", "kind": "postgres"})

        assert folder_without_path.status_code == 422
        assert database_without_connection.status_code == 422


# -- a real Postgres -------------------------------------------------------

REAL = os.environ.get("BEACON_TEST_POSTGRES_URL")


@pytest.fixture
def real_schema():
    """A schema of its own in the test database, dropped afterwards."""
    psycopg = pytest.importorskip("psycopg")
    schema = f"beacon_test_{uuid.uuid4().hex[:12]}"

    with psycopg.connect(REAL, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(f'''
            CREATE TABLE "{schema}".prices (ticker text, day date, px numeric);
            INSERT INTO "{schema}".prices VALUES ('AAA', '2024-01-02', 100.5);
            CREATE VIEW "{schema}".market AS
                SELECT ticker AS identifier, day AS date, px AS close
                FROM "{schema}".prices;
            CREATE TABLE "{schema}".reference (identifier text, name text,
                currency text, exchange text, date_from date);
            INSERT INTO "{schema}".reference
                VALUES ('AAA', 'Alpha', 'USD', 'XNYS', '2020-01-01');
        ''')

    yield schema

    with psycopg.connect(REAL, autocommit=True) as connection:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


def real_source(schema: str,
                monkeypatch) -> postgres.PostgresSource:
    parts = urlparse(REAL)
    monkeypatch.setenv("BEACON_TEST_POSTGRES_PASSWORD", parts.password or "")

    return postgres.PostgresSource(host=parts.hostname or "localhost",
                                   port=parts.port or 5432,
                                   database=parts.path.lstrip("/"),
                                   user=parts.username or "postgres",
                                   schema=schema,
                                   password_env="BEACON_TEST_POSTGRES_PASSWORD")


@pytest.mark.skipif(not REAL, reason="set BEACON_TEST_POSTGRES_URL to a "
                                     "disposable database to run these")
class TestARealPostgres:

    def test_it_loads_through_a_view(self,
                                     real_schema,
                                     monkeypatch):
        data = postgres.load(real_source(real_schema, monkeypatch))

        assert data.identifiers == ["AAA"]
        assert data.fetch_price("AAA", "2024-01-02") == pytest.approx(100.5)

    def test_the_engines_connection_cannot_write(self,
                                                 real_schema,
                                                 monkeypatch):
        """Read-only is enforced by the database, not only by what the
        engine chooses to send."""
        psycopg = pytest.importorskip("psycopg")
        connection = postgres.connect(real_source(real_schema, monkeypatch))

        with connection, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            connection.execute(f'CREATE TABLE "{real_schema}".intruder (x int)')
