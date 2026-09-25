# src/beacon/data/postgres.py
"""Read data from a Postgres database, in the layout `beacon.data.layout` sets out.

A Postgres store is tables or views named after the layout's sheets (`market`,
`reference`, and optionally `fx`, `corporate_actions`, `features`), with the
same columns, in one schema. To use data you already hold, create views with
those names and columns over your own tables.

The engine only reads: it connects with a read-only session, and never
creates, changes or deletes anything in the database.

The password is never stored. Name an environment variable that holds it
(`password_env`), and it is read from there each time the engine connects.

    from beacon.data.postgres import PostgresSource, load

    source = PostgresSource(host="localhost", database="markets",
                            user="analyst", password_env="MARKETS_PASSWORD")
    data = load(source)

Needs the `postgres` extra: `pip install "py-beacon-kit[postgres]"`.
"""
import os
import re
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._optional import require
from . import layout
from .fetcher import DataFetcher

# A schema name is put into the query text, so it is limited to what an
# unquoted Postgres identifier can hold. Table names come from the layout.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Seconds to wait for the database before reporting it unreachable.
CONNECT_TIMEOUT = 10

# The pseudo-sheet a connection problem is reported against.
CONNECTION = "connection"


@dataclass(frozen=True)
class PostgresSource:
    """Where a Postgres store's tables are.

    Attributes:
        host: The database server.
        database: The database name.
        user: The user to connect as. Read access is all it needs.
        port: The server's port.
        schema: The schema holding the tables or views.
        password_env: The environment variable holding the password, or None
            when the server needs none (trusted or peer authentication).
    """
    host: str
    database: str
    user: str
    port: int = 5432
    schema: str = "public"
    password_env: str | None = None

    def __post_init__(self) -> None:
        if not IDENTIFIER.fullmatch(self.schema):
            raise ValueError(f"'{self.schema}' is not a schema name: use "
                             f"letters, digits and underscores.")

    def describe(self) -> str:
        """Where the data is, without the password."""
        return (f"postgresql://{self.user}@{self.host}:{self.port}/"
                f"{self.database} (schema {self.schema})")

    def password(self) -> str | None:
        """The password from its environment variable, or None."""
        return os.environ.get(self.password_env) if self.password_env else None

    def password_missing(self) -> bool:
        """Whether a password variable is named but not set."""
        return self.password_env is not None and self.password() is None


# Opens a database connection following the Python database API (DB-API 2):
# `cursor()`, `rollback()`, `close()`. Postgres by default; tests pass another.
Connector = Callable[[PostgresSource], Any]


def connect(source: PostgresSource) -> Any:
    """A read-only connection to the database."""
    psycopg = require("psycopg", "Postgres data stores")

    connection = psycopg.connect(host=source.host,
                                 port=source.port,
                                 dbname=source.database,
                                 user=source.user,
                                 password=source.password(),
                                 connect_timeout=CONNECT_TIMEOUT)
    # Every transaction this connection starts is read-only, so nothing the
    # engine sends could change the database.
    connection.read_only = True

    return connection


def read(source: PostgresSource,
         connector: Connector | None = None) -> tuple[dict[str, pd.DataFrame],
                                                      list[layout.Problem]]:
    """Read each layout table the schema holds, and the problems reading them.

    A table the schema does not have is simply absent, and `layout.check`
    says so if it was required. Any other failure (no permission, say) is a
    problem naming the table. *connector* defaults to `connect`, looked up
    when called.
    """
    if source.password_missing():
        return {}, [layout.Problem(CONNECTION, None, None, "NO_PASSWORD",
                                   f"The environment variable "
                                   f"{source.password_env} is not set.")]

    try:
        connection = (connector or connect)(source)
    except Exception as error:  # the driver's errors share no base we can name
        return {}, [layout.Problem(CONNECTION, None, None, "CONNECTION_FAILED",
                                   f"Cannot connect to {source.describe()}: "
                                   f"{error}".strip())]

    sheets: dict[str, pd.DataFrame] = {}
    problems: list[layout.Problem] = []

    with closing(connection):
        for sheet in layout.SHEETS:
            try:
                sheets[sheet.name] = _read_table(connection, source.schema,
                                                 sheet.name)
            except Exception as error:
                # A failed statement ends the transaction in Postgres; the
                # next table needs a fresh one.
                connection.rollback()

                if not _is_missing_table(error):
                    problems.append(layout.Problem(
                        sheet.name, None, None, "UNREADABLE_TABLE",
                        f"{source.schema}.{sheet.name} cannot be read: "
                        f"{error}".strip()))

    return sheets, problems


def load(source: PostgresSource,
         connector: Connector | None = None,
         **settings: Any) -> DataFetcher:
    """Read, check and load a Postgres store.

    Raises:
        DataImportError: If the database cannot be reached, a table cannot be
            read, or any row has a problem. Nothing is loaded in that case.
    """
    sheets, problems = read(source, connector)

    # A database that could not be reached has no tables to check, and
    # saying the required ones are missing would only add noise.
    if any(problem.sheet == CONNECTION for problem in problems):
        raise layout.DataImportError(problems, len(problems))

    found, total = layout.check(sheets)
    problems = [*problems, *found]
    total += len(problems) - len(found)

    if problems:
        raise layout.DataImportError(problems[:layout.MAX_PROBLEMS], total)

    return layout.to_fetcher(sheets, **settings)


def _read_table(connection: Any,
                schema: str,
                table: str) -> pd.DataFrame:
    """Every row of one table or view, with its columns tidied."""
    cursor = connection.cursor()

    try:
        cursor.execute(f'SELECT * FROM "{schema}"."{table}"')
        columns = [description[0] for description in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        cursor.close()

    return layout.tidy(frame.map(_as_text))


def _as_text(value: Any) -> Any:
    """A value as the layout's checks expect it: text, or blank.

    Dates become YYYY-MM-DD, so a date column is read the same way whether it
    came from a file or a database.
    """
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def _is_missing_table(error: Exception) -> bool:
    """Whether the error says the table does not exist.

    Postgres raises UndefinedTable; other databases that speak the same
    interface say "no such table".
    """
    return (type(error).__name__ == "UndefinedTable"
            or "no such table" in str(error).lower()
            or "does not exist" in str(error).lower())
