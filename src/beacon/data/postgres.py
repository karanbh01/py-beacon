# src/beacon/data/postgres.py
"""Read data from a Postgres database, in the layout `beacon.data.layout` sets out.

A Postgres store is tables or views named after the layout's sheets (`market`,
`reference`, and optionally `corporate_actions`, `features`), with the
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
"""
import os
import re
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import pandas as pd

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
    import psycopg  # noqa: PLC0415

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

    # A retired table is read too, so that one still there is refused by
    # `layout.check` rather than ignored (BN-308).
    names = [*(sheet.name for sheet in layout.SHEETS), *layout.RETIRED_SHEETS]

    with closing(connection):
        for name in names:
            try:
                sheets[name], wrong = _read_table(connection, source.schema, name)
                problems.extend(wrong)
            except Exception as error:
                # A failed statement ends the transaction in Postgres; the
                # next table needs a fresh one.
                connection.rollback()

                if not _is_missing_table(error):
                    problems.append(layout.Problem(
                        name, None, None, "UNREADABLE_TABLE",
                        f"{source.schema}.{name} cannot be read: "
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
                table: str) -> tuple[pd.DataFrame, list[layout.Problem]]:
    """Every row of one table or view, with its columns tidied, and any
    column whose type the layout cannot use.

    From Postgres the table is streamed out with COPY and its columns read
    by type, a whole column at a time; another database that speaks the
    same interface is read row by row.
    """
    cursor = connection.cursor()

    try:
        if hasattr(cursor, "copy"):
            return _copied(cursor, schema, table)

        cursor.execute(f'SELECT * FROM "{schema}"."{table}"')
        columns = [description[0] for description in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        cursor.close()

    return layout.tidy(frame), []


def _copied(cursor: Any,
            schema: str,
            table: str) -> tuple[pd.DataFrame, list[layout.Problem]]:
    """A Postgres table streamed out with COPY, each column read as its own
    type, and the date and number columns that have another type.

    Each value used to come back as a Python object, be written as text and
    be read again from it: 58 s and 13.8 GiB for a 6,000-name store's 11.4
    million market rows. A NULL in a number column came back as NaN, which
    became the text 'nan' and was refused as not a number.
    """
    # BN-350.
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.csv as pa_csv  # noqa: PLC0415

    cursor.execute("SELECT column_name, data_type FROM information_schema.columns "
                   "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                   (schema, table))
    types = {str(name): str(kind) for name, kind in cursor.fetchall()}

    if not types:
        raise LookupError(f'relation "{schema}.{table}" does not exist')

    problems = list(_wrong_types(schema, table, types))
    query = f'COPY (SELECT * FROM "{schema}"."{table}") TO STDOUT (FORMAT CSV, HEADER)'

    with cursor.copy(query) as copy:
        read = pa_csv.read_csv(
            _Stream(copy),
            parse_options=pa_csv.ParseOptions(newlines_in_values=True),
            convert_options=pa_csv.ConvertOptions(
                column_types={name: _arrow_type(pa, kind) for name, kind in types.items()},
                # A NULL is an unquoted empty field; a quoted one is empty text.
                null_values=[""], strings_can_be_null=True,
                quoted_strings_can_be_null=False,
                true_values=["t"], false_values=["f"]))

    return layout.tidy(read.to_pandas(date_as_object=False)), problems


# What a Postgres column of each layout kind may be.
DATE_TYPES = frozenset({"date", "timestamp without time zone", "timestamp with time zone"})
NUMBER_TYPES = frozenset({"double precision", "real", "numeric", "integer", "bigint",
                          "smallint"})


def _arrow_type(pa: Any,
                kind: str) -> Any:
    """How a Postgres column type is read."""
    if kind == "date":
        return pa.date32()

    if kind in DATE_TYPES:
        return pa.timestamp("us")

    if kind in NUMBER_TYPES:
        return pa.float64()

    if kind == "boolean":
        return pa.bool_()

    return pa.string()


def _wrong_types(schema: str,
                 table: str,
                 types: dict[str, str]) -> list[layout.Problem]:
    """The layout's date and number columns that the table does not hold
    as dates and numbers."""
    sheet = layout.BY_NAME.get(table)

    if sheet is None:
        return []

    problems = []

    for name, kind in types.items():
        column = layout.sheet_name(name).upper()

        if column in sheet.dates and kind not in DATE_TYPES:
            problems.append(layout.Problem(
                table, None, column, "WRONG_TYPE",
                f"{schema}.{table}.{name} is {kind}; it must be a date (date or "
                f"timestamp)."))
        elif column in sheet.numbers and kind not in NUMBER_TYPES:
            problems.append(layout.Problem(
                table, None, column, "WRONG_TYPE",
                f"{schema}.{table}.{name} is {kind}; it must be a number (double "
                f"precision, real, numeric or an integer type)."))

    return problems


class _Stream:
    """A COPY's output as the file pyarrow reads, a block at a time."""

    def __init__(self,
                 copy: Any):
        self._blocks = iter(copy)
        self._held = b""
        self.closed = False

    def read(self,
             size: int = -1) -> bytes:
        """Up to *size* bytes, or everything left."""
        # Collected and joined once: COPY hands over small blocks, and adding
        # each to what is held copied it all again every time.
        parts = [self._held]
        held = len(self._held)

        while size < 0 or held < size:
            block = next(self._blocks, None)

            if block is None:
                break

            parts.append(bytes(block))
            held += len(parts[-1])

        joined = b"".join(parts)
        cut = len(joined) if size < 0 else size
        self._held = joined[cut:]

        return joined[:cut]


def _is_missing_table(error: Exception) -> bool:
    """Whether the error says the table does not exist.

    Postgres raises UndefinedTable; other databases that speak the same
    interface say "no such table".
    """
    return (type(error).__name__ == "UndefinedTable"
            or "no such table" in str(error).lower()
            or "does not exist" in str(error).lower())
