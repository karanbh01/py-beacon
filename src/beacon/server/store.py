# src/beacon/server/store.py
"""
Versioned JSON document storage on the platform's app-data directory.

The server is a local process with no database. User-authored artefacts —
watchlists now, index definitions and report templates later — are small JSON
documents that must survive a restart and, more importantly, must survive a
schema change without the user losing them. Every document therefore carries a
``schema_version``, and reads run it forward through the migration chain before
it reaches the caller.
"""
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._optional import require
from ..exceptions import (
    ConfigurationError,
    DocumentFromNewerBuildError,
    InvalidIdentifierError,
)
from ..index.schedule import DEFAULT_CALENDAR

# Path segments that name an *endpoint*, and so cannot also name a document.
#
# `POST /indices/preview` previews a definition; `PUT /indices/{index_id}`
# saves one. Those are different routes that overlap on exactly one value, so
# `PUT /indices/preview` used to fall through to the second and store a
# document called "preview" -- and `PUT /optimise/constraint-sets/validate`
# returned 200 having done just that.
#
# The document stayed addressable afterwards, so the harm was small, but the
# URL space was ambiguous: whether `/indices/preview` means an action or a
# document depended on the verb. Reserving the handful of literals costs a few
# names nobody wants and makes the answer the same for every method.
#
# `test_reserved_identifiers_match_the_routes` derives this from the running
# app, so a new endpoint added beside a path parameter cannot silently
# reintroduce the collision.
RESERVED_IDENTIFIERS = frozenset({
    "preview",
    "validate",
    "rule-types",
    "calendars",
})

require("platformdirs", "Document storage for the Beacon API server")

import platformdirs  # noqa: E402

APP_NAME = "beacon"
SCHEMA_VERSION_KEY = "schema_version"

# Bump when a stored shape changes, and add the matching entry to MIGRATIONS.
CURRENT_SCHEMA_VERSION = 3

# The field an index document is discriminated by. One version chain covers
# every collection — watchlists, indices, constraint sets — so a migration that
# only concerns one of them has to say which documents it applies to. This is
# the field no other collection has, and the one an index cannot be without.
_INDEX_MARKER = "rebalancing_frequency"


def _add_default_calendar(document: dict[str, Any]) -> dict[str, Any]:
    """v1 -> v2: an index document without a calendar gets the default.

    BN-180 made `IndexDocument.calendar` required, because null meant Monday to
    Friday and so scheduled rebalances on 1 January, 4 July and 25 December.
    Every index stored before that was implicitly assumed to run on the New
    York calendar, so that is what it is given.

    This redates those indices — a January rebalance moves from the 1st to the
    2nd — which is the point rather than a side effect: the old dates were days
    the data had no session for. A backtest result stored before the migration
    keeps the old dates and will disagree with the index's schedule until it is
    re-run.

    Documents from other collections pass through untouched apart from the
    version stamp: they have no calendar and never had a schedule.
    """
    if _INDEX_MARKER not in document or document.get("calendar"):
        return document

    return {**document, "calendar": DEFAULT_CALENDAR}


# The field a stored job result is discriminated by, on the same reasoning as
# `_INDEX_MARKER`: one version chain covers every collection, so a migration
# that concerns only one has to say which documents it applies to.
_JOB_MARKER = "job_id"

# What a job failure recorded before BN-199 publishes as its code. Not a guess
# at what the failure was: these documents kept `str(exc)` and nothing else, so
# the code genuinely is not known, and inventing a plausible one would be worse
# than saying so. Deliberately not `BEACON_ERROR`, which means "a refusal whose
# subclass nobody registered" and would misreport half of these.
UNCLASSIFIED_FAILURE = "UNCLASSIFIED_FAILURE"


def _wrap_job_error(document: dict[str, Any]) -> dict[str, Any]:
    """v2 -> v3: a job's bare `error` string becomes an `ErrorDetail` body.

    BN-199 made `Job.error` the same `{code, message, detail}` an HTTP error
    carries, so a client can branch on `error.code` instead of reading prose.
    A document written before that holds a string, which no longer validates
    against the response model -- so without this migration one old failed job
    would 500 the whole jobs listing, which is the BN-174 failure exactly.

    The message is kept verbatim; the code says it is unknown. Documents from
    other collections, and jobs that succeeded, pass through untouched.
    """
    if _JOB_MARKER not in document or not isinstance(document.get("error"), str):
        return document

    return {**document,
            "error": {"code": UNCLASSIFIED_FAILURE,
                      "message": document["error"],
                      "detail": None}}


# version -> function producing the next version's shape. Keyed by the version
# being migrated FROM, so applying 1 turns a v1 document into a v2 one.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _add_default_calendar,
    2: _wrap_job_error,
}


class DocumentStore:
    """A namespaced directory of versioned JSON documents.

    Args:
        collection: Subdirectory name, e.g. ``"watchlists"``. Documents from
            different collections never collide.
        root: Base directory. Defaults to the platform app-data location.
            Tests pass a temporary path.
    """

    def __init__(self,
                 collection: str,
                 root: Path | None = None):
        if not collection:
            raise ValueError("collection cannot be empty.")

        base = root if root is not None else Path(platformdirs.user_data_dir(APP_NAME))
        self.directory = base / collection
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self,
              document_id: str) -> Path:
        """Resolve a document id to a path, refusing anything path-like.

        A document id reaches this from a URL path parameter, so an id
        containing a separator or ``..`` could otherwise write outside the
        collection directory.

        Both refusals raise `InvalidIdentifierError`, which maps to 422. They
        used to raise `ValueError` and `ConfigurationError` respectively, and
        both surfaced as 500s -- the guard did its job and then reported
        itself as a server fault. That single function accounted for most of
        the 5xx the first working fuzz run found.
        """
        if not document_id:
            raise InvalidIdentifierError(
                document_id, "it must not be empty.")

        if os.sep in document_id or "/" in document_id or ".." in document_id:
            raise InvalidIdentifierError(
                document_id, "it must not contain path separators.")

        if document_id in RESERVED_IDENTIFIERS:
            raise InvalidIdentifierError(
                document_id,
                f"it is reserved: {', '.join(sorted(RESERVED_IDENTIFIERS))} "
                f"name endpoints rather than documents.")

        return self.directory / f"{document_id}.json"

    def exists(self,
               document_id: str) -> bool:
        """Whether a document with this id is stored."""
        return self._path(document_id).exists()

    def read(self,
             document_id: str) -> dict[str, Any] | None:
        """Read a document, migrating it forward to the current schema.

        Args:
            document_id: Identifier of the document.

        Returns:
            dict or None: The document at the current schema version, or None
            if it does not exist.

        Raises:
            ConfigurationError: If the stored file is not valid JSON, or was
                written by a newer version of the application than this one
                understands.
        """
        path = self._path(document_id)
        if not path.exists():
            return None

        try:
            document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                str(path), f"stored document is not valid JSON: {exc}") from exc

        return self._migrate(document, path)

    def _migrate(self,
                 document: dict[str, Any],
                 path: Path) -> dict[str, Any]:
        """Run a document forward to CURRENT_SCHEMA_VERSION."""
        version = int(document.get(SCHEMA_VERSION_KEY, CURRENT_SCHEMA_VERSION))

        if version > CURRENT_SCHEMA_VERSION:
            raise DocumentFromNewerBuildError(
                str(path),
                f"document schema version {version} is newer than this "
                f"application understands ({CURRENT_SCHEMA_VERSION}). "
                "Upgrade py-beacon-kit to read it.")

        while version < CURRENT_SCHEMA_VERSION:
            migrate = MIGRATIONS.get(version)
            if migrate is None:
                raise ConfigurationError(
                    str(path),
                    f"no migration registered from schema version {version}; "
                    "the document cannot be read.")

            document = migrate(document)
            version += 1
            document[SCHEMA_VERSION_KEY] = version

        return document

    def write(self,
              document_id: str,
              document: dict[str, Any]) -> dict[str, Any]:
        """Write a document, stamping it with the current schema version.

        The write goes to a temporary file in the same directory and is then
        moved into place, so a crash mid-write leaves the previous document
        intact rather than a truncated one.

        Args:
            document_id: Identifier of the document.
            document: Payload to store.

        Returns:
            dict: The stored document, including its schema_version.
        """
        stored = {**document, SCHEMA_VERSION_KEY: CURRENT_SCHEMA_VERSION}
        path = self._path(document_id)

        handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(stored, file, indent=2, sort_keys=True)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

        return stored

    def delete(self,
               document_id: str) -> bool:
        """Delete a document.

        Args:
            document_id: Identifier of the document.

        Returns:
            bool: True if a document was removed, False if none existed.
        """
        path = self._path(document_id)
        if not path.exists():
            return False

        path.unlink()

        return True

    def list_ids(self) -> list[str]:
        """Return every stored document id, sorted."""
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def read_all(self) -> list[dict[str, Any]]:
        """Read every document in the collection, migrating each forward."""
        documents = []
        for document_id in self.list_ids():
            document = self.read(document_id)
            if document is not None:
                documents.append(document)

        return documents
