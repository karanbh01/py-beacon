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
from ..exceptions import ConfigurationError

require("platformdirs", "Document storage for the Beacon API server")

import platformdirs  # noqa: E402

APP_NAME = "beacon"
SCHEMA_VERSION_KEY = "schema_version"

# Bump when a stored shape changes, and add the matching entry to MIGRATIONS.
CURRENT_SCHEMA_VERSION = 1

# version -> function producing the next version's shape. Keyed by the version
# being migrated FROM, so applying 1 turns a v1 document into a v2 one. Empty
# while nothing has changed shape yet; the machinery is here so the first
# change does not have to invent it under time pressure.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


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
        """
        if not document_id:
            raise ValueError("document id cannot be empty.")

        if os.sep in document_id or "/" in document_id or ".." in document_id:
            raise ConfigurationError(
                "document_id",
                f"'{document_id}' is not a valid document id: it must not "
                "contain path separators.")

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
            raise ConfigurationError(
                str(path),
                f"document schema version {version} is newer than this "
                f"application understands ({CURRENT_SCHEMA_VERSION}). "
                "Upgrade py-beacon to read it.")

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
