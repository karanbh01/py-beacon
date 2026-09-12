# src/beacon/server/documents.py
"""
Reading stored documents: tolerant listings, strict details.

A listing and a detail route over the same store answer two different
questions, and BN-174 is what happens when they answer them inconsistently.

**A listing answers "what is there."** A document the server cannot parse or
cannot validate genuinely is not there, as far as anything a client can do with
it goes — so it is skipped, with a WARNING naming the id and the fault, and the
rest of the collection is served. The alternative is what `/indices`,
`/universes` and `/optimise/constraint-sets` used to do: one unreadable file
500d the whole listing, so a picker went blank and every *other* document became
unreachable through the UI because of one bad file.

**A detail route answers "give me this one."** There the only honest reply is a
refusal, and the refusal is 404 with the same envelope and pointer an absent
document gets: a stored artefact the server cannot interpret is
indistinguishable, from the client's side, from one that was never written.
The underlying fault is logged at WARNING rather than being returned, because it
is about this server's storage and not about the request.

These are not two halves of a compromise. They are one pattern, and a pair that
has only half of it diverges — which is the bug this module exists to make
impossible: the listing and the detail now share the same definition of
"unreadable", so they cannot disagree about what exists.

**Write paths need the two questions apart (BN-177).** A read asks "can you give
me this", and a document it cannot parse is answered as absent. A *write* asks
something else, and which something depends on the verb: a delete needs the file
to be PRESENT, not valid, and a PUT needs to know whether a guard that inspects
the predecessor can run at all. Conflating the two is what left a corrupt
universe undeletable *and* unrepairable — 404 on read, 500 on delete, 500 on
overwrite, with the id occupied forever and only a manual file deletion on the
server to clear it. `stored` is the vocabulary for asking explicitly: it hands
back the document when it reads and records the fault when it does not, so
`present` and `readable` are separate properties of one answer and a route
cannot accidentally ask one while meaning the other.

**Why here rather than on `DocumentStore`.** The store deals in dicts: it knows
about JSON and schema versions, and nothing about pydantic models or HTTP. But
most of the documents this guards against are perfectly good JSON — the failure
is that they do not satisfy the response model, which is a fact about the API
layer and is where the forward-compatibility hazard lives (the day a required
field is added to a model, every document stored before it lands on exactly
this path). `store.read_all()` stays strict and keeps its raw-dict callers; the
tolerance is applied where the model is known.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..exceptions import ConfigurationError, DataNotFoundError
from .store import DocumentStore

logger = logging.getLogger(__name__)

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

# What "the server cannot read this document" means, in one place so a listing
# and its detail route cannot drift apart.
#
# `ConfigurationError` is the store's: malformed JSON, or a schema version it
# cannot migrate. `ValidationError` is the model's: valid JSON that is not a
# valid document. Nothing else is caught -- an `InvalidIdentifierError` from the
# path guard is the caller's mistake and must stay a 422.
UNREADABLE: tuple[type[Exception], ...] = (ConfigurationError, ValidationError)


def validated(model: type[ModelT]) -> Callable[[str, dict[str, Any]], ModelT]:
    """A build function that validates a document against *model*.

    The common case: a document that carries its own identity, so the id the
    file was found under adds nothing and is ignored.

    Args:
        model: The response model the stored document must satisfy.

    Returns:
        Callable: ``(document_id, document) -> model`` instance.
    """
    def build(document_id: str,
              document: dict[str, Any]) -> ModelT:
        return model.model_validate(document)

    return build


def raw(document_id: str,
        document: dict[str, Any]) -> dict[str, Any]:
    """A build function that hands back the stored dict unchanged.

    For a caller that wants the document as stored rather than as a model: the
    delete cascade reads derivations off raw dicts, and a guard that only reads
    one key has no reason to hold the whole document to a model.
    """
    return document


@dataclass(frozen=True)
class Stored(Generic[T]):
    """What a collection holds under one id: both questions, answered once.

    Three states, and every write path branches on some pair of them: absent
    (nothing stored), present-but-unreadable (a file is there and the server
    cannot interpret it), and readable (here is the document). A bool alone
    cannot express the middle one, which is precisely the state that made a
    corrupt document undeletable.

    Built from a single `store.read`, so `present` cannot contradict the
    document the same call returned -- an `exists` followed by a `read` can.

    Attributes:
        document: The built document, or None when absent or unreadable.
        fault: Why it could not be read; None when it read, and also None when
            nothing is stored. `fault is not None` *is* "present but
            unreadable".
    """

    document: T | None = None
    fault: Exception | None = None

    @property
    def present(self) -> bool:
        """Whether something is stored under this id, readable or not."""
        return self.document is not None or self.fault is not None

    @property
    def readable(self) -> bool:
        """Whether the stored document could be read and built."""
        return self.document is not None

    def warn_guard_skipped(self,
                           guard: str,
                           describe: str) -> None:
        """Log a guard that this document's fault made impossible to run.

        A no-op when the document is absent or readable: there is no guard to
        skip in either case.

        The log line is the only record, and it has to name both halves -- which
        check did not run, and why -- because the request succeeds and the
        response says nothing about it. Skipping is the lesser evil (a guard
        that cannot run must not make a document permanently unfixable), but it
        is still a guard that did not run.

        Args:
            guard: The check that was skipped, as a noun phrase.
            describe: The document it would have guarded, e.g. ``"universe
                'tech'"``.
        """
        if self.fault is None:
            return

        logger.warning("Skipped %s for %s: the stored document cannot be read "
                       "(%s). Proceeding, because a document the server cannot "
                       "read would otherwise be impossible to delete or "
                       "replace through the API.",
                       guard, describe, self.fault)


def stored(store: DocumentStore,
           document_id: str,
           build: Callable[[str, dict[str, Any]], T]) -> Stored[T]:
    """Ask what a collection holds under an id, without conflating the answers.

    The write-path counterpart to `load_document`: where a read turns an
    unreadable document into a not-found, a write needs the distinction kept,
    so this one refuses nothing and reports.

    Args:
        store: The collection to look in.
        document_id: Identifier of the document.
        build: Turns ``(document_id, document)`` into whatever the caller needs
            -- `raw` for the stored dict, `validated(Model)` for a model.

    Returns:
        Stored: The document, or the fault that stopped it being read.
    """
    try:
        document = store.read(document_id)
    except UNREADABLE as error:
        return Stored(fault=error)

    if document is None:
        return Stored()

    try:
        return Stored(document=build(document_id, document))
    except UNREADABLE as error:
        return Stored(fault=error)


def read_collection(store: DocumentStore,
                    build: Callable[[str, dict[str, Any]], T],
                    describe: str) -> tuple[list[T], int]:
    """Read every document in a collection, skipping the unreadable ones.

    Args:
        store: The collection to read.
        build: Turns ``(document_id, document)`` into the listing's row.
            `validated` covers the usual case.
        describe: What these documents are, for the log line, singular:
            "universe", "index definition".

    Returns:
        tuple: ``(rows, skipped)``. The count is published by the response
        model rather than only logged: a picker silently short by three is
        indistinguishable from a correct one, and the server is the only side
        that knows.
    """
    rows: list[T] = []
    skipped = 0

    for document_id in store.list_ids():
        try:
            document = store.read(document_id)

            # Deleted between listing the ids and reading it. Not a fault, and
            # not a skip: it is simply gone, which is what the listing will say.
            if document is None:
                continue

            rows.append(build(document_id, document))
        except UNREADABLE as error:
            logger.warning("Skipping unreadable %s '%s': %s",
                           describe, document_id, error)
            skipped += 1

    return rows, skipped


def load_document(store: DocumentStore,
                  document_id: str,
                  build: Callable[[str, dict[str, Any]], T],
                  describe: str,
                  source: str = "DocumentStore") -> T:
    """Read one document, answering not-found when it cannot be read.

    Args:
        store: The collection to read from.
        document_id: Identifier of the document.
        build: Turns ``(document_id, document)`` into the response model.
        describe: The subject of the not-found message, e.g. ``"universe
            'tech'"``.
        source: The pointer the not-found message carries. Defaults to the
            store; a route with somewhere better to send the client (running a
            backtest, say) passes its own.

    Returns:
        Whatever *build* returns.

    Raises:
        DataNotFoundError: If the document is absent, is not valid JSON, or
            does not satisfy its model. One answer for all three, because a
            client can act on none of them differently.
    """
    try:
        document = store.read(document_id)
    except UNREADABLE as error:
        raise _absent(describe, source, error) from error

    if document is None:
        raise DataNotFoundError(describe, source=source)

    try:
        return build(document_id, document)
    except UNREADABLE as error:
        raise _absent(describe, source, error) from error


def _absent(describe: str,
            source: str,
            error: Exception) -> DataNotFoundError:
    """The not-found an unreadable document earns, logged on the way out."""
    logger.warning("Answering not-found for unreadable %s: %s", describe, error)

    return DataNotFoundError(describe, source=source)
