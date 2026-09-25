# tests/test_server_listing_discovery.py
"""BN-178: every store-backed listing survives a bad file — discovered, not listed.

This is the third time one unreadable document has been found blanking a
collection. BN-174 fixed four listings, BN-177 fixed the write paths beside
them, and BN-178 found three more still broken plus a detail route that read a
collection on the side. Every round fixed the endpoints somebody had thought of,
because the property only ever existed as a list of endpoints somebody had
thought of. A listing written next week would meet nothing.

So this file states the property instead, over whatever the app actually holds:

    For every `DocumentStore` the running app owns, some listing serves the rest
    of that collection when a file in it cannot be read, and says how many it
    left out.

Nothing here names a collection or a route. Both sides are discovered:

* **The stores** by walking `app.state` — and one level inside each value it
  holds, which is what reaches the `JobRegistry`'s results collection, the one
  store that is not an attribute of the app.
* **The routes** by walking the app's own OpenAPI document, which is the same
  description clients generate from, so a route absent from it is a route no
  client can call.

A store added to the app is therefore covered the day it is added, and a listing
that forgets the pattern fails here rather than in somebody's UI: it will either
500 under `test_no_get_route_500s_because_of_an_unreadable_document` or serve a
truncated collection silently under `test_every_store_has_a_listing_that_...`.

**What discovery cannot reach**, stated rather than papered over: a collection
read through a store that never lands on `app.state` — built inside a route, say
— is invisible to the walk, and so is any fault that needs a path parameter this
file has no way to guess. `test_no_server_code_reads_a_collection_strictly` is
the backstop for both. It parses the package and fails on any surviving
`read_all()` call, which is the one API that turns a single bad file into a
failed request, wherever it is called from.

The sweep is GET-only on purpose. A write path asks different questions of the
same document — BN-177 — and its answers are pinned down in
`test_server_document_faults.py`, where a delete and a PUT can be asserted
individually instead of swept.
"""
import ast
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import beacon
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.store import DocumentStore

TOKEN = "discovery-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Substituted for every path parameter in the sweep. An instrument the dataset
# below carries, so a detail route reaches its handler rather than bouncing off
# a validator -- the point of the sweep is the code behind the route.
PLACEHOLDER = "AAA"

# Two documents, both unreadable, in the two ways a document can be. The second
# is the forward-compatibility shape: perfectly good JSON that satisfies no
# model, which is what every document stored before a model gains a required
# field turns into. Deliberately carrying no field any Beacon model declares, so
# it cannot accidentally be valid for some collection.
UNPARSEABLE = ('{"id": "wreckage", "nam', "wreckage")
INVALID = ('{"beacon_discovery_probe": true}', "nonsense")

# Lower bounds, not exact counts: adding a store or a route must not fail this
# file -- covering them automatically is the whole point. What these catch is
# discovery silently finding nothing, which would leave every test here passing
# over an empty sweep. Measured when BN-178 was written: 7 stores, 42 GET routes.
LEAST_STORES = 7
LEAST_GET_ROUTES = 40


def build_fetcher() -> DataFetcher:
    """One instrument over a short window.

    Enough that the data routes answer from their handlers instead of refusing
    for want of a data source, which would hide a fault behind a 500 of its own.
    """
    reference = pd.DataFrame([{"IDENTIFIER": PLACEHOLDER,
                               "DATE_FROM": "2020-01-01",
                               "NAME": "Alpha Corp",
                               "CURRENCY": "USD",
                               "EXCHANGE": "NYSE"}])
    market = pd.DataFrame([{"IDENTIFIER": PLACEHOLDER, "DATE": date, "CLOSE": 100.0}
                           for date in pd.bdate_range("2025-01-02", periods=10)])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def build_app():
    """A server on a fresh storage root, with a data source."""
    return create_app(ServerConfig(auth_token=TOKEN,
                                   storage_root=Path(tempfile.mkdtemp()),
                                   data_fetcher=build_fetcher()))


def document_stores(app) -> dict[str, DocumentStore]:
    """Every collection the app holds, by the name it is reachable under.

    One level deep as well as at the top, because the job registry owns a store
    rather than the app doing so — and a registry is exactly the shape a future
    component that owns a collection will have. `app.state` keeps its contents
    in `_state`; there is no public iteration over a Starlette `State`.

    Returns:
        dict: ``"index_store"`` or ``"jobs._results"`` to the store.
    """
    found: dict[str, DocumentStore] = {}

    for name, value in app.state._state.items():
        if isinstance(value, DocumentStore):
            found[name] = value
            continue

        for inner_name, inner in vars(value).items() if hasattr(value, "__dict__") else ():
            if isinstance(inner, DocumentStore):
                found[f"{name}.{inner_name}"] = inner

    return found


def get_paths(app) -> list[str]:
    """Every path the app publishes a GET for, template form."""
    return sorted(path for path, operations in app.openapi()["paths"].items()
                  if "get" in operations)


def listing_paths(app) -> list[str]:
    """The GET paths that take no parameter — a collection's own URL."""
    return [path for path in get_paths(app) if "{" not in path]


def wreck(store: DocumentStore) -> int:
    """Write both kinds of unreadable document into a collection.

    Returns:
        int: How many were written, which is what a listing over this
        collection must now report as skipped.
    """
    for content, document_id in (UNPARSEABLE, INVALID):
        (store.directory / f"{document_id}.json").write_text(content, encoding="utf-8")

    return 2


def sweep(client,
          paths: list[str]) -> dict[str, int]:
    """GET every path, substituting the placeholder for path parameters."""
    statuses = {}

    for path in paths:
        url = path
        while "{" in url:
            url = url[:url.index("{")] + PLACEHOLDER + url[url.index("}") + 1:]

        statuses[path] = client.get(url, headers=HEADERS).status_code

    return statuses


def skip_counts(client,
                paths: list[str]) -> dict[str, int]:
    """What each listing that answered says it left out.

    A path missing from the result either did not answer 200 or publishes no
    count, and both are failures of the property — reported by the test rather
    than smoothed over here.
    """
    counts = {}

    for path in paths:
        response = client.get(path, headers=HEADERS)

        if response.status_code != 200:
            continue

        # A listing is JSON. A download (the import template is a workbook)
        # is not a listing, and has no count to report.
        if not response.headers.get("content-type", "").startswith("application/json"):
            continue

        body = response.json()

        if isinstance(body, dict) and isinstance(body.get("skipped"), int):
            counts[path] = body["skipped"]

    return counts


@pytest.fixture
def clean():
    """A server whose collections are all empty and readable."""
    app = build_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestDiscoveryItself:
    """The walks have to actually find things.

    Without these, every other test in this file would pass vacuously the day a
    refactor moved the stores behind something the walk does not open — which is
    the failure mode that makes a discovery test worse than an enumerated one.
    """

    def test_the_walk_finds_the_stores_the_app_holds(self,
                                                     clean):
        found = document_stores(clean.app)

        assert len(found) >= LEAST_STORES, sorted(found)

    def test_the_walk_reaches_a_store_the_app_does_not_hold_directly(self,
                                                                     clean):
        """The job registry owns its results collection. A walk that only read
        `app.state`'s own values would have missed the listing BN-178 found
        broken, which is precisely how it stayed broken."""
        owned = [name for name in document_stores(clean.app) if "." in name]

        assert owned, sorted(document_stores(clean.app))

    def test_the_walk_finds_the_routes_the_app_serves(self,
                                                      clean):
        assert len(get_paths(clean.app)) >= LEAST_GET_ROUTES

    def test_every_discovered_store_resolves_to_a_real_directory(self,
                                                                 clean):
        """A store the walk found but cannot write to would make `wreck` a
        no-op and the property untested."""
        for name, store in document_stores(clean.app).items():
            assert store.directory.is_dir(), name


class TestTheProperty:

    def test_no_get_route_500s_because_of_an_unreadable_document(self):
        """The whole read surface, not the listings somebody remembered.

        Measured against a clean baseline rather than against 200: a route that
        refuses for its own reasons is not this file's business, and a route
        that starts 500ing only once a document is unreadable is exactly its
        business. `/data/reference/{identifier}` is the one BN-178 found this
        way — a detail route that scanned the universe collection on the side,
        so a bad file in a collection it does not serve took out reference data
        for every instrument.
        """
        app = build_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            paths = get_paths(app)
            baseline = sweep(client, paths)

        app = build_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            for store in document_stores(app).values():
                wreck(store)

            after = sweep(client, paths)

        broken = [path for path in paths
                  if after[path] == 500 and baseline[path] != 500]

        assert not broken, ("These GET routes answer 500 only when a stored "
                            f"document cannot be read: {broken}. Read the "
                            "collection through beacon.server.documents."
                            "read_collection, which skips and counts, instead "
                            "of DocumentStore.read_all, which raises.")

    @pytest.mark.parametrize("target", sorted(document_stores(build_app())))
    def test_every_store_has_a_listing_that_survives_and_counts_its_bad_files(self,
                                                                             target):
        """One store at a time, so the count is attributable.

        Two claims in one assertion, and they are the two halves of the pattern.
        That a listing still answers is the first: a collection must not be lost
        because one file in it is. That its `skipped` moved by exactly the
        number of unreadable files is the second, and it is the half a tolerant
        listing is most likely to be written without — a picker silently three
        short is indistinguishable from a correct one, and the server is the
        only side that knows.

        Parametrised over a throwaway app built at collection time, so a new
        store gets its own failing case by existing rather than by being added
        to a list here.
        """
        app = build_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            paths = listing_paths(app)
            before = skip_counts(client, paths)

            written = wreck(document_stores(app)[target])

            after = skip_counts(client, paths)

        moved = {path: after[path] - before.get(path, 0)
                 for path in after
                 if after[path] != before.get(path, 0)}

        assert moved == dict.fromkeys(moved, written), (
            f"{target}: expected one listing to report {written} skipped "
            f"documents, got {moved}.")
        assert moved, (
            f"No listing accounted for the unreadable documents in {target}. "
            "Either its listing 500d — check the sweep — or it skips them "
            "without publishing a `skipped` count, which leaves a client "
            "unable to tell a short collection from a complete one.")


class TestTheBackstopForWhatDiscoveryCannotSee:
    """A store built inside a route reaches no walk, and neither does a fault
    that needs a path parameter nothing here can guess. What both have in
    common is the call: `DocumentStore.read_all` is the only API that turns one
    unreadable file into a failed request, so its absence is checkable
    everywhere at once, including in code no request reaches.
    """

    # Call sites that have been looked at and are deliberately strict. Empty,
    # and adding to it is a decision to be argued in a commit message: the
    # tolerant readers cover every shape found so far, including the two that
    # want a raw dict and the one that wants to see the fault.
    ALLOWED: frozenset = frozenset()

    def test_no_server_code_reads_a_collection_strictly(self):
        """Parsed rather than grepped, so the two places that *discuss*
        `read_all()` in prose — a docstring and a comment, both explaining why
        it is not used — are not mistaken for callers."""
        package = Path(beacon.__file__).parent
        callers = []

        for path in sorted(package.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

            callers.extend(
                f"{path.relative_to(package).as_posix()}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "read_all")

        unexpected = [caller for caller in callers if caller not in self.ALLOWED]

        assert not unexpected, (
            f"read_all() raises on the first unreadable document, so these "
            f"call sites turn one bad file into a failed request: {unexpected}. "
            "Use beacon.server.documents.read_collection for a listing, "
            "load_document for a detail route, or stored() when the caller "
            "genuinely needs to know a document is present but unreadable.")
