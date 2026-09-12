# tests/test_server_document_faults.py
"""BN-174: what a listing and its detail route say about a document neither can read.

Four pairs read four document stores, and before this they disagreed in two
different ways. `/beacon/backtests` skipped what it could not read while
`/beacon/{id}/record` 500d on the same file, so a record could be listed and
then refused. `/indices`, `/universes` and `/optimise/constraint-sets` were
consistent and brittle instead: one unreadable document 500d the ENTIRE listing,
so the picker went blank and every other document became unreachable through the
UI because of one bad file.

The fix is one pattern applied to all four, so this is one parameterised matrix
rather than four sets of near-identical tests — if a pair ever drifts, a row
here fails for that pair alone:

| document state             | listing            | detail |
| -------------------------- | ------------------ | ------ |
| complete                   | listed             | 200    |
| unparseable JSON           | skipped, counted   | 404    |
| parseable, missing a field | skipped, counted   | 404    |
| absent                     | not listed         | 404    |

**The third row is the one that matters most.** It is a forward-compatibility
hazard, not a curiosity: the day a required field is added to any of these
models, every document stored before it lands on exactly that path. Discipline
about optional fields is what keeps it rare; this is what keeps it survivable.

Documents are written straight into the stores, as the probe on the issue did.
Nothing that goes through an endpoint could produce these states — every store
is written through a model that validates — so they arise from an interrupted
write, manual editing, or a schema change.
"""
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.server.schemas import (
    BacktestMetrics,
    BacktestResultSummary,
    ConstraintSet,
    PortfolioBookPayload,
    SeriesPayload,
    TableFrame,
    Universe,
)

TOKEN = "faults-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Ids used in every pair. `GOOD` is the document that must survive whatever
# happens to the others — the brittleness case is not "a bad row is lost" but
# "every good row is lost with it".
GOOD = "good"
UNPARSEABLE = "unparseable"
INVALID = "invalid"
ABSENT = "absent"

EMPTY_TABLE = TableFrame(index=[], columns=[], data=[])
EMPTY_SERIES = SeriesPayload(index=[], data=[])


def index_document(document_id: str = GOOD) -> dict:
    """A valid stored index definition."""
    return {
        "id": document_id,
        "name": "Valid index",
        "base_date": "2023-01-02",
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "universe": {"universe_id": None, "identifiers": ["AAA", "BBB"]},
        "pipeline": {"selection": [],
                     "weighting": {"id": "weighting",
                                   "scheme": "EqualWeighted",
                                   "params": {},
                                   "max_weight": None},
                     "treatment": {"corporate_actions": "ADJUST_DIVISOR"}},
    }


def universe_document(document_id: str = GOOD) -> dict:
    """A valid stored universe."""
    return Universe(id=document_id,
                    name="Valid universe",
                    identifiers=["AAA", "BBB"]).model_dump()


def constraint_set_document(document_id: str = GOOD) -> dict:
    """A valid stored constraint set."""
    return ConstraintSet(id=document_id, name="Valid set").model_dump()


def record_document(document_id: str = GOOD) -> dict:
    """A valid stored backtest record.

    Assembled from the payload models rather than from a run: the subject is
    what the server does with a document, and a real backtest would cost a
    calculation per test. The id is unused — a record is addressed by the index
    it belongs to, and carries no id of its own, which is why the listing needs
    the filename to build its row.
    """
    return BacktestResultSummary(
        portfolio=PortfolioBookPayload(portfolio_id=document_id,
                                       initial_capital=1_000_000.0,
                                       nav=EMPTY_SERIES,
                                       cash=EMPTY_SERIES,
                                       weights=EMPTY_TABLE,
                                       weights_dates_total=0,
                                       positions=EMPTY_TABLE,
                                       positions_total=0,
                                       transactions=EMPTY_TABLE),
        metrics=BacktestMetrics(total_return=0.1,
                               annualised_return=0.05,
                               volatility=0.12,
                               sharpe_ratio=0.4,
                               max_drawdown=-0.08),
        run_at="2025-01-01T00:00:00+00:00").model_dump(mode="json")


# One row per listing/detail pair. `invalid` is a document that is valid JSON
# and not a valid document -- in every case because a required field is missing,
# which is the shape a model gaining a field produces.
PAIRS = [
    pytest.param({"store": "index_store",
                  "listing": "/indices",
                  "rows": "indices",
                  "detail": "/indices/{id}",
                  "row_id": "id",
                  "valid": index_document,
                  "invalid": {"id": INVALID, "name": "No base date"}},
                 id="indices"),
    pytest.param({"store": "universe_store",
                  "listing": "/universes",
                  "rows": "universes",
                  "detail": "/universes/{id}",
                  "row_id": "id",
                  "valid": universe_document,
                  "invalid": {"id": INVALID}},
                 id="universes"),
    pytest.param({"store": "constraint_store",
                  "listing": "/optimise/constraint-sets",
                  "rows": "constraint_sets",
                  "detail": "/optimise/constraint-sets/{id}",
                  "row_id": "id",
                  "valid": constraint_set_document,
                  "invalid": {"id": INVALID}},
                 id="constraint-sets"),
    pytest.param({"store": "backtest_record_store",
                  "listing": "/beacon/backtests",
                  "rows": "backtests",
                  "detail": "/beacon/{id}/record",
                  "row_id": "index_id",
                  "valid": record_document,
                  # The skeletal record from the issue, exactly: enough for the
                  # store, and nothing the record endpoint can serve.
                  "invalid": {"run_at": "2025-01-01T00:00:00+00:00"}},
                 id="backtest-records"),
]


@pytest.fixture
def client():
    """A server with empty stores and no data source.

    No data source deliberately: every route under test is document CRUD, and
    an unconfigured process also leaves the `GLOBAL` universe unseeded, so each
    listing holds exactly the documents a test wrote.
    """
    root = tempfile.mkdtemp()
    app = create_app(ServerConfig(auth_token=TOKEN, storage_root=Path(root)))

    with TestClient(app, raise_server_exceptions=False) as entered:
        yield entered


def store_of(client,
             pair) -> object:
    """The document store behind a pair."""
    return getattr(client.app.state, pair["store"])


def write_unparseable(client,
                      pair,
                      document_id: str = UNPARSEABLE) -> None:
    """Leave a truncated file in the collection, as an interrupted write would."""
    path = store_of(client, pair).directory / f"{document_id}.json"
    path.write_text('{"id": "' + document_id + '", "nam', encoding="utf-8")


def listing(client,
            pair) -> dict:
    """GET the listing, asserting it answered at all."""
    response = client.get(pair["listing"], headers=HEADERS)

    assert response.status_code == 200, response.text

    return response.json()


def listed_ids(body,
               pair) -> list[str]:
    """The ids a listing body offers."""
    return [row[pair["row_id"]] for row in body[pair["rows"]]]


def detail(client,
           pair,
           document_id: str):
    """GET one document's detail route."""
    return client.get(pair["detail"].format(id=document_id), headers=HEADERS)


@pytest.mark.parametrize("pair", PAIRS)
class TestTheMatrix:
    """Every row of the table, for every pair.

    The pairs are asserted against the same expectations rather than each
    against its own, which is the point of the issue: "records was never too
    tolerant, it had half the pattern". A pair that needed different
    expectations would be a pair that had drifted.
    """

    def test_a_complete_document_is_listed_and_served(self,
                                                      client,
                                                      pair):
        store_of(client, pair).write(GOOD, pair["valid"](GOOD))

        body = listing(client, pair)

        assert listed_ids(body, pair) == [GOOD]
        assert body["skipped"] == 0
        assert detail(client, pair, GOOD).status_code == 200

    def test_an_unparseable_document_is_skipped_and_404s(self,
                                                         client,
                                                         pair):
        """The brittleness case. What is lost must be the bad document and
        nothing else: a listing that 500s costs the reader every other
        document, which is worse than being offered a row that then fails."""
        store_of(client, pair).write(GOOD, pair["valid"](GOOD))
        write_unparseable(client, pair)

        body = listing(client, pair)

        assert listed_ids(body, pair) == [GOOD]
        assert body["skipped"] == 1

        response = detail(client, pair, UNPARSEABLE)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_a_document_missing_a_required_field_is_skipped_and_404s(self,
                                                                     client,
                                                                     pair):
        """The forward-compatibility row: the day a required field is added to
        a model, every document stored before it reaches exactly this path."""
        store_of(client, pair).write(INVALID, pair["invalid"])
        store_of(client, pair).write(GOOD, pair["valid"](GOOD))

        body = listing(client, pair)

        assert listed_ids(body, pair) == [GOOD]
        assert body["skipped"] == 1

        response = detail(client, pair, INVALID)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_an_absent_document_is_not_listed_and_404s(self,
                                                       client,
                                                       pair):
        """The row the other three are measured against: a document the server
        cannot read is answered exactly as one that was never written, because
        a client can act on neither differently."""
        body = listing(client, pair)

        assert listed_ids(body, pair) == []
        assert body["skipped"] == 0

        response = detail(client, pair, ABSENT)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_an_unreadable_document_answers_exactly_as_an_absent_one(self,
                                                                     client,
                                                                     pair):
        """Not merely the same status: the same envelope and the same pointer.

        A 404 that reads differently is a client branching on prose, and the
        whole claim of this issue is that the two states are indistinguishable
        from outside.
        """
        store_of(client, pair).write(INVALID, pair["invalid"])

        unreadable = detail(client, pair, INVALID).json()["error"]
        missing = detail(client, pair, ABSENT).json()["error"]

        assert unreadable["code"] == missing["code"]
        assert unreadable["detail"]["source"] == missing["detail"]["source"]
        # The subject differs only by the id, which is what each names.
        assert (unreadable["message"].replace(INVALID, ABSENT)
                == missing["message"])

    def test_the_skip_count_counts_every_skipped_document(self,
                                                          client,
                                                          pair):
        """A count, not a flag. "12 of 15" is a sentence a client can show; a
        boolean leaves the reader unable to tell how much is missing."""
        store_of(client, pair).write(GOOD, pair["valid"](GOOD))
        store_of(client, pair).write(INVALID, pair["invalid"])
        write_unparseable(client, pair)
        write_unparseable(client, pair, "also-unparseable")

        body = listing(client, pair)

        assert listed_ids(body, pair) == [GOOD]
        assert body["skipped"] == 3

    def test_the_listing_logs_each_skip_with_the_id_and_the_reason(self,
                                                                   client,
                                                                   pair,
                                                                   caplog):
        """The fault is not returned — it is about this server's storage, not
        about the request — so the log is the only place it exists. A skip
        nobody can diagnose is a document quietly deleted."""
        store_of(client, pair).write(INVALID, pair["invalid"])

        with caplog.at_level("WARNING"):
            listing(client, pair)

        assert any(INVALID in record.getMessage()
                   for record in caplog.records
                   if record.levelname == "WARNING"), caplog.text

    def test_the_detail_logs_the_fault_it_does_not_return(self,
                                                          client,
                                                          pair,
                                                          caplog):
        store_of(client, pair).write(INVALID, pair["invalid"])

        with caplog.at_level("WARNING"):
            assert detail(client, pair, INVALID).status_code == 404

        assert any(record.levelname == "WARNING" for record in caplog.records)


class TestTheRecordPairInParticular:
    """The divergence the issue was filed for, with its own pointer intact.

    The record endpoint's 404 carries advice the other three do not — run the
    backtest — and an unreadable record must get the same advice rather than a
    generic not-found, or the 404 the fix introduces would be a worse answer
    than the 500 it replaced.
    """

    PAIR = PAIRS[-1].values[0]

    def test_an_unreadable_record_keeps_the_run_a_backtest_pointer(self,
                                                                   client):
        store_of(client, self.PAIR).write(INVALID, self.PAIR["invalid"])

        body = detail(client, self.PAIR, INVALID).json()

        assert "backtest" in body["error"]["message"]

    def test_the_listing_shape_carries_the_count(self,
                                                 client):
        """The wire change: a bare array had nowhere to put it."""
        body = listing(client, self.PAIR)

        assert sorted(body) == ["backtests", "skipped"]

    def test_the_skeletal_record_from_the_issue_is_no_longer_listed(self,
                                                                    client):
        """The measured finding: `{"run_at": ...}` was listed and then 500d.

        The listing validated only `run_at` while the endpoint validated the
        whole payload, so the two surfaces could not both be acted on. They now
        apply one standard — the endpoint's.
        """
        records = store_of(client, self.PAIR)
        records.write("skeletal", {"run_at": "2025-06-01T00:00:00+00:00"})

        body = listing(client, self.PAIR)

        assert body["backtests"] == []
        assert body["skipped"] == 1
        assert detail(client, self.PAIR, "skeletal").status_code == 404


class TestTheStoreStaysStrict:
    """The tolerance lives beside the routers, not in `DocumentStore`.

    Deliberate: the store deals in dicts and knows nothing about the models
    most of these documents fail against, and its raw-dict callers — the delete
    cascade reads derivations off stored documents — want the unvalidated shape.
    A store that swallowed faults would also make a single-document `read()`
    return None for a file that is present, which is the confusion between
    absence and fault this issue exists to stop.
    """

    def test_read_still_raises_on_an_unparseable_document(self,
                                                          client):
        from beacon.exceptions import ConfigurationError

        store = client.app.state.universe_store
        (store.directory / "broken.json").write_text("{", encoding="utf-8")

        with pytest.raises(ConfigurationError):
            store.read("broken")

    def test_an_invalid_document_reads_back_as_stored(self,
                                                      client):
        """The store does not validate, and must not start: it is what a
        migration reads from."""
        store = client.app.state.universe_store
        store.write("partial", {"id": "partial"})

        assert store.read("partial")["id"] == "partial"


UNIVERSES = PAIRS[1].values[0]
INDICES = PAIRS[0].values[0]


def derived_document(document_id: str,
                     source_index_id: str) -> dict:
    """A stored index definition whose methodology is a derivation."""
    document = index_document(document_id)
    document.pop("universe")
    document["pipeline"] = None
    document["derivation"] = {"source_index_id": source_index_id,
                              "objective": "min_tracking_error",
                              "constraints": []}

    return document


class TestAWriteCanFixWhatAReadCannotServe:
    """BN-177: the write paths BN-174 deliberately left alone.

    BN-174 made a read answer not-found for a document it cannot interpret,
    which is right for a read and left every *write* path going through the same
    strict read for its guards. The result, measured on a corrupt universe: 404
    on read, 500 on delete, 500 on overwrite. Three answers, no action available,
    and the id occupied until somebody deletes the file on the server.

    The fix is to stop conflating two questions. A delete needs the file to be
    PRESENT, not valid. A PUT needs to know whether a guard on the predecessor
    can run at all — and when it cannot, running it is impossible and refusing
    makes the document permanently unfixable, so it is skipped and logged. A
    guard that *can* run still runs, which the last two tests here pin down:
    weakening it for readable documents would trade one bug for a worse one.
    """

    def test_a_corrupt_universe_can_be_deleted_and_its_id_reused(self,
                                                                 client):
        """The actual point is freeing the id, so this asserts the reuse.

        A 204 that left the name taken would fix nothing a client can see.
        """
        write_unparseable(client, UNIVERSES)

        removed = client.delete(f"/universes/{UNPARSEABLE}", headers=HEADERS)

        assert removed.status_code == 204, removed.text

        recreated = client.post("/universes",
                                json={"name": UNPARSEABLE,
                                      "identifiers": ["AAA"]},
                                headers=HEADERS)

        assert recreated.status_code == 201, recreated.text
        assert recreated.json()["id"] == UNPARSEABLE
        assert client.get(f"/universes/{UNPARSEABLE}",
                          headers=HEADERS).status_code == 200

    def test_a_universe_missing_a_required_field_can_be_deleted(self,
                                                               client):
        """The forward-compatibility shape, not only a truncated file: a model
        that gains a required field makes every older document land here."""
        store_of(client, UNIVERSES).write(INVALID, UNIVERSES["invalid"])

        response = client.delete(f"/universes/{INVALID}", headers=HEADERS)

        assert response.status_code == 204, response.text
        assert not store_of(client, UNIVERSES).exists(INVALID)

    def test_deleting_an_absent_universe_is_still_404(self,
                                                      client):
        """Existence is the question the delete now asks, so it still has to
        answer it: "present" must not quietly become "anything at all"."""
        response = client.delete(f"/universes/{ABSENT}", headers=HEADERS)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_a_corrupt_universe_can_be_replaced_and_reads_back(self,
                                                              client):
        """A PUT over an unreadable document is a repair: it carries a complete
        valid replacement, so the result has to be readable afterwards."""
        write_unparseable(client, UNIVERSES)

        replaced = client.put(f"/universes/{UNPARSEABLE}",
                              json={"name": "Repaired",
                                    "identifiers": ["AAA", "BBB"]},
                              headers=HEADERS)

        assert replaced.status_code == 200, replaced.text

        served = client.get(f"/universes/{UNPARSEABLE}", headers=HEADERS)

        assert served.status_code == 200
        assert served.json()["name"] == "Repaired"
        assert served.json()["identifiers"] == ["AAA", "BBB"]

    def test_skipping_the_read_only_check_is_logged_with_the_reason(self,
                                                                    client,
                                                                    caplog):
        """The request succeeds and says nothing about the guard that did not
        run, so the log is the only record that one was skipped."""
        write_unparseable(client, UNIVERSES)

        with caplog.at_level("WARNING"):
            client.delete(f"/universes/{UNPARSEABLE}", headers=HEADERS)

        warnings = [record.getMessage() for record in caplog.records
                    if record.levelname == "WARNING"]

        assert any("seeded" in message and UNPARSEABLE in message
                   for message in warnings), caplog.text

    def test_a_readable_seeded_universe_is_still_refused(self,
                                                         client):
        """The guard is skipped only because it cannot run. A document that
        reads is held to it exactly as before — otherwise this issue would have
        made every generator-written universe editable."""
        store_of(client, UNIVERSES).write("seeded-one",
                                          {**universe_document("seeded-one"),
                                           "source": "seeded"})

        removed = client.delete("/universes/seeded-one", headers=HEADERS)
        replaced = client.put("/universes/seeded-one",
                              json={"name": "Hijacked", "identifiers": ["AAA"]},
                              headers=HEADERS)

        assert removed.status_code == 422, removed.text
        assert "read-only" in removed.json()["error"]["message"]
        assert replaced.status_code == 422, replaced.text
        assert "read-only" in replaced.json()["error"]["message"]
        assert store_of(client, UNIVERSES).read("seeded-one")["source"] == "seeded"

    def test_a_corrupt_index_can_be_replaced_and_reads_back(self,
                                                            client):
        """`PUT /indices/{id}` held `derivation.source_index_id` still by
        reading the stored document, so an unreadable predecessor 500d."""
        write_unparseable(client, INDICES)

        replaced = client.put(f"/indices/{UNPARSEABLE}",
                              json=index_document(UNPARSEABLE),
                              headers=HEADERS)

        assert replaced.status_code == 200, replaced.text

        served = client.get(f"/indices/{UNPARSEABLE}", headers=HEADERS)

        assert served.status_code == 200
        assert served.json()["name"] == "Valid index"

    def test_a_readable_derivation_still_cannot_be_repointed(self,
                                                            client):
        """The other half of the same claim, for the index guard."""
        store_of(client, INDICES).write("child", derived_document("child", "one"))

        response = client.put("/indices/child",
                              json=derived_document("child", "two"),
                              headers=HEADERS)

        assert response.status_code == 422, response.text
        assert "re-pointing" in response.json()["error"]["message"]

    def test_a_taken_and_readable_id_still_reads_as_a_name_clash(self,
                                                                client):
        """The ordinary collision is unchanged: this is the message every
        client has been rendering, and nothing about it was wrong."""
        store_of(client, INDICES).write("parent", index_document("parent"))

        response = client.post("/indices/parent/optimise",
                               json={"id": "parent", "name": "Optimised"},
                               headers=HEADERS)

        assert response.status_code == 409, response.text
        assert "already exists" in response.json()["error"]["message"]

    def test_a_taken_but_unreadable_id_says_so_and_names_the_way_out(self,
                                                                     client):
        """The 409 and the 404 are each correct and together unactionable: the
        client is told the index does not exist AND that its id is taken. Only
        the message can say which, and delete is the way out.

        Prose, deliberately — no new code and no flag on the envelope, so this
        asserts the wording and the unchanged code rather than a machine-
        readable bit (see `_taken` for why).
        """
        store_of(client, INDICES).write("parent", index_document("parent"))
        write_unparseable(client, INDICES, "wreckage")

        response = client.post("/indices/parent/optimise",
                               json={"id": "wreckage", "name": "Optimised"},
                               headers=HEADERS)
        message = response.json()["error"]["message"]

        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "CONFLICT"
        assert "cannot be read" in message
        assert "DELETE /indices/wreckage" in message
        assert client.get("/indices/wreckage",
                          headers=HEADERS).status_code == 404

    def test_an_unreadable_derivation_source_is_404_not_500(self,
                                                            client):
        """The one other write path with the same fault: saving a child reads
        its parent to resolve the derivation, and did that strictly. An
        unreadable parent is a child that can never be calculated, so it is a
        refusal — but the same refusal an absent parent earns, not a 500."""
        write_unparseable(client, INDICES, "ancestor")

        response = client.put("/indices/child",
                              json=derived_document("child", "ancestor"),
                              headers=HEADERS)

        assert response.status_code == 404, response.text
        assert "ancestor" in response.json()["error"]["message"]


class TestTheDeleteCascadeSurvivesABadFile:
    """A delete reads every index definition to find what derives from the one
    being removed, and did that through the strict reader — so one unparseable
    file in the collection 500d a delete of an unrelated index. The docstring
    on that helper already promised otherwise.
    """

    def test_an_index_can_be_deleted_beside_an_unparseable_one(self,
                                                               client):
        store = client.app.state.index_store
        store.write("keeper", index_document("keeper"))
        (store.directory / "wreckage.json").write_text(
            json.dumps({"id": "wreckage"})[:-3], encoding="utf-8")

        response = client.delete("/indices/keeper", headers=HEADERS)

        assert response.status_code == 200, response.text
        assert store.read("keeper") is None
