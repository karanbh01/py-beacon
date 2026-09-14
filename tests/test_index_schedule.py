# tests/test_index_schedule.py
"""BN-121/BN-180: rebalance schedules, trading calendars and index metadata.

BN-121's load-bearing class was `TestUnchangedDefault`, which pinned the
promise that an index naming no calendar produced exactly the dates it always
had. BN-180 deliberately breaks that promise, so those three tests are gone and
`TestTheCalendarIsRequired` stands in their place. The old promise was worth
keeping right up until somebody measured what it preserved: a calendar-less
index schedules 1 January, 4 July and 25 December, days no exchange has a
session for, so the dates it "always had" were partly dates that do not exist.

`previous_algorithm` survives, no longer as proof that nothing moved but as the
other side of a diff: `TestTheCalendarIsRequired` names, date by date, what the
migration changes for a quarterly index.

`TestSessionGaps` is the fixture the issue's comment asked for. Every other
test in this repository runs against data generated over `pd.bdate_range`, so
the schedule and the data share the Monday-to-Friday assumption and agree with
each other whether or not either is right. The frame there genuinely omits
1 January, 4 July and 25 December, which is the only way this class of defect
is visible at all.

The hand-computed dates below are chosen for the cases where a naive
implementation is wrong: the third Friday of April 2025 is Good Friday, and the
1st of January is not a session on any US exchange.
"""
import inspect
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index import schedule
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.server import ServerConfig, create_app
from beacon.server.routers.indices import build_schedule
from beacon.server.schemas import IndexDocument
from beacon.server.store import SCHEMA_VERSION_KEY, DocumentStore

TOKEN = "test-token-value"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def previous_algorithm(frequency: str,
                       start: str,
                       end: str) -> list[pd.Timestamp]:
    """`get_rebalance_dates` exactly as it stood before BN-121.

    Kept verbatim rather than imported, because the point is to compare the new
    implementation against the old *code*, not against the new code's idea of
    what the old code did.
    """
    months = {"MONTHLY": 1, "QUARTERLY": 3, "SEMI-ANNUAL": 6, "ANNUAL": 12}
    interval = months[frequency]
    first, last = pd.Timestamp(start), pd.Timestamp(end)

    starts = pd.date_range(start=first - pd.offsets.MonthBegin(1),
                           end=last + pd.offsets.MonthEnd(1), freq="BMS")
    candidates = [date for date in starts if first <= date <= last]
    if not candidates:
        return []

    kept = [candidates[0]]
    for date in candidates[1:]:
        elapsed = (date.year - kept[-1].year) * 12 + (date.month - kept[-1].month)
        if elapsed >= interval:
            kept.append(date)

    return kept


def definition(**overrides) -> IndexDefinition:
    """An index definition with the scheduling fields overridable."""
    settings = {
        "index_id": "IDX", "index_name": "Test", "base_date": "2020-01-01",
        "base_value": 1000.0, "currency": "USD", "eligibility_rules": [],
        "weighting_scheme": EqualWeighted(), "rebalancing_frequency": "QUARTERLY",
        "calendar": schedule.DEFAULT_CALENDAR,
        "universe_identifiers": ["AAA"],
    }
    settings.update(overrides)

    return IndexDefinition(**settings)


def document(**overrides) -> dict:
    """A stored index document, valid on the current wire."""
    payload = {
        "id": "IDX", "name": "Test", "base_date": "2020-01-01",
        "base_value": 1000.0, "currency": "USD", "calendar": "XNYS",
        "rebalancing_frequency": "QUARTERLY", "description": None,
        "universe": {"universe_id": None, "identifiers": ["AAA"]},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "w", "scheme": "EqualWeighted",
                          "params": {}, "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }
    payload.update(overrides)

    return payload


def legacy_document(**overrides) -> dict:
    """A document as stored before BN-180: no `calendar` key at all.

    Written by hand rather than by round-tripping the current model, because
    the current model cannot produce one — which is the whole point of the
    migration.
    """
    payload = document(**overrides)
    payload.pop("calendar", None)

    return payload


def store_raw(tmp_path,
              payload: dict,
              version: int) -> DocumentStore:
    """Put a document on disk at a given schema version, bypassing the model."""
    store = DocumentStore("indices", root=tmp_path)
    (store.directory / f"{payload['id']}.json").write_text(
        json.dumps({**payload, SCHEMA_VERSION_KEY: version}), encoding="utf-8")

    return store


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                              storage_root=tmp_path)))


class TestTheCalendarIsRequired:
    """BN-180. Replaces BN-121's `TestUnchangedDefault`.

    Those three tests pinned the opposite promise — that an index naming no
    calendar kept every date it had — and they were true and passing when they
    were deleted. What changed is not the code's correctness but the owner's
    judgement about which of two incompatible properties to keep: dates that
    never move, or dates the exchange is open on. The migration is what makes
    the second one affordable.
    """

    def test_a_definition_cannot_be_built_without_one(self):
        """Required, with no default — the argument is simply absent.

        A default was tried while this issue was being written and rejected:
        `IndexDefinition(currency="EUR")` would have silently scheduled a
        European index on New York's holidays, coherently, with nothing to say
        so. Choosing a calendar for an index that already exists is repair,
        which is what the migration does; choosing one for an index being
        created is a guess only the caller can make.
        """
        settings = {"index_id": "IDX", "index_name": "Test",
                    "base_date": "2020-01-01", "base_value": 1000.0,
                    "currency": "USD", "eligibility_rules": [],
                    "weighting_scheme": EqualWeighted(),
                    "rebalancing_frequency": "QUARTERLY"}

        with pytest.raises(TypeError, match="calendar"):
            IndexDefinition(**settings)

        assert IndexDefinition(**settings, calendar="XNYS").calendar == "XNYS"

    def test_an_empty_calendar_is_refused(self):
        """Present but empty is the shape a caller reaches for when they want
        the old "no calendar" behaviour back, so it fails too."""
        with pytest.raises(ValueError, match="calendar cannot be empty"):
            definition(calendar="")

    def test_the_default_calendar_is_the_migration_s_target_only(self):
        """`DEFAULT_CALENDAR` survives the removal of the constructor default,
        because the migration still needs a value to write. What it no longer
        is, is something a newly created index can pick up by omission."""
        assert schedule.DEFAULT_CALENDAR == "XNYS"

        parameter = inspect.signature(IndexDefinition).parameters["calendar"]

        assert parameter.default is inspect.Parameter.empty

    def test_an_index_on_xnys_does_not_rebalance_on_a_holiday(self):
        """The defect, stated directly. A monthly first-business-day index
        rebalances on 1 January and an annual one on 1 January, neither of
        which is a session; on XNYS both become the 2nd."""
        monthly = definition(rebalancing_frequency="MONTHLY")
        dates = monthly.get_rebalance_dates("2025-01-01", "2025-12-31")

        assert pd.Timestamp("2025-01-01") not in dates
        assert dates[0] == pd.Timestamp("2025-01-02")

        # And no date it names is a holiday, for the whole year.
        holidays = {pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-04"),
                    pd.Timestamp("2025-12-25")}

        assert not holidays & set(dates)

    @pytest.mark.parametrize("frequency", schedule.FREQUENCIES)
    @pytest.mark.parametrize(("start", "end"), [
        ("2019-01-01", "2024-06-28"),
        ("2020-02-14", "2025-12-31"),
        ("2021-06-30", "2026-08-03"),
        ("2022-12-31", "2025-12-31"),
        ("2023-03-15", "2024-06-28"),
    ])
    def test_the_new_dates_are_the_old_ones_or_later_in_the_same_month(self,
                                                                      frequency,
                                                                      start,
                                                                      end):
        """The shape of the redating, over the same grid BN-121 pinned.

        Each date either stays put or rolls forward to the first session of the
        same month — it never moves to another month, never disappears, and the
        count never changes. That is the whole scope of what the migration does
        to a stored index, asserted rather than asserted-about.
        """
        old = previous_algorithm(frequency, start, end)
        new = schedule.rebalance_dates(frequency, start, end,
                                       schedule.DEFAULT_CALENDAR)

        assert len(new) == len(old)
        for before, after in zip(old, new, strict=True):
            assert after >= before
            assert (after.year, after.month) == (before.year, before.month)

    def test_the_redating_is_named_date_by_date(self):
        """Exactly which dates a quarterly index moves over 2020-2026.

        Every one is a January rebalance and every one is New Year's Day in one
        form or another — the whole redating for this index, in six dates out
        of twenty-eight. 2022 does not appear because 1 January 2022 was a
        Saturday, so the old algorithm had already moved it to Monday the 3rd,
        which is a session. 2023 does appear and is the interesting one: 1
        January was a Sunday, the old algorithm moved to Monday the 2nd, and
        NYSE *observed* the holiday on that Monday — a case no "skip the
        weekend" rule can reach.

        2021-01-01 moves to the 4th rather than the 2nd because the 2nd is a
        Saturday.
        """
        old = previous_algorithm("QUARTERLY", "2020-01-01", "2026-12-31")
        new = schedule.rebalance_dates("QUARTERLY", "2020-01-01", "2026-12-31",
                                       schedule.DEFAULT_CALENDAR)

        moved = {str(before.date()): str(after.date())
                 for before, after in zip(old, new, strict=True)
                 if before != after}

        assert moved == {"2020-01-01": "2020-01-02",
                         "2021-01-01": "2021-01-04",
                         "2023-01-02": "2023-01-03",
                         "2024-01-01": "2024-01-02",
                         "2025-01-01": "2025-01-02",
                         "2026-01-01": "2026-01-02"}

    def test_the_default_day_rule_is_the_old_behaviour(self):
        """The day rule genuinely did not change; only the calendar did."""
        assert schedule.DEFAULT_DAY_RULE == schedule.FIRST_BUSINESS_DAY

    def test_business_days_are_still_reachable_but_only_on_request(self):
        """The `None` branch survives for a library caller doing plain
        business-day arithmetic. What it no longer is, is the default: the
        argument is required, so Monday-to-Friday cannot be had by omission,
        and nothing a user stores can reach it."""
        assert (schedule.rebalance_dates("ANNUAL", "2025-01-01", "2025-12-31",
                                         None)
                == [pd.Timestamp("2025-01-01")])

        with pytest.raises(TypeError):
            schedule.rebalance_dates("ANNUAL", "2025-01-01", "2025-12-31")

    def test_an_empty_range_is_empty(self):
        assert schedule.rebalance_dates("MONTHLY", "2025-06-01", "2025-05-01",
                                        "XNYS") == []


class TestDayRules:
    """Which day of a scheduled month. Business days, asked for explicitly, so
    the day rule is the only thing varying."""

    def test_first_business_day(self):
        dates = schedule.rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31",
                                         None, schedule.FIRST_BUSINESS_DAY)

        assert [str(date.date()) for date in dates] == [
            "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]

    def test_last_business_day(self):
        dates = schedule.rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31",
                                         None, schedule.LAST_BUSINESS_DAY)

        assert [str(date.date()) for date in dates] == [
            "2025-01-31", "2025-04-30", "2025-07-31", "2025-10-31"]

    def test_third_friday(self):
        """Hand-computed: January 2025's Fridays are the 3rd, 10th, 17th,
        24th and 31st."""
        dates = schedule.rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31",
                                         None, schedule.THIRD_FRIDAY)

        assert [str(date.date()) for date in dates] == [
            "2025-01-17", "2025-04-18", "2025-07-18", "2025-10-17"]

    def test_an_unknown_day_rule_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported day rule"):
            schedule.rebalance_dates("MONTHLY", "2025-01-01", "2025-12-31",
                                     "XNYS", "SECOND_TUESDAY")

    def test_an_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported rebalancing frequency"):
            schedule.rebalance_dates("FORTNIGHTLY", "2025-01-01", "2025-12-31",
                                     "XNYS")

    def test_the_cadence_is_anchored_on_the_first_date(self):
        """A quarterly index starting in February rebalances in February, May,
        August and November — not on calendar quarters. That was the previous
        behaviour and changing it would move every existing index."""
        dates = schedule.rebalance_dates("QUARTERLY", "2025-02-01", "2025-12-31",
                                         "XNYS")

        assert [date.month for date in dates] == [2, 5, 8, 11]


class TestTradingCalendar:
    """Real holidays, via exchange_calendars."""

    def test_the_first_session_of_january_is_not_new_years_day(self):
        """1 January is a weekday in 2025 and not a session anywhere."""
        dates = schedule.rebalance_dates("ANNUAL", "2025-01-01", "2025-12-31",
                                         "XNYS", schedule.FIRST_BUSINESS_DAY)

        assert [str(date.date()) for date in dates] == ["2025-01-02"]

    def test_a_third_friday_on_good_friday_rolls_back(self):
        """April 2025's third Friday is the 18th, which is Good Friday. It must
        become Thursday the 17th, not the 25th — the 25th is the fourth Friday
        of the month and a week late, which is what counting *open* Fridays
        gives you."""
        dates = schedule.rebalance_dates("MONTHLY", "2025-04-01", "2025-04-30",
                                         "XNYS", schedule.THIRD_FRIDAY)

        assert [str(date.date()) for date in dates] == ["2025-04-17"]

    def test_the_same_date_without_a_calendar_stays_on_good_friday(self):
        """Which is the difference the calendar buys, stated as a test."""
        dates = schedule.rebalance_dates("MONTHLY", "2025-04-01", "2025-04-30",
                                         None, schedule.THIRD_FRIDAY)

        assert [str(date.date()) for date in dates] == ["2025-04-18"]

    def test_christmas_is_not_a_session(self):
        sessions = schedule.sessions("2025-12-20", "2025-12-31", "XNYS")

        assert pd.Timestamp("2025-12-25") not in sessions
        assert pd.Timestamp("2025-12-26") in sessions

    def test_christmas_is_a_business_day_without_one(self):
        sessions = schedule.sessions("2025-12-20", "2025-12-31", None)

        assert pd.Timestamp("2025-12-25") in sessions

    def test_a_non_us_calendar_works(self):
        """Boxing Day is a London holiday and a New York session."""
        london = schedule.sessions("2025-12-24", "2025-12-31", "XLON")

        assert pd.Timestamp("2025-12-26") not in london

    def test_a_known_calendar_is_recognised(self):
        assert schedule.is_known_calendar("XNYS")

    def test_an_unknown_calendar_is_not(self):
        assert not schedule.is_known_calendar("NOPE")

    def test_a_range_outside_the_calendar_is_empty_not_an_error(self):
        """Asking past the published holidays is a normal thing to do."""
        assert len(schedule.sessions("1700-01-01", "1700-12-31", "XNYS")) == 0

    def test_the_calendar_package_is_a_core_dependency(self):
        """BN-121 had the opposite test here: `exchange_calendars` was an
        extra, and a definition naming a calendar without it raised a
        `MissingDependencyError` rather than falling back. BN-180 makes the
        calendar required, and a required input cannot sit behind an optional
        install — so the package is core and the failure mode it guarded
        against no longer exists.

        Run in a subprocess with every *remaining* optional package blocked, so
        this says the schedule needs none of them, not merely that this
        interpreter happens to have them.
        """
        import subprocess
        import sys

        from beacon._optional import EXTRA_FOR_MODULE

        assert "exchange_calendars" not in EXTRA_FOR_MODULE

        script = (
            "import sys\n"
            f"BLOCKED = {sorted(EXTRA_FOR_MODULE)!r}\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError('blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from beacon.index import schedule\n"
            "dates = schedule.rebalance_dates('ANNUAL', '2025-01-01',\n"
            "                                 '2025-12-31', 'XNYS')\n"
            "assert [str(d.date()) for d in dates] == ['2025-01-02'], dates\n"
            "print('ok')\n")

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout


class TestNextRebalance:
    """Hand-computed across month, quarter and holiday boundaries."""

    def test_quarterly_from_a_january_base(self):
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-08-03",
                                        "XNYS")

        assert str(found.date()) == "2026-10-01"

    def test_it_crosses_a_quarter_boundary(self):
        """Asked the day before a rebalance, the answer is that rebalance."""
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-09-30",
                                        "XNYS")

        assert str(found.date()) == "2026-10-01"

    def test_the_rebalance_date_itself_is_not_next(self):
        """Strictly after: on the day, the next one is the following period.

        The answer moved with BN-180: on business days the following January
        rebalance was the 1st, and on XNYS it is the 2nd — New Year's Day is
        not a session.
        """
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-10-01",
                                        "XNYS")

        assert str(found.date()) == "2027-01-04"

    def test_third_friday_on_a_calendar(self):
        """October 2026's Fridays are the 2nd, 9th, 16th, 23rd and 30th."""
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-08-03",
                                        "XNYS", schedule.THIRD_FRIDAY)

        assert str(found.date()) == "2026-10-16"

    def test_a_holiday_shifts_the_answer(self):
        """Monthly third-Friday, asked from March 2025: April's is Good Friday,
        so the answer is the 17th rather than the 18th."""
        found = schedule.next_rebalance("MONTHLY", "2020-01-01", "2025-03-25",
                                        "XNYS", schedule.THIRD_FRIDAY)

        assert str(found.date()) == "2025-04-17"

    def test_asking_before_the_base_date_gives_the_first(self):
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2019-06-01",
                                        "XNYS")

        assert str(found.date()) == "2020-01-02"

    def test_the_definition_exposes_it(self):
        assert str(definition().next_rebalance("2026-08-03").date()) == "2026-10-01"


class TestDefinitionValidation:
    """The library refuses a schedule it cannot honour."""

    def test_an_unknown_day_rule_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="Unsupported day rule"):
            definition(rebalance_day_rule="SECOND_TUESDAY")

    def test_a_day_rule_reaches_the_dates(self):
        """April's third Friday is Good Friday, so on the default calendar it
        is the 17th. BN-121 read 2025-04-18 here, when the definition's default
        was no calendar at all."""
        index = definition(rebalance_day_rule=schedule.THIRD_FRIDAY)
        dates = index.get_rebalance_dates("2025-01-01", "2025-12-31")

        assert [str(date.date()) for date in dates] == [
            "2025-01-17", "2025-04-17", "2025-07-18", "2025-10-17"]

    def test_a_calendar_reaches_the_dates(self):
        index = definition(calendar="XNYS")
        dates = index.get_rebalance_dates("2025-01-01", "2025-12-31")

        assert str(dates[0].date()) == "2025-01-02"


class TestDocumentMetadata:
    """The five Figma fields, as stored."""

    def test_the_defaults_are_the_previous_behaviour(self):
        parsed = IndexDocument.model_validate(document())

        assert parsed.return_type == "PRICE"
        assert parsed.rebalance_day_rule == "FIRST_BUSINESS_DAY"
        assert parsed.publication_time is None
        assert parsed.effective_lag_sessions == 0

    def test_the_calendar_is_required_on_the_wire(self):
        """The one field of the five that is not defaulted. A client creating
        an index must send it; there is no null to fall back to."""
        with pytest.raises(ValueError, match="calendar"):
            IndexDocument.model_validate(legacy_document())

    def test_a_create_without_a_calendar_is_a_422(self, client):
        response = client.post("/indices", json=legacy_document(),
                               headers=auth())

        assert response.status_code == 422
        assert "calendar" in response.text

    def test_a_null_calendar_is_refused_too(self, client):
        """Explicitly null is the shape a client written against the old
        schema sends, and it has to fail as loudly as omitting the field."""
        response = client.post("/indices", json=document(calendar=None),
                               headers=auth())

        assert response.status_code == 422

    def test_they_round_trip(self, client):
        payload = document(calendar="XNYS", rebalance_day_rule="THIRD_FRIDAY",
                           publication_time="18:00 America/New_York",
                           effective_lag_sessions=2)

        created = client.post("/indices", json=payload, headers=auth())
        assert created.status_code == 200, created.text

        fetched = client.get("/indices/IDX", headers=auth()).json()

        assert fetched["calendar"] == "XNYS"
        assert fetched["rebalance_day_rule"] == "THIRD_FRIDAY"
        assert fetched["publication_time"] == "18:00 America/New_York"
        assert fetched["effective_lag_sessions"] == 2

    def test_a_document_written_before_these_fields_still_loads(self):
        """Every field except the calendar is defaulted; that one is migrated
        rather than defaulted, which `TestTheStoredMigration` covers."""
        parsed = IndexDocument.model_validate(document())

        assert parsed.id == "IDX"

    def test_total_return_is_accepted_since_bn_125(self, client):
        """BN-121 restricted this to PRICE because the calculator had no
        dividend reinvestment; BN-125 gave it one, so the enum widened."""
        response = client.post("/indices",
                               json=document(return_type="TOTAL_RETURN"),
                               headers=auth())

        assert response.status_code == 200, response.text

    def test_an_unknown_return_type_is_still_refused(self, client):
        response = client.post("/indices",
                               json=document(return_type="GROSS_OF_FEES"),
                               headers=auth())

        assert response.status_code == 422

    def test_a_negative_lag_is_refused(self, client):
        response = client.post("/indices",
                               json=document(effective_lag_sessions=-1),
                               headers=auth())

        assert response.status_code == 422


class TestScheduleValidation:
    """Findings for combinations that cannot be honoured."""

    def test_an_unknown_day_rule_is_an_error(self, client):
        response = client.post("/indices/validate",
                               json=document(rebalance_day_rule="SECOND_TUESDAY"),
                               headers=auth())
        body = response.json()

        assert body["valid"] is False
        assert any(finding["code"] == "UNKNOWN_DAY_RULE"
                   for finding in body["findings"])

    def test_an_unknown_calendar_is_an_error_not_a_fallback(self, client):
        """Falling back to business days would make the index compute
        differently from the one that was defined, with nothing to say so."""
        response = client.post("/indices/validate",
                               json=document(calendar="XNOPE"), headers=auth())
        body = response.json()

        assert body["valid"] is False
        assert any(finding["code"] == "UNKNOWN_CALENDAR"
                   for finding in body["findings"])

    def test_a_known_calendar_is_accepted(self, client):
        response = client.post("/indices/validate",
                               json=document(calendar="XNYS"), headers=auth())

        assert response.json()["valid"] is True

    def test_an_effective_lag_is_accepted_without_comment(self, client):
        """BN-121 warned that the lag was recorded but not applied. BN-126
        applies it, so the warning is gone rather than left as a stale caution
        about behaviour that now exists."""
        response = client.post("/indices/validate",
                               json=document(effective_lag_sessions=3),
                               headers=auth())
        body = response.json()

        assert body["valid"] is True
        assert not any(finding["code"] == "EFFECTIVE_LAG_NOT_APPLIED"
                       for finding in body["findings"])


class TestScheduleEndpoint:
    """`GET /indices/{id}/schedule`."""

    def _saved(self, client, **overrides):
        created = client.post("/indices", json=document(**overrides),
                              headers=auth())
        assert created.status_code == 200, created.text

        return client

    def test_it_reports_the_next_rebalance_and_days_until(self, client):
        self._saved(client)

        body = client.get("/indices/IDX/schedule", params={"asof": "2026-08-03"},
                          headers=auth()).json()

        assert body["next_rebalance"] == "2026-10-01"
        assert body["days_until"] == 59

    def test_days_until_are_calendar_days(self, client):
        """It renders as "in 57 days" and a reader counts those on a wall
        calendar, not in sessions."""
        self._saved(client)

        body = client.get("/indices/IDX/schedule", params={"asof": "2026-09-30"},
                          headers=auth()).json()

        assert body["days_until"] == 1

    def test_it_honours_the_day_rule_and_calendar(self, client):
        self._saved(client, rebalance_day_rule="THIRD_FRIDAY", calendar="XNYS")

        body = client.get("/indices/IDX/schedule", params={"asof": "2026-08-03"},
                          headers=auth()).json()

        assert body["next_rebalance"] == "2026-10-16"
        assert body["calendar"] == "XNYS"

    def test_it_lists_dates_either_side(self, client):
        self._saved(client)

        body = client.get("/indices/IDX/schedule", params={"asof": "2026-08-03"},
                          headers=auth()).json()

        assert body["recent"], "no history shown"
        assert body["upcoming"][0] == body["next_rebalance"]
        assert all(date <= body["as_of"] for date in body["recent"])
        assert all(date > body["as_of"] for date in body["upcoming"])

    def test_an_unknown_index_is_a_404(self, client):
        assert client.get("/indices/nope/schedule",
                          headers=auth()).status_code == 404

    def test_it_requires_authentication(self, client):
        self._saved(client)

        assert client.get("/indices/IDX/schedule").status_code == 401

    def test_it_defaults_to_today(self, client):
        self._saved(client)

        body = client.get("/indices/IDX/schedule", headers=auth()).json()

        assert body["as_of"] == str(pd.Timestamp.today().normalize().date())

    def test_it_is_derived_rather_than_stored(self, client):
        """Two different as-of dates give two different answers off one stored
        document, which a stored next-rebalance could not do."""
        self._saved(client)

        first = build_schedule(IndexDocument.model_validate(document()),
                               "2026-08-03")
        later = build_schedule(IndexDocument.model_validate(document()),
                               "2026-11-03")

        assert first.next_rebalance != later.next_rebalance


class TestTheStoredMigration:
    """Schema version 2: a stored document without a calendar gets XNYS.

    The document store has carried a version and an empty migration chain since
    it was written, on the stated reasoning that the first change should not
    have to invent the machinery. This is that first change, so these tests are
    also the first exercise of the chain against a real migration rather than a
    monkeypatched one.
    """

    def test_a_document_without_a_calendar_migrates_to_xnys(self,
                                                            tmp_path):
        store = store_raw(tmp_path, legacy_document(), version=1)

        migrated = store.read("IDX")

        assert migrated["calendar"] == schedule.DEFAULT_CALENDAR
        assert migrated[SCHEMA_VERSION_KEY] == 2

    def test_the_migrated_document_is_valid_and_schedules_on_sessions(self,
                                                                      tmp_path):
        """End to end: the thing that could not be parsed before the migration
        parses after it, and the dates it then produces are sessions."""
        store = store_raw(tmp_path, legacy_document(), version=1)

        parsed = IndexDocument.model_validate(store.read("IDX"))
        dates = build_schedule(parsed, "2025-01-15").recent

        assert parsed.calendar == "XNYS"
        assert "2025-01-01" not in dates
        assert "2025-01-02" in dates

    def test_a_stored_calendar_is_left_alone(self,
                                             tmp_path):
        """The migration backfills; it does not overwrite. An index deliberately
        on London must not be moved to New York by an upgrade."""
        store = store_raw(tmp_path, document(calendar="XLON"), version=1)

        assert store.read("IDX")["calendar"] == "XLON"

    def test_documents_from_other_collections_are_untouched(self,
                                                            tmp_path):
        """One version chain covers every collection. A watchlist has no
        schedule, so it must come back with a version stamp and nothing else
        added — a stray `calendar` key on a watchlist would be a migration
        writing outside its own subject.
        """
        store = DocumentStore("watchlists", root=tmp_path)
        (store.directory / "w.json").write_text(
            json.dumps({"id": "w", "identifiers": ["AAA"],
                        SCHEMA_VERSION_KEY: 1}), encoding="utf-8")

        migrated = store.read("w")

        assert "calendar" not in migrated
        assert migrated[SCHEMA_VERSION_KEY] == 2

    def test_a_newer_version_is_still_refused(self,
                                              tmp_path):
        """The guard that stops a v3 document being read by a v2 build. It
        behaved correctly against an empty chain; the point of checking it
        again is that the chain is no longer empty.
        """
        store = store_raw(tmp_path, document(), version=3)

        with pytest.raises(Exception, match="newer than this"):
            store.read("IDX")

    def test_the_endpoint_serves_a_migrated_document(self,
                                                     tmp_path):
        """Through the API, because that is where a user meets it: an index
        stored before BN-180 is still listed and still readable."""
        store_raw(tmp_path, legacy_document(), version=1)
        served = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                    storage_root=tmp_path)))

        fetched = served.get("/indices/IDX", headers=auth())

        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["calendar"] == "XNYS"


# ---------------------------------------------------------------------------
# Data that genuinely observes holidays.
#
# Every other fixture in this repository is generated over `pd.bdate_range`,
# which is the same Monday-to-Friday assumption the schedule used to make. Two
# components sharing one assumption agree with each other whether or not the
# assumption is right, so no test built on the generator can see this defect.
# These three dates are the cheapest data that can: all three are weekdays in
# 2025, and none is a session on any US exchange.
# ---------------------------------------------------------------------------
GAP_HOLIDAYS = ("2025-01-01", "2025-07-04", "2025-12-25")


def gap_dates() -> pd.DatetimeIndex:
    """2025's weekdays, less the three holidays the frame omits."""
    weekdays = pd.bdate_range("2025-01-01", "2025-12-31")

    return weekdays.drop([pd.Timestamp(date) for date in GAP_HOLIDAYS])


def gap_fetcher() -> DataFetcher:
    """A hand-built panel with genuine session gaps."""
    dates = gap_dates()
    market = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE": date, "CLOSE": 100.0 + index,
         "SHARES_OUTSTANDING": 1_000_000.0}
        for index, date in enumerate(dates)
    ])
    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01", "NAME": "AAA",
         "CURRENCY": "USD", "EXCHANGE": "NYSE"}])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


class TestSessionGaps:
    """The fixture the issue's comment asked for, and what it proves."""

    def test_the_fixture_really_has_gaps(self):
        """Asserted rather than assumed: a fixture that quietly stopped having
        gaps would make every test below pass for the wrong reason."""
        available = set(gap_fetcher().fetch_market_data(
            "AAA", "2025-01-01", "2025-12-31").index)

        for holiday in GAP_HOLIDAYS:
            assert pd.Timestamp(holiday).dayofweek < 5, "not a weekday"
            assert pd.Timestamp(holiday) not in available

    def test_a_calendar_less_schedule_names_days_the_data_lacks(self):
        """The defect, reproduced. This is the old default, and every one of
        these dates is a rebalance the calculator would price against nothing.
        """
        available = set(gap_fetcher().fetch_market_data(
            "AAA", "2025-01-01", "2025-12-31").index)
        business_days = schedule.sessions("2025-01-01", "2025-12-31", None)

        missing = sorted(set(business_days) - available)

        assert [str(date.date()) for date in missing] == list(GAP_HOLIDAYS)

    def test_the_migrated_index_names_only_days_the_data_has(self):
        """The fix, over the same data. Every rebalance date an index on XNYS
        produces is a date the frame has a bar for — for every cadence and
        every day rule, not just the one that happened to be tested."""
        available = set(gap_fetcher().fetch_market_data(
            "AAA", "2025-01-01", "2025-12-31").index)

        for frequency in schedule.FREQUENCIES:
            for day_rule in schedule.DAY_RULES:
                index = definition(rebalancing_frequency=frequency,
                                   rebalance_day_rule=day_rule)
                dates = index.get_rebalance_dates("2025-01-01", "2025-12-31")

                assert dates, f"{frequency}/{day_rule} scheduled nothing"
                assert not set(dates) - available, (
                    f"{frequency}/{day_rule} scheduled a date the data has no "
                    f"session for: {sorted(set(dates) - available)}")

    def test_the_same_index_before_the_migration_did_not(self):
        """The contrast, so the test above cannot pass vacuously. A monthly
        first-business-day index on business days lands on 1 January; on XNYS
        it lands on the 2nd, which the data has."""
        available = set(gap_fetcher().fetch_market_data(
            "AAA", "2025-01-01", "2025-12-31").index)

        before = schedule.rebalance_dates("MONTHLY", "2025-01-01", "2025-12-31",
                                          None)
        after = schedule.rebalance_dates("MONTHLY", "2025-01-01", "2025-12-31",
                                         schedule.DEFAULT_CALENDAR)

        assert sorted(set(before) - available) == [pd.Timestamp("2025-01-01")]
        assert not set(after) - available
        assert str(after[0].date()) == "2025-01-02"


class TestPublishedCalendars:
    """`GET /indices/calendars`.

    A required field whose accepted values are not published forces every
    client to hard-code a hundred MICs or ship free text that now 422s. This is
    the same answer `/indices/rule-types` and `/optimise/constraint-types`
    give for their own closed sets, and it is read from the library at request
    time rather than copied.
    """

    def test_it_lists_the_calendars_the_engine_accepts(self, client):
        body = client.get("/indices/calendars", headers=auth()).json()
        codes = [row["code"] for row in body["calendars"]]

        assert "XNYS" in codes
        assert "XLON" in codes
        assert body["default"] == schedule.DEFAULT_CALENDAR

    def test_every_published_value_is_actually_accepted(self, client):
        """The property that makes publishing worth anything: a client that
        picks any listed value must not then be refused by validation."""
        listed = client.get("/indices/calendars", headers=auth()).json()

        assert listed["calendars"], "nothing published"
        assert all(schedule.is_known_calendar(row["code"])
                   for row in listed["calendars"])

    def test_it_is_sourced_from_the_package_not_a_copy(self, client):
        body = client.get("/indices/calendars", headers=auth()).json()

        assert ([row["code"] for row in body["calendars"]]
                == schedule.known_calendars())

    def test_the_region_is_the_timezone_s_own_first_segment(self, client):
        """Derived, and sent as the tz database spells it — a noun, not an
        adjective. 'European' would need a mapping, and a mapping would have to
        guess at Atlantic, Pacific and UTC, which have no distinct adjective at
        all. How a dropdown words its headings is the client's business.
        """
        rows = {row["code"]: row
                for row in client.get("/indices/calendars",
                                      headers=auth()).json()["calendars"]}

        assert rows["XNYS"]["region"] == "America"
        assert rows["XNYS"]["tz"] == "America/New_York"
        assert rows["XLON"]["region"] == "Europe"
        assert rows["XTKS"]["region"] == "Asia"

        for row in rows.values():
            assert row["region"] == row["tz"].split("/", maxsplit=1)[0]

    def test_every_calendar_resolves_to_a_region(self, client):
        """All 102, with no gaps and no fallback path: a row whose region
        failed to derive would reach a dropdown as a blank heading."""
        rows = client.get("/indices/calendars", headers=auth()).json()["calendars"]

        assert len(rows) == len(schedule.known_calendars())
        assert all(row["region"] and row["tz"] for row in rows)

    def test_the_two_utc_calendars_report_utc_as_their_region(self, client):
        """Not a region, and deliberately not special-cased. `24/5` and `24/7`
        are round-the-clock calendars with no venue and no continent; inventing
        one would be a hand-kept mapping, and 'Other' is a heading the client
        is better placed to choose than the server."""
        rows = {row["code"]: row
                for row in client.get("/indices/calendars",
                                      headers=auth()).json()["calendars"]}

        assert rows["24/7"]["region"] == "UTC"
        assert rows["24/5"]["region"] == "UTC"

    def test_a_curated_name_is_used_and_an_uncurated_one_falls_back(self, client):
        """The fallback is what makes a partial list safe: an unnamed calendar
        is labelled less well, never wrongly, and never excluded."""
        rows = {row["code"]: row
                for row in client.get("/indices/calendars",
                                      headers=auth()).json()["calendars"]}

        assert rows["XNYS"]["name"] == "New York Stock Exchange"
        assert rows["AIXK"]["name"] == "AIXK"
        assert all(row["name"] for row in rows.values())

    def test_every_curated_name_is_keyed_on_a_real_calendar(self):
        """A curated name for a code this installation does not have is dead
        weight, and one attached to the wrong code would mislabel a valid
        calendar — the single failure mode the fallback cannot cover. Two
        entries were removed by this test: XNSE and XBSP, neither of which
        `exchange_calendars` carries (India is XBOM, Brazil is BVMF).
        """
        known = set(schedule.known_calendars())

        assert set(schedule.DISPLAY_NAMES) <= known, (
            set(schedule.DISPLAY_NAMES) - known)

    def test_there_is_no_region_mapping_to_drift(self):
        """Stated as a test because the whole argument for deriving the region
        is that no table exists to fall out of date. A mapping that exists will
        be used."""
        source = inspect.getsource(schedule.calendar_region)

        assert "get_calendar" in source
        assert not any(word in source
                       for word in ("European", "American", "Asian"))

    def test_it_needs_no_data_source(self, client):
        """Like `/indices/rule-types`: it describes what the library can do,
        not what this server happens to hold, so it answers on a process
        started with nothing configured."""
        assert client.get("/indices/calendars",
                          headers=auth()).status_code == 200

    def test_it_is_not_swallowed_as_an_index_id(self, client):
        """`calendars` sits beside `/indices/{index_id}`, so it is reserved —
        otherwise `PUT /indices/calendars` would store a document there and the
        URL would mean different things by verb."""
        response = client.put("/indices/calendars", json=document(id="calendars"),
                              headers=auth())

        assert response.status_code == 422
