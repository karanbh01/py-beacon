# tests/test_index_schedule.py
"""BN-121: rebalance schedules, trading calendars and index metadata.

The load-bearing test in this file is `TestUnchangedDefault`. Everything else
adds behaviour; that one proves the addition took nothing away. An index that
names no calendar and no day rule must produce exactly the dates it always
did — otherwise every stored backtest silently redates, and the symptom is
numbers that moved with no change anyone made.

The hand-computed dates below are chosen for the cases where a naive
implementation is wrong: the third Friday of April 2025 is Good Friday, and the
1st of January is not a session on any US exchange.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.index import schedule
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.server import ServerConfig, create_app
from beacon.server.routers.indices import build_schedule
from beacon.server.schemas import IndexDocument

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
        "universe_identifiers": ["AAA"],
    }
    settings.update(overrides)

    return IndexDefinition(**settings)


def document(**overrides) -> dict:
    """A stored index document."""
    payload = {
        "id": "IDX", "name": "Test", "base_date": "2020-01-01",
        "base_value": 1000.0, "currency": "USD",
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


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                              storage_root=tmp_path)))


class TestUnchangedDefault:
    """The addition must take nothing away."""

    @pytest.mark.parametrize("frequency", schedule.FREQUENCIES)
    @pytest.mark.parametrize(("start", "end"), [
        ("2019-01-01", "2024-06-28"),
        ("2020-02-14", "2025-12-31"),
        ("2021-06-30", "2026-08-03"),
        ("2022-12-31", "2025-12-31"),
        ("2023-03-15", "2024-06-28"),
    ])
    def test_it_reproduces_the_previous_algorithm(self, frequency, start, end):
        assert (schedule.rebalance_dates(frequency, start, end)
                == previous_algorithm(frequency, start, end))

    def test_the_definition_default_is_unchanged(self):
        """Through the class every stored index is built from, not just the
        module underneath it."""
        index = definition()

        assert (index.get_rebalance_dates("2021-06-30", "2026-08-03")
                == previous_algorithm("QUARTERLY", "2021-06-30", "2026-08-03"))

    def test_the_default_day_rule_is_the_old_behaviour(self):
        assert schedule.DEFAULT_DAY_RULE == schedule.FIRST_BUSINESS_DAY

    def test_no_calendar_means_business_days(self):
        """Including holidays, which is what it always did — the calendar is
        opt-in, so nothing changes for an index that names none."""
        dates = schedule.rebalance_dates("ANNUAL", "2025-01-01", "2025-12-31")

        assert dates == [pd.Timestamp("2025-01-01")]

    def test_an_empty_range_is_empty(self):
        assert schedule.rebalance_dates("MONTHLY", "2025-06-01", "2025-05-01") == []


class TestDayRules:
    """Which day of a scheduled month."""

    def test_first_business_day(self):
        dates = schedule.rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31",
                                         schedule.FIRST_BUSINESS_DAY)

        assert [str(date.date()) for date in dates] == [
            "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]

    def test_last_business_day(self):
        dates = schedule.rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31",
                                         schedule.LAST_BUSINESS_DAY)

        assert [str(date.date()) for date in dates] == [
            "2025-01-31", "2025-04-30", "2025-07-31", "2025-10-31"]

    def test_third_friday(self):
        """Hand-computed: January 2025's Fridays are the 3rd, 10th, 17th,
        24th and 31st."""
        dates = schedule.rebalance_dates("QUARTERLY", "2025-01-01", "2025-12-31",
                                         schedule.THIRD_FRIDAY)

        assert [str(date.date()) for date in dates] == [
            "2025-01-17", "2025-04-18", "2025-07-18", "2025-10-17"]

    def test_an_unknown_day_rule_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported day rule"):
            schedule.rebalance_dates("MONTHLY", "2025-01-01", "2025-12-31",
                                     "SECOND_TUESDAY")

    def test_an_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported rebalancing frequency"):
            schedule.rebalance_dates("FORTNIGHTLY", "2025-01-01", "2025-12-31")

    def test_the_cadence_is_anchored_on_the_first_date(self):
        """A quarterly index starting in February rebalances in February, May,
        August and November — not on calendar quarters. That was the previous
        behaviour and changing it would move every existing index."""
        dates = schedule.rebalance_dates("QUARTERLY", "2025-02-01", "2025-12-31")

        assert [date.month for date in dates] == [2, 5, 8, 11]


class TestTradingCalendar:
    """Real holidays, via exchange_calendars."""

    def test_the_first_session_of_january_is_not_new_years_day(self):
        """1 January is a weekday in 2025 and not a session anywhere."""
        dates = schedule.rebalance_dates("ANNUAL", "2025-01-01", "2025-12-31",
                                         schedule.FIRST_BUSINESS_DAY, "XNYS")

        assert [str(date.date()) for date in dates] == ["2025-01-02"]

    def test_a_third_friday_on_good_friday_rolls_back(self):
        """April 2025's third Friday is the 18th, which is Good Friday. It must
        become Thursday the 17th, not the 25th — the 25th is the fourth Friday
        of the month and a week late, which is what counting *open* Fridays
        gives you."""
        dates = schedule.rebalance_dates("MONTHLY", "2025-04-01", "2025-04-30",
                                         schedule.THIRD_FRIDAY, "XNYS")

        assert [str(date.date()) for date in dates] == ["2025-04-17"]

    def test_the_same_date_without_a_calendar_stays_on_good_friday(self):
        """Which is the difference the calendar buys, stated as a test."""
        dates = schedule.rebalance_dates("MONTHLY", "2025-04-01", "2025-04-30",
                                         schedule.THIRD_FRIDAY)

        assert [str(date.date()) for date in dates] == ["2025-04-18"]

    def test_christmas_is_not_a_session(self):
        sessions = schedule.sessions("2025-12-20", "2025-12-31", "XNYS")

        assert pd.Timestamp("2025-12-25") not in sessions
        assert pd.Timestamp("2025-12-26") in sessions

    def test_christmas_is_a_business_day_without_one(self):
        sessions = schedule.sessions("2025-12-20", "2025-12-31")

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

    def test_a_declared_calendar_without_the_extra_is_an_error(self):
        """Not a silent fall back to Monday-to-Friday. Two installations would
        otherwise compute different indices from the same definition, and
        nothing would say which was which — so the failure has to be loud and
        name the extra to install.

        Run in a subprocess because the package has to be unimportable before
        the first import, which cannot be arranged in this one.
        """
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'exchange_calendars':\n"
            "            raise ImportError('blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from beacon.exceptions import MissingDependencyError\n"
            "from beacon.index import schedule\n"
            "# No calendar: unaffected, still business days.\n"
            "assert len(schedule.sessions('2025-01-01', '2025-01-31')) == 23\n"
            "try:\n"
            "    schedule.sessions('2025-01-01', '2025-01-31', 'XNYS')\n"
            "except MissingDependencyError as exc:\n"
            "    assert 'calendars' in str(exc), str(exc)\n"
            "    print('ok')\n"
            "else:\n"
            "    raise SystemExit('fell back silently instead of raising')\n")

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout


class TestNextRebalance:
    """Hand-computed across month, quarter and holiday boundaries."""

    def test_quarterly_from_a_january_base(self):
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-08-03")

        assert str(found.date()) == "2026-10-01"

    def test_it_crosses_a_quarter_boundary(self):
        """Asked the day before a rebalance, the answer is that rebalance."""
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-09-30")

        assert str(found.date()) == "2026-10-01"

    def test_the_rebalance_date_itself_is_not_next(self):
        """Strictly after: on the day, the next one is the following period."""
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-10-01")

        assert str(found.date()) == "2027-01-01"

    def test_third_friday_on_a_calendar(self):
        """October 2026's Fridays are the 2nd, 9th, 16th, 23rd and 30th."""
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2026-08-03",
                                        schedule.THIRD_FRIDAY, "XNYS")

        assert str(found.date()) == "2026-10-16"

    def test_a_holiday_shifts_the_answer(self):
        """Monthly third-Friday, asked from March 2025: April's is Good Friday,
        so the answer is the 17th rather than the 18th."""
        found = schedule.next_rebalance("MONTHLY", "2020-01-01", "2025-03-25",
                                        schedule.THIRD_FRIDAY, "XNYS")

        assert str(found.date()) == "2025-04-17"

    def test_asking_before_the_base_date_gives_the_first(self):
        found = schedule.next_rebalance("QUARTERLY", "2020-01-01", "2019-06-01")

        assert str(found.date()) == "2020-01-01"

    def test_the_definition_exposes_it(self):
        assert str(definition().next_rebalance("2026-08-03").date()) == "2026-10-01"


class TestDefinitionValidation:
    """The library refuses a schedule it cannot honour."""

    def test_an_unknown_day_rule_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="Unsupported day rule"):
            definition(rebalance_day_rule="SECOND_TUESDAY")

    def test_a_day_rule_reaches_the_dates(self):
        index = definition(rebalance_day_rule=schedule.THIRD_FRIDAY)
        dates = index.get_rebalance_dates("2025-01-01", "2025-12-31")

        assert [str(date.date()) for date in dates] == [
            "2025-01-17", "2025-04-18", "2025-07-18", "2025-10-17"]

    def test_a_calendar_reaches_the_dates(self):
        index = definition(calendar="XNYS")
        dates = index.get_rebalance_dates("2025-01-01", "2025-12-31")

        assert str(dates[0].date()) == "2025-01-02"


class TestDocumentMetadata:
    """The five Figma fields, as stored."""

    def test_the_defaults_are_the_previous_behaviour(self):
        parsed = IndexDocument.model_validate(document())

        assert parsed.return_type == "PRICE"
        assert parsed.calendar is None
        assert parsed.rebalance_day_rule == "FIRST_BUSINESS_DAY"
        assert parsed.publication_time is None
        assert parsed.effective_lag_sessions == 0

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
        """Every field is defaulted, so no stored document needs migrating."""
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

    def test_an_effective_lag_warns_that_it_is_not_yet_applied(self, client):
        """A warning, not an error: the field is stored deliberately. Silence
        would let a user believe the lag is in force."""
        response = client.post("/indices/validate",
                               json=document(effective_lag_sessions=3),
                               headers=auth())
        body = response.json()

        assert body["valid"] is True, "a lag must not block saving"
        assert any(finding["code"] == "EFFECTIVE_LAG_NOT_APPLIED"
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
