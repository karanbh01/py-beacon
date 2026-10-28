# tests/test_run_attribution.py
"""BN-393: a run's attribution by name and group, for the fund, a comparator
and the active return.

The run is test_risk_contributions_over_time's thirty names in three
sectors, the fund held at fixed weights and the index equally weighted, each
book's daily return exactly its weights at the close before times its names'
returns. So every attribution reconciles to the last digits, and the groups
add up to the names.
"""
import pytest

from beacon.analysis.attribution import Attribution, attribute
from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS
from test_factor_analytics import DAYS, EQUAL, FUND
from test_risk_contributions_over_time import noisy


def consistent() -> RunData:
    """noisy()'s run with an index whose levels are its weights' returns."""
    data = noisy()
    index = data.tables[NAME_RETURNS] @ EQUAL
    index.iloc[0] = 0.0

    return RunData(currency="USD", returns=data.returns, nav=data.nav, units=data.units,
                   initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                   tables=dict(data.tables),
                   comparators={"index": 100.0 * (1.0 + index).cumprod()})


@pytest.fixture(scope="module")
def attribution():
    return Attribution(consistent())


class TestByName:

    def test_it_is_attribute_on_the_run_s_tables(self,
                                                 attribution):
        run = attribution.run
        expected = attribute(run.returns, run.tables[FUND_WEIGHTS], run.tables[NAME_RETURNS])

        found = attribution.contribution()

        assert found.to_frame().equals(expected.to_frame())
        assert found.residual == expected.residual

    def test_a_book_made_of_its_names_reconciles(self,
                                                 attribution):
        for book in ("fund", "index"):
            found = attribution.contribution(book)
            assert found.reconciles(1e-12), book
            assert found.explained == pytest.approx(found.total_return)

    def test_over_a_window(self,
                           attribution):
        found = attribution.contribution(start=DAYS[50], end=DAYS[99])

        assert (found.start, found.periods) == (DAYS[51].isoformat(), 49)
        assert found.reconciles(1e-12)


class TestByGroup:

    def test_the_groups_add_up_to_the_names(self,
                                            attribution):
        names = attribution.contribution()
        groups = attribution.contribution(field="SECTOR")

        assert sorted(item.asset_id for item in groups.contributions) == [
            "Energy", "Health", "Tech"]
        assert groups.explained == pytest.approx(names.explained, abs=1e-12)
        assert groups.residual == pytest.approx(names.residual, abs=1e-12)

    def test_a_fund_that_starts_holding_later_covers_the_same_days(self):
        """A fund all cash for its first days keeps weights only from when it
        holds something; by name and by group cover the same days."""
        data = consistent()
        tables = dict(data.tables)
        tables[FUND_WEIGHTS] = tables[FUND_WEIGHTS].iloc[40:]
        later = RunData(currency="USD", returns=data.returns + 0.0001, nav=data.nav,
                        units=data.units, initial_capital=1e6, periods_per_year=252,
                        risk_free_rate=0.0, tables=tables, comparators=data.comparators)
        attribution = Attribution(later)

        names = attribution.contribution()
        groups = attribution.contribution(field="SECTOR")

        assert (groups.start, groups.periods) == (names.start, names.periods)
        assert groups.total_return == pytest.approx(names.total_return)

    def test_a_group_s_weight_is_its_names(self,
                                           attribution):
        groups = {item.asset_id: item for item in
                  attribution.contribution(field="SECTOR").contributions}
        tech = [f"N{i:02d}" for i in range(0, 30, 3)]

        assert groups["Tech"].average_weight == pytest.approx(float(FUND[tech].sum()))

    def test_a_field_the_run_did_not_keep(self,
                                          attribution):
        with pytest.raises(KeyError, match="REGION"):
            attribution.contribution(field="REGION")


class TestActive:

    def test_against_the_index(self,
                               attribution):
        found = attribution.active()
        active = {f"N{i:02d}": FUND.iloc[i] - EQUAL.iloc[i] for i in range(30)}

        assert found.reconciles(1e-9)
        assert {item.asset_id: item.average_weight for item in found.contributions} == (
            pytest.approx(active))

    def test_by_sector(self,
                       attribution):
        names = attribution.active()
        groups = attribution.active(field="SECTOR")

        assert groups.explained == pytest.approx(names.explained, abs=1e-12)

    def test_costs_and_cash_are_the_residual(self):
        data = consistent()
        tables = dict(data.tables)
        tables[FUND_WEIGHTS] = tables[FUND_WEIGHTS] * 0.9
        held = RunData(currency="USD", returns=data.returns, nav=data.nav, units=data.units,
                       initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                       tables=tables, comparators=data.comparators)

        found = Attribution(held).contribution()

        assert not found.reconciles()
        assert found.residual == pytest.approx(found.total_return - found.explained)

    def test_an_unknown_comparator(self,
                                   attribution):
        with pytest.raises(KeyError, match="benchmark"):
            attribution.active("benchmark")

    def test_a_book_without_weights(self):
        data = consistent()
        tables = {name: table for name, table in data.tables.items() if name != INDEX_WEIGHTS}
        bare = RunData(currency="USD", returns=data.returns, nav=data.nav, units=data.units,
                       initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                       tables=tables, comparators=data.comparators)

        with pytest.raises(KeyError, match="index"):
            Attribution(bare).contribution("index")
