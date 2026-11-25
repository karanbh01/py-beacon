# tests/test_statistic_enum.py
"""BN-432: `Statistic`, every statistic's id as an enum, held to the catalogue."""
import json

import pytest

from beacon.analysis import Statistic
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.request import AnalysisRequest


class TestAgainstTheCatalogue:
    def test_every_id_once_in_the_catalogues_order(self):
        assert [member.value for member in Statistic] == [spec.id for spec in CATALOGUE]

    def test_none_extra(self):
        assert len(Statistic) == len(CATALOGUE) == len(set(Statistic))

    def test_each_member_is_named_by_its_id_in_capitals(self):
        assert all(member.name == member.value.upper() for member in Statistic)


class TestAsAnId:
    def test_a_member_is_its_id(self):
        assert Statistic.SHARPE_RATIO == "sharpe_ratio"
        assert str(Statistic.SHARPE_RATIO) == "sharpe_ratio"
        assert f"{Statistic.SHARPE_RATIO}" == "sharpe_ratio"

    def test_an_id_gives_its_member(self):
        assert Statistic("tracking_error") is Statistic.TRACKING_ERROR

    def test_the_catalogue_is_read_by_member(self):
        assert CATALOGUE[Statistic.SHARPE_RATIO].label == "Sharpe ratio"
        assert Statistic.SHARPE_RATIO in CATALOGUE

    def test_an_unknown_id_has_no_member(self):
        with pytest.raises(ValueError, match="sharp_ratio"):
            Statistic("sharp_ratio")


class TestInARequest:
    MEMBERS = AnalysisRequest(statistics=[Statistic.SHARPE_RATIO],
                              add={"factsheet": [Statistic.SORTINO_RATIO]},
                              remove={"factsheet": [Statistic.VOLATILITY]},
                              time_views={"factsheet": {Statistic.TOTAL_RETURN: "both"}})
    IDS = AnalysisRequest(statistics=["sharpe_ratio"],
                          add={"factsheet": ["sortino_ratio"]},
                          remove={"factsheet": ["volatility"]},
                          time_views={"factsheet": {"total_return": "both"}})

    def test_members_choose_what_their_ids_choose(self):
        assert self.MEMBERS.selection() == self.IDS.selection()
        assert self.MEMBERS.wanted() == self.IDS.wanted()

    def test_members_are_kept_as_plain_ids(self):
        data = self.MEMBERS.as_data()
        ids = [*data["statistics"], *data["add"]["factsheet"], *data["remove"]["factsheet"],
               *data["time_views"]["factsheet"]]

        assert {type(id_) for id_ in ids} == {str}
        assert json.dumps(data) == json.dumps(self.IDS.as_data())

    def test_a_member_not_optional_is_refused_as_an_id_is(self):
        with pytest.raises(ValueError, match="volatility"):
            AnalysisRequest(add={"factsheet": [Statistic.VOLATILITY]})
