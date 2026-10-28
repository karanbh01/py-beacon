# tests/test_factor_definitions.py
"""BN-387: factors, standardised and centred as agreed.

Twelve names on one day, each with a market cap and a sector and region:
five in Tech, five in Energy and two in Health, the first eight in Europe and
the rest in Asia. Each factor reads a value the test sets per name through a
FunctionSignal, so every score can be worked out by hand.
"""
import math

import numpy as np
import pandas as pd
import pytest

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.strategy import Factor, FactorTilts, FieldSignal, default_factors
from beacon.strategy.base import StrategyContext
from beacon.strategy.factors import FactorInputs
from beacon.strategy.signals import SCORE_LIMIT, FunctionSignal, standardised

DAY = pd.Timestamp("2024-06-03")
NAMES = [f"N{i:02d}" for i in range(12)]
SECTORS = ["Tech"] * 5 + ["Energy"] * 5 + ["Health"] * 2
REGIONS = ["Europe"] * 8 + ["Asia"] * 4
CAPS = dict(zip(NAMES, [10.0, 20, 5, 40, 8, 15, 30, 2, 12, 6, 25, 4], strict=True))
VALUES = dict(zip(NAMES, [1.0, 3, -2, 0.5, 4, -1, 2, 6, -3, 1.5, 0, 2.5], strict=True))


def fetcher() -> DataFetcher:
    days = pd.bdate_range("2024-05-27", DAY)
    market = pd.DataFrame([{"IDENTIFIER": name, "DATE": day, "CLOSE": 10.0,
                            "SHARES_OUTSTANDING": CAPS[name] / 10.0 * 1e6}
                           for name in NAMES for day in days])
    reference = pd.DataFrame([{"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                               "CURRENCY": "USD", "EXCHANGE": "XNYS", "SECTOR": sector,
                               "REGION": region}
                              for name, sector, region in zip(NAMES, SECTORS, REGIONS,
                                                              strict=True)])

    return DataFetcher(MarketData.from_dataframe(market), ReferenceData.from_dataframe(reference))


CONTEXT = StrategyContext(fetcher(), "USD")


def given(values: dict[str, float | None] = VALUES,
          **settings) -> FunctionSignal:
    return FunctionSignal(lambda name, date, fetched: values.get(name), **settings)


def scores(factor: Factor,
           names: list[str] = NAMES,
           universe: list[str] | None = None) -> pd.Series:
    return pd.Series(factor.exposures(names, DAY, CONTEXT, universe=universe))


def caps(names: list[str] = NAMES) -> pd.Series:
    """Market caps scaled to add to 1, so a cap-weighted sum is a mean."""
    found = pd.Series({name: CAPS[name] * 1e6 for name in names})

    return found / found.sum()


class TestCentring:

    def test_equal_centring_is_a_signal_s_standardising(self):
        found = scores(Factor("f", given(), centre="equal"))

        assert found.to_dict() == pytest.approx(standardised(VALUES))

    def test_the_market_has_no_exposure_when_cap_weighted(self):
        found = scores(Factor("f", given()))

        assert (found * caps()).sum() == pytest.approx(0.0, abs=1e-9)
        assert found.std(ddof=0) == pytest.approx(1.0)

    def test_by_hand(self):
        values = pd.Series(VALUES)
        centred = values - (values * caps()).sum() / caps().sum()

        found = scores(Factor("f", given()))

        assert found.to_numpy() == pytest.approx((centred / centred.std(ddof=0)).to_numpy())

    def test_scores_are_capped_at_three(self):
        extreme = {**VALUES, "N05": 1e6}

        assert scores(Factor("f", given(extreme), centre="equal")).max() == SCORE_LIMIT


class TestTransformsAndDirection:

    def test_size_is_minus_log_market_cap(self):
        size = Factor("size", FieldSignal(default_factors()[0].signals[0].field),
                      transform="log", higher_is_better=False)
        logs = np.log(caps())
        centred = logs - (logs * caps()).sum() / caps().sum()

        assert scores(size).to_numpy() == pytest.approx(
            (-centred / centred.std(ddof=0)).to_numpy())

    def test_a_value_a_transform_cannot_take_is_missing(self):
        found = scores(Factor("f", given({**VALUES, "N03": -1.0}), transform="log"))

        assert found["N03"] == 0.0

    def test_a_lower_is_better_signal_is_turned_round_first(self):
        found = scores(Factor("f", given(higher_is_better=False), centre="equal"))

        assert found.to_dict() == pytest.approx({name: -score for name, score
                                                 in standardised(VALUES).items()})


class TestComposites:

    def test_a_name_missing_a_part_is_scored_on_the_rest(self):
        other = {name: 2.0 * value for name, value in VALUES.items()}
        other["N04"] = None
        raw = Factor("f", [given(), given(other)], weights=[3.0, 1.0], standardise=False)

        found = scores(raw)

        assert found["N04"] == pytest.approx(VALUES["N04"])
        assert found["N00"] == pytest.approx((3 * 1.0 + 2.0) / 4)

    def test_the_average_is_standardised_again(self):
        found = scores(Factor("f", [given(), given({n: v ** 2 for n, v in VALUES.items()})]))

        assert found.std(ddof=0) == pytest.approx(1.0)
        assert (found * caps()).sum() == pytest.approx(0.0, abs=1e-9)

    def test_a_part_may_be_a_factor_with_its_own_settings(self):
        inner = Factor("inner", given(), centre="equal")

        found = scores(Factor("outer", inner, centre="equal"))

        assert found.to_dict() == pytest.approx(standardised(VALUES))


class TestGroups:

    def test_each_sector_is_centred_on_itself(self):
        found = scores(Factor("f", given(), centre_by="sector"))
        frame = pd.DataFrame({"score": found, "cap": caps(), "sector": SECTORS}, index=NAMES)

        for sector in ("Tech", "Energy"):
            rows = frame[frame["sector"] == sector]
            assert (rows["score"] * rows["cap"]).sum() == pytest.approx(0.0, abs=1e-9)

    def test_a_small_group_is_centred_on_the_universe(self):
        factor = Factor("f", given(), centre_by="sector")
        inputs = FactorInputs(CONTEXT, DAY, NAMES)
        factor.exposures(NAMES, DAY, CONTEXT, inputs=inputs)

        assert inputs.fallbacks == {"f": {("Health",): ()}}

    def test_two_levels_fall_back_one_at_a_time(self):
        factor = Factor("f", given(), centre_by=["region", "sector"])
        inputs = FactorInputs(CONTEXT, DAY, NAMES)
        factor.exposures(NAMES, DAY, CONTEXT, inputs=inputs)

        # Europe holds 5 Tech names and 3 Energy; Asia 2 Energy and 2 Health.
        assert inputs.fallbacks["f"] == {("Europe", "Energy"): ("Europe",),
                                         ("Asia", "Energy"): (),
                                         ("Asia", "Health"): ()}

    def test_scaled_within_groups(self):
        found = scores(Factor("f", given(), centre_by="sector", scale_within_groups=True))
        frame = pd.DataFrame({"score": found, "sector": SECTORS}, index=NAMES)

        assert frame[frame["sector"] == "Tech"]["score"].std(ddof=0) == pytest.approx(1.0)


class TestTheUniverse:

    def test_names_outside_it_are_scored_on_its_figures(self):
        universe = NAMES[:10]
        values = pd.Series(VALUES)[universe]
        weights = caps(universe)
        centred = pd.Series(VALUES) - (values * weights).sum() / weights.sum()
        spread = centred[universe].std(ddof=0)

        found = scores(Factor("f", given()), names=NAMES, universe=universe)

        assert found["N11"] == pytest.approx(centred["N11"] / spread)

    def test_imported_exposures_are_taken_as_they_are(self):
        found = scores(Factor("f", given(), standardise=False))

        assert found.to_dict() == VALUES


class TestWhereFactorsServe:

    def test_a_factor_tilt_takes_a_factor(self):
        factor = Factor("value", given())
        tilts = FactorTilts({"value": factor}, {"value": 0.3})

        inputs = tilts.inputs(NAMES, DAY, CONTEXT)

        assert inputs["value"] == pytest.approx(factor.exposures(NAMES, DAY, CONTEXT))

    def test_the_defaults_are_size_value_and_momentum(self):
        assert [factor.name for factor in default_factors()] == ["size", "value", "momentum"]

    def test_the_server_builds_one_from_json(self):
        from beacon import catalogue
        from beacon.server.backtest_settings import _built, typed_payload

        factor = Factor("size", FieldSignal(default_factors()[0].signals[0].field),
                        transform="log", higher_is_better=False, centre_by=["region", "sector"])

        payload = typed_payload(catalogue.SIGNAL, factor)
        rebuilt = _built(catalogue.SIGNAL, payload)

        assert isinstance(rebuilt, Factor)
        assert typed_payload(catalogue.SIGNAL, rebuilt) == payload
        assert payload["params"]["signals"][0]["type"] == "FieldSignal"


class TestRefused:

    @pytest.mark.parametrize(("settings", "message"), [
        ({"name": ""}, "name"), ({"signals": []}, "signals"),
        ({"weights": [1.0, 2.0]}, "weights"), ({"weights": [-1.0]}, "weights"),
        ({"transform": "sqrt"}, "sqrt"), ({"centre": "median"}, "median")])
    def test_settings_out_of_range(self,
                                   settings,
                                   message):
        arguments = {"name": "f", "signals": given(), **settings}

        with pytest.raises(ValueError, match=message):
            Factor(**arguments)

    def test_no_values_score_zero(self):
        assert set(scores(Factor("f", given({}))).to_numpy()) == {0.0}
        assert not math.isnan(scores(Factor("f", given({}))).sum())
