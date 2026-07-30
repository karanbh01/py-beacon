# tests/test_testing_dataset.py
"""BN-95: the canonical synthetic dataset.

The dataset's whole value is that it does not move. These tests are what makes
that true rather than hoped: a change to the generation breaks them loudly,
which is the point — every baseline built on it depends on the numbers staying
put.
"""
import hashlib
import subprocess
import sys

import pandas as pd
import pytest

from beacon.testing import dataset

# The hash of the price frame as first generated. If this changes, every image
# baseline and every recorded expectation built on the dataset has changed too,
# so the failure is meant to be loud rather than a nuisance to update.
PRICES_DIGEST = "6263faa7aab1c2ee"


def digest(frame: pd.DataFrame) -> str:
    """A short, stable fingerprint of a frame's numbers.

    Hashes the raw float64 bytes rather than a CSV rendering. The first version
    of this used `to_csv()` and failed on every non-Windows runner: pandas
    picks the platform line terminator, so `

` against `
` changed the
    hash while every number in the frame was identical. The values are what the
    dataset promises to keep stable, so the values are what this hashes.
    """
    return hashlib.sha256(frame.to_numpy().tobytes()).hexdigest()[:16]


class TestDeterminism:

    def test_two_calls_agree(self):
        pd.testing.assert_frame_equal(dataset.prices(), dataset.prices())

    def test_two_independent_modules_see_identical_frames(self):
        """The acceptance criterion, at module scope.

        Imported here under a second name to stand in for a second test module:
        what matters is that nothing about the first read changes the second.
        """
        from beacon.testing import dataset as second_import

        pd.testing.assert_frame_equal(dataset.prices(), second_import.prices())
        pd.testing.assert_frame_equal(dataset.market_frame(),
                                      second_import.market_frame())

    def test_a_fresh_interpreter_produces_the_same_numbers(self):
        """No dependence on import order, RNG state or anything else ambient."""
        script = ("from beacon.testing import dataset;"
                  "import hashlib;"
                  "print(hashlib.sha256(dataset.prices().to_numpy().tobytes())"
                  ".hexdigest()[:16])")

        first = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, check=True)
        second = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, check=True)

        assert first.stdout == second.stdout
        assert first.stdout.strip() == PRICES_DIGEST

    def test_the_prices_match_their_recorded_digest(self):
        assert digest(dataset.prices()) == PRICES_DIGEST

    def test_the_index_is_stable_too(self):
        """The digest covers the values; the dates need their own check."""
        index = dataset.prices().index

        assert (str(index[0].date()), str(index[-1].date()), len(index)) == (
            "2023-01-02", "2025-12-31", 783)

    def test_callers_cannot_disturb_each_other(self):
        """Each caller gets a copy; one test's mutation is not another's bug."""
        first = dataset.prices()
        first.iloc[0, 0] = -999.0

        assert dataset.prices().iloc[0, 0] != -999.0

    def test_the_paths_avoid_transcendental_functions(self):
        """A guard on the reproducibility argument, not on style.

        `exp` and `log` come from the platform's libm and may differ in the last
        bit between operating systems, which compounding amplifies. The paths
        are built from `+` and `*` so every runner agrees exactly, and this
        test is here so that reasoning cannot be quietly undone.
        """
        from pathlib import Path

        source = Path(dataset.__file__).read_text(encoding="utf-8")
        body = source.split('"""', 2)[-1]

        assert "np.exp" not in body
        assert "np.log" not in body


class TestShape:

    def test_every_constituent_has_a_price_path(self):
        assert list(dataset.prices().columns) == list(dataset.UNIVERSE)

    def test_the_span_matches_the_declared_dates(self):
        prices = dataset.prices()

        assert prices.index[0] == pd.Timestamp(dataset.START)
        assert prices.index[-1] <= pd.Timestamp(dataset.END)

    def test_the_calendar_is_business_days_only(self):
        assert (dataset.prices().index.dayofweek < 5).all()

    def test_prices_are_positive_throughout(self):
        """A path that crossed zero would break every return calculation."""
        assert (dataset.prices() > 0).all().all()

    def test_no_missing_values(self):
        assert not dataset.prices().isna().any().any()

    def test_returns_drop_the_undefined_first_row(self):
        assert len(dataset.returns()) == len(dataset.prices()) - 1
        assert not dataset.returns().isna().any().any()


class TestDocumentedBehaviour:
    """The docstring makes claims about the universe. They have to be true."""

    @pytest.fixture(scope="class")
    def annualised_volatility(self):
        return dataset.returns().std() * (252 ** 0.5)

    def test_ccc_is_the_least_volatile(self,
                                       annualised_volatility):
        assert annualised_volatility.idxmin() == "CCC"

    def test_ddd_is_the_most_volatile(self,
                                      annualised_volatility):
        assert annualised_volatility.idxmax() == "DDD"

    def test_aaa_and_bbb_are_close_substitutes(self):
        """Documented as the pair a risk model should notice."""
        returns = dataset.returns()

        assert returns["AAA"].corr(returns["BBB"]) > 0.75

    def test_ccc_diversifies_rather_than_merely_being_quiet(self):
        """The retuning that turned a corner solution into an interior one."""
        returns = dataset.returns()

        assert returns["AAA"].corr(returns["CCC"]) < 0.55

    def test_risk_and_return_rankings_disagree(self):
        """Otherwise an optimiser has no trade-off to make."""
        prices = dataset.prices()
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1.0
        volatility = dataset.returns().std()

        assert total_return.idxmax() != volatility.idxmax()


class TestMarketData:

    def test_the_container_covers_the_universe_and_the_fx_pair(self):
        identifiers = dataset.market_data().identifiers

        assert set(dataset.UNIVERSE) <= set(identifiers)
        assert dataset.FX_PAIR in identifiers

    def test_it_carries_the_columns_the_pipeline_reads(self):
        columns = set(dataset.market_data().columns)

        assert {"CLOSE", "OPEN", "HIGH", "LOW", "VOLUME",
                "SHARES_OUTSTANDING", "FREE_FLOAT"} <= columns

    def test_the_intraday_range_is_ordered(self):
        """Low above high inside an unrelated test is a bad afternoon."""
        frame = dataset.market_frame()
        quoted = frame[frame["IDENTIFIER"] != dataset.FX_PAIR]

        assert (quoted["HIGH"] >= quoted["CLOSE"]).all()
        assert (quoted["LOW"] <= quoted["CLOSE"]).all()

    def test_close_matches_the_price_frame(self):
        """Two views of one dataset, not two datasets."""
        frame = dataset.market_frame()
        rows = frame[frame["IDENTIFIER"] == "AAA"].set_index("DATE")["CLOSE"]

        pd.testing.assert_series_equal(rows, dataset.prices()["AAA"],
                                       check_names=False, check_freq=False)


class TestFetcher:

    @pytest.fixture(scope="class")
    def fetcher(self):
        return dataset.data_fetcher()

    def test_market_data_comes_back_for_one_identifier(self,
                                                       fetcher):
        frame = fetcher.fetch_market_data("AAA", dataset.START, dataset.END)

        assert not frame.empty
        assert "CLOSE" in frame.columns

    def test_reference_data_resolves(self,
                                     fetcher):
        reference = fetcher.fetch_reference_data("FFF")

        assert reference.iloc[0]["CURRENCY"] == "GBP"
        assert reference.iloc[0]["NAME"] == "Zeta Holdings"

    def test_shares_outstanding_are_available(self,
                                              fetcher):
        shares = fetcher.fetch_shares_outstanding("AAA", pd.Timestamp("2024-06-03"))

        assert shares == 1_000_000_000

    def test_free_float_is_available(self,
                                     fetcher):
        assert fetcher.fetch_free_float_factor(
            "DDD", pd.Timestamp("2024-06-03")) == pytest.approx(0.75)

    def test_fx_rates_resolve_for_the_non_usd_constituent(self,
                                                          fetcher):
        """FFF trades in GBP, so anything touching FX has a real case."""
        rates = fetcher.fetch_fx_rates("GBP", "USD")

        assert not rates.empty
        assert (rates > 0).all()


class TestHelpers:

    def test_sectors_partition_the_universe(self):
        grouped = dataset.sectors()
        members = [name for names in grouped.values() for name in names]

        assert sorted(members) == sorted(dataset.UNIVERSE)

    def test_the_technology_sector_holds_the_substitute_pair(self):
        assert dataset.sectors()["Technology"] == ["AAA", "BBB"]

    def test_equal_weights_sum_to_one(self):
        assert sum(dataset.equal_weights().values()) == pytest.approx(1.0)

    def test_trading_days_match_the_price_index(self):
        pd.testing.assert_index_equal(dataset.trading_days(), dataset.prices().index)

    def test_reference_frame_covers_every_constituent(self):
        assert sorted(dataset.reference_frame()["IDENTIFIER"]) == sorted(
            dataset.UNIVERSE)
