# tests/test_synthetic_features.py
"""BN-138: generated features.

Four fundamental ratios and two alternative series — enough to screen on and
to exercise the point-in-time path, not an attempt to simulate a data vendor.

**Coherence is the property under test.** A P/E drawn independently of the
prices in the same dataset contradicts them, and the contradiction is
invisible until somebody checks. So `pe x eps` must equal the close at the
period end, exactly, and there is a test that says so rather than a comment
that hopes so.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from beacon.synthetic import SyntheticConfig, generate
from beacon.synthetic import features as features_module

START = "2016-01-04"
END = "2025-12-31"


@pytest.fixture(scope="module")
def panel():
    """A generated dataset carrying features."""
    logging.disable(logging.ERROR)

    try:
        return generate(SyntheticConfig(assets=200, start=START, end=END,
                                        seed=3))
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def rows(panel):
    return panel.features.data.reset_index()


def period_end_of(detail: str) -> pd.Timestamp:
    """The period a fundamental describes, from its DETAIL."""
    body = detail.split("period ending ", maxsplit=1)[1]

    return pd.Timestamp(body.split(",", maxsplit=1)[0])


class TestWhatIsGenerated:
    """The field set the owner asked for."""

    def test_the_fundamental_ratios(self,
                                    panel):
        assert set(panel.features.fields("fundamentals")) == {
            "pe_ratio", "pb_ratio", "eps", "debt_to_equity"}

    def test_the_alternative_series(self,
                                    panel):
        assert set(panel.features.fields("alternative")) == {
            "x_sentiment", "wikipedia_views"}

    def test_it_is_a_small_fraction_of_the_market_panel(self,
                                                        panel):
        """Cheap enough that every store can carry it. At the 6,000-name
        default this is about 8% of the market rows."""
        market = len(panel.market.data)
        features = len(panel.features.data)

        assert features < market * 0.20, (
            f"{features:,} feature rows against {market:,} market rows")

    def test_it_can_be_turned_off(self):
        """A caller who wants the prices and not the ratios should not pay for
        them."""
        logging.disable(logging.ERROR)

        try:
            bare = generate(SyntheticConfig(assets=20, start=START, end=END,
                                            seed=1, features=False))
        finally:
            logging.disable(logging.NOTSET)

        assert bare.features.is_empty


class TestCoherenceWithThePrices:
    """The reason this module derives rather than draws."""

    def test_pe_times_eps_is_the_price(self,
                                       panel,
                                       rows):
        """Exactly, not approximately. A valuation screen and a price screen
        have to agree about the same company, and the only way to guarantee
        that is to derive one from the other.
        """
        fundamentals = rows[rows["TYPE"] == "fundamentals"]
        checked = 0

        for identifier in fundamentals["IDENTIFIER"].unique()[:10]:
            subset = fundamentals[fundamentals["IDENTIFIER"] == identifier]

            for date in subset["DATE"].unique()[:4]:
                on_date = subset[subset["DATE"] == date]
                pe = on_date[on_date["FIELD"] == "pe_ratio"]["VALUE"]
                eps = on_date[on_date["FIELD"] == "eps"]["VALUE"]

                if pe.empty or eps.empty:
                    continue

                detail = on_date[on_date["FIELD"] == "eps"]["DETAIL"].iloc[0]
                closes = panel.market.data.loc[identifier]["CLOSE"]
                usable = closes[closes.index <= period_end_of(detail)]

                assert float(pe.iloc[0]) * float(eps.iloc[0]) == pytest.approx(
                    float(usable.iloc[-1]), rel=1e-5)
                checked += 1

        assert checked > 10, "too few pairs checked to mean anything"

    def test_the_multiples_differ_by_sector(self,
                                            panel,
                                            rows):
        """A generator giving every sector the same multiple would make a
        sector screen and a valuation screen the same screen."""
        fundamentals = rows[(rows["TYPE"] == "fundamentals")
                            & (rows["FIELD"] == "pe_ratio")]
        sectors = panel.universe["SECTOR"]
        by_sector = fundamentals.assign(
            sector=fundamentals["IDENTIFIER"].map(sectors)
        ).groupby("sector")["VALUE"].median()

        assert by_sector.max() > by_sector.min() * 1.4, (
            f"sector medians are nearly flat: {by_sector.round(1).to_dict()}")

    def test_sentiment_is_bounded(self,
                                  rows):
        sentiment = rows[rows["FIELD"] == "x_sentiment"]["VALUE"]

        assert sentiment.between(-1.0, 1.0).all()

    def test_page_views_scale_with_size(self,
                                        panel,
                                        rows):
        """A name nobody has heard of does not trend on a quiet day."""
        views = rows[rows["FIELD"] == "wikipedia_views"]
        median = views.groupby("IDENTIFIER")["VALUE"].median()
        caps = panel.universe.loc[median.index, "market_cap"]

        correlation = np.corrcoef(np.log(caps), np.log(median))[0, 1]

        assert correlation > 0.2, f"views barely track size ({correlation:.2f})"


class TestTheAnnouncementLag:
    """What makes the look-ahead tests mean something."""

    def test_a_value_is_dated_after_the_period_it_describes(self,
                                                            rows):
        fundamentals = rows[rows["TYPE"] == "fundamentals"].head(200)

        for _, row in fundamentals.iterrows():
            assert row["DATE"] > period_end_of(row["DETAIL"])

    def test_the_lag_varies(self,
                            rows):
        """A constant lag would make every look-ahead test pass whether or not
        the accessor was correct: with everything 45 days late, an off-by-one
        still lands in the same gap."""
        fundamentals = rows[(rows["TYPE"] == "fundamentals")
                            & (rows["FIELD"] == "pe_ratio")].head(500)

        lags = {(row["DATE"] - period_end_of(row["DETAIL"])).days
                for _, row in fundamentals.iterrows()}

        assert len(lags) > 10, f"only {len(lags)} distinct lag(s)"

    def test_the_lag_is_plausible(self,
                                  rows):
        fundamentals = rows[(rows["TYPE"] == "fundamentals")
                            & (rows["FIELD"] == "pe_ratio")].head(500)

        lags = [(row["DATE"] - period_end_of(row["DETAIL"])).days
                for _, row in fundamentals.iterrows()]

        assert min(lags) >= features_module.MIN_LAG_DAYS
        assert max(lags) < features_module.MAX_LAG_DAYS

    def test_alternative_data_arrives_far_sooner(self,
                                                 rows):
        """It is near-real-time; a monthly sentiment figure published six
        weeks later would not be alternative data."""
        alternative = rows[rows["TYPE"] == "alternative"].head(300)

        assert not alternative.empty

        lags = [(row["DATE"] - pd.Timestamp(
                    row["DETAIL"].split("month ending ", maxsplit=1)[1])).days
                for _, row in alternative.iterrows()]

        assert max(lags) < features_module.MIN_LAG_DAYS, (
            "alternative data is arriving on fundamental-like delay")


class TestCoverageIsIncomplete:
    """So the missing-coverage behaviour is exercisable against this data."""

    def test_not_every_name_has_fundamentals(self,
                                             panel,
                                             rows):
        covered = set(rows[rows["TYPE"] == "fundamentals"]["IDENTIFIER"])

        assert covered, "nothing was generated at all"
        assert len(covered) < len(panel.universe), (
            "every name has fundamentals, so FeatureRule's missing-coverage "
            "path cannot be exercised against this data")

    def test_alternative_data_covers_far_fewer(self,
                                               panel,
                                               rows):
        """A fraction of the universe, skewed to the visible — which is both
        realistic and what keeps a monthly series affordable."""
        alternative = set(rows[rows["TYPE"] == "alternative"]["IDENTIFIER"])
        fundamentals = set(rows[rows["TYPE"] == "fundamentals"]["IDENTIFIER"])

        assert len(alternative) < len(fundamentals) / 2


    def test_the_covered_names_skew_large(self,
                                          panel,
                                          rows):
        """Coverage is not a coin flip per name.

        A sentiment vendor scrapes what people post about, and people post
        about companies they have heard of. Drawing coverage uniformly would
        make the coverage gap independent of size — so a screen on
        `x_sentiment` would drop names at random rather than dropping the
        obscure ones, which is the wrong shape of hole to test against.
        """
        alternative = set(rows[rows["TYPE"] == "alternative"]["IDENTIFIER"])
        rank = panel.universe["market_cap"].rank(pct=True)

        covered = rank[rank.index.isin(alternative)].mean()
        uncovered = rank[~rank.index.isin(alternative)].mean()

        assert covered > uncovered, (
            f"covered names sit at rank {covered:.2f} against {uncovered:.2f} "
            "for uncovered — coverage is not tilted towards size")

    def test_the_tilt_does_not_change_the_volume(self,
                                                 panel,
                                                 rows):
        """The tilt redistributes coverage, it does not add any: the floor and
        slope are chosen so the mean over a uniform rank is 1."""
        alternative = {*rows[rows["TYPE"] == "alternative"]["IDENTIFIER"]}
        share = len(alternative) / len(panel.universe)

        assert share == pytest.approx(features_module.ALTERNATIVE_COVERAGE,
                                      abs=0.08)


class TestItIsReproducible:
    """The same seed gives the same features."""

    def test_two_runs_agree(self):
        logging.disable(logging.ERROR)

        try:
            tables = [generate(SyntheticConfig(assets=30, start=START,
                                               end=END, seed=8)
                               ).features.data for _ in range(2)]
        finally:
            logging.disable(logging.NOTSET)

        pd.testing.assert_frame_equal(tables[0], tables[1])


    def test_turning_them_off_does_not_move_the_prices(self):
        """The two datasets must agree about everything else.

        Features are drawn from the same generator as the prices, so building
        them before the market data would make `--no-features` silently
        produce a *different* market panel from the same seed — and a bug
        reproduced with the flag on would not reproduce with it off.
        """
        logging.disable(logging.ERROR)

        try:
            panels = [generate(SyntheticConfig(assets=30, start=START,
                                               end=END, seed=11,
                                               features=flag)).market.data
                      for flag in (True, False)]
        finally:
            logging.disable(logging.NOTSET)

        pd.testing.assert_frame_equal(panels[0], panels[1])


class TestItReachesTheServer:
    """End to end, through the surface a client uses."""

    def test_a_generated_feature_is_readable(self,
                                             panel):
        import tempfile
        from pathlib import Path

        from fastapi.testclient import TestClient

        from beacon.server import ServerConfig, create_app

        client = TestClient(create_app(ServerConfig(
            auth_token="t", data_fetcher=panel.fetcher(),
            storage_root=Path(tempfile.mkdtemp()))))

        catalogue = client.get("/data/features/catalogue",
                               headers={"Authorization": "Bearer t"}).json()

        assert {entry["type"] for entry in catalogue["types"]} == {
            "fundamentals", "alternative"}
