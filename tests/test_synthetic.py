# tests/test_synthetic.py
"""BN-114: the synthetic generator, and whether its output behaves like a market.

Most of this file tests *distributions*, which is a different discipline from
testing a return value. Two rules follow from that and are worth stating,
because breaking either produces a test suite that passes while the data is
wrong:

**Assert against the target, not the sample, wherever the target exists.** A
name's volatility target is exact by construction; its realised volatility is
one draw from a GARCH process and can legitimately land well above it. Testing
the realised figure against the design band would be testing the sampler.

**Fix the seed and pick tolerances from measurement.** Every statistical
assertion here was run across several seeds first, and the bound set outside
the observed spread. A tolerance loosened until a failure went away would be
worse than no test.
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from beacon.data import store
from beacon.data.corporate_actions import CASH_ACTIONS, RATIO_ACTIONS
from beacon.synthetic import SyntheticConfig, generate, write
from beacon.synthetic import universe as universe_module
from beacon.synthetic.__main__ import build_parser, default_window, main, resolve_window
from beacon.synthetic.prices import PRICE_DECIMALS, SPLIT_THRESHOLD

TOKEN = "test-token-value"

# Big enough for the statistics to mean something, small enough to generate in
# a couple of seconds. Every distributional assertion below was checked across
# seeds 1-7 before its bound was set.
PANEL_ASSETS = 150
PANEL_SEED = 7

# Explicitly a calm window. The stylized facts below are properties of a
# *stationary* market, and since BN-128 the default configuration spans covid
# and the 2022 drawdown, where volatility trebles and correlations rise toward
# one by design. Testing "average pairwise correlation is 0.3 to 0.5" inside a
# crisis is testing the wrong claim, not finding a defect -- the crisis
# behaviour has its own assertions in TestCrisisRegimes.
CALM_START = "2013-01-02"
CALM_END = "2018-12-31"

# Well-known real symbols. The CMP prefix makes a collision impossible by
# construction, so this asserts the construction rather than hunting for luck.
REAL_TICKERS = frozenset({
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "CVX", "KO",
    "PEP", "ABBV", "COST", "MRK", "BAC", "CRM", "AMD", "NFLX", "INTC", "IBM",
    "T", "F", "GM", "GE", "DIS", "SPY", "QQQ", "VOO", "A", "C", "M", "X",
})


@pytest.fixture(scope="module")
def panel():
    """One generated dataset over a calm window, shared by the statistics."""
    return generate(SyntheticConfig(assets=PANEL_ASSETS, seed=PANEL_SEED,
                                    start=CALM_START, end=CALM_END))


def ljung_box(series: np.ndarray,
              lags: int = 10,
              robust: bool = False) -> float:
    """The Ljung-Box p-value for autocorrelation up to `lags`.

    Args:
        series: The series to test.
        lags: How many autocorrelations to pool.
        robust: Use the heteroskedasticity-consistent variance.

    Returns:
        float: The p-value under a chi-squared with `lags` degrees of freedom.

    The robust form matters here and is not a technicality. The textbook
    statistic assumes the series is iid under the null, but a GARCH process has
    returns that are *uncorrelated and not independent* — which inflates the
    sampling variance of every autocorrelation and makes the naive test reject
    for roughly a third of names that have no serial correlation at all. The
    correction (Diebold 1986) divides each squared autocorrelation by its
    actual variance under conditional heteroskedasticity. Without it, this file
    would report a defect that is entirely an artefact of the wrong test.
    """
    from scipy import stats

    centred = np.asarray(series, dtype=float)
    centred = centred - centred.mean()
    count = len(centred)
    denominator = float(centred @ centred)

    statistic = 0.0
    for lag in range(1, lags + 1):
        head, tail = centred[:-lag], centred[lag:]
        rho = float(head @ tail) / denominator

        variance = 1.0
        if robust:
            products = (head * tail) ** 2
            variance = products.sum() * count / denominator ** 2

        statistic += count * (count + 2) * rho ** 2 / ((count - lag) * variance)

    return float(1.0 - stats.chi2.cdf(statistic, lags))


class TestNaming:
    """Nothing generated may resemble a real listing."""

    def test_suffixes_run_in_spreadsheet_order(self):
        assert universe_module.ticker_suffix(0) == "A"
        assert universe_module.ticker_suffix(25) == "Z"
        assert universe_module.ticker_suffix(26) == "AA"
        assert universe_module.ticker_suffix(27) == "AB"
        assert universe_module.ticker_suffix(51) == "AZ"
        assert universe_module.ticker_suffix(52) == "BA"

    def test_identifiers_are_unique_at_scale(self):
        tickers = universe_module.identifiers(2048)

        assert len(set(tickers)) == 2048

    def test_no_generated_ticker_is_a_real_one(self):
        """The prefix makes this impossible rather than unlikely, but the
        guarantee is only as good as the function that provides it."""
        assert not set(universe_module.identifiers(5000)) & REAL_TICKERS

    def test_names_and_tickers_agree(self, panel):
        reference = panel.reference.data

        for identifier, name in reference["NAME"].items():
            assert name == universe_module.company_name(identifier)

    def test_names_carry_no_real_company(self, panel):
        names = panel.reference.data["NAME"]

        assert names.str.startswith("Company ").all()

    def test_sub_industries_are_generic(self, panel):
        """Real GICS sub-industry names would imply a classification a random
        draw has not earned."""
        for value in panel.reference.data["SUB_INDUSTRY"].unique():
            sector, segment = value.split(" — ")

            assert sector in universe_module.SECTORS
            assert segment in universe_module.SEGMENTS


class TestUniverse:
    """The static draw."""

    def test_every_sector_is_populated(self, panel):
        sectors = set(panel.reference.data["SECTOR"])

        assert sectors == set(universe_module.SECTORS)

    def test_volatility_targets_sit_in_the_design_band(self, panel):
        """Exact by construction, unlike the realised figures below."""
        volatility = panel.universe["volatility"]

        assert volatility.min() >= universe_module.MIN_VOLATILITY
        assert volatility.max() <= universe_module.MAX_VOLATILITY

    def test_market_cap_has_a_heavy_tail(self, panel):
        """Cap weighting and cap rules only behave interestingly if a few names
        dominate. On a uniform draw the cap would never bind."""
        caps = panel.universe["market_cap"].sort_values(ascending=False)
        share = caps.iloc[:len(caps) // 10].sum() / caps.sum()

        assert share > 0.35, f"top decile holds only {share:.1%} of the cap"
        assert caps.max() / caps.median() > 15

    def test_free_float_is_a_fraction(self, panel):
        floats = panel.universe["free_float"]

        assert floats.between(0.0, 1.0).all()
        assert floats.min() < 0.9, "no name is meaningfully float-restricted"

    def test_shares_follow_from_cap_and_price(self, panel):
        """Drawn independently, a name could carry a share count its own price
        contradicts."""
        implied = panel.universe["shares_outstanding"] * panel.universe["initial_price"]

        assert implied.div(panel.universe["market_cap"]).between(0.99, 1.01).all()


class TestStylizedFacts:
    """The reason this is a model and not a random walk."""

    def test_every_name_has_positive_excess_kurtosis(self, panel):
        kurtosis = panel.returns.kurt()

        assert (kurtosis > 0).all(), (
            f"{(kurtosis <= 0).sum()} name(s) are not fat-tailed; "
            f"minimum was {kurtosis.min():.2f}")

    def test_returns_are_negatively_skewed(self, panel):
        """Per-name skew is a noisy estimate on t-distributed data, so this
        asserts on the cross-section rather than on every name."""
        skew = panel.returns.skew()

        assert skew.median() < -0.2
        assert (skew < 0).mean() > 0.8

    def test_squared_returns_are_autocorrelated(self, panel):
        """The volatility-clustering signature."""
        rejected = [ljung_box(panel.returns[name].to_numpy() ** 2) < 0.05
                    for name in panel.returns]

        assert np.mean(rejected) > 0.95

    def test_raw_returns_are_not(self, panel):
        """Same test, same lags, on the levels — and it must mostly *not*
        reject, or the series has a predictable drift it should not have."""
        rejected = [ljung_box(panel.returns[name].to_numpy(), robust=True) < 0.05
                    for name in panel.returns]

        assert np.mean(rejected) < 0.15, (
            f"{np.mean(rejected):.0%} of names show serial correlation in "
            f"levels; returns should be close to a martingale difference")

    def test_average_pairwise_correlation_is_realistic(self, panel):
        correlation = panel.returns.corr().to_numpy()
        upper = np.triu_indices_from(correlation, 1)

        assert 0.30 <= correlation[upper].mean() <= 0.50

    def test_sector_pairs_correlate_more_than_cross_sector_pairs(self, panel):
        correlation = panel.returns.corr().to_numpy()
        upper = np.triu_indices_from(correlation, 1)

        sectors = panel.universe["SECTOR"].to_numpy()
        same = sectors[upper[0]] == sectors[upper[1]]

        within = correlation[upper][same].mean()
        across = correlation[upper][~same].mean()

        assert within > across + 0.05, f"within {within:.3f}, across {across:.3f}"

    def test_volatility_is_dispersed_across_the_universe(self, panel):
        realised = panel.returns.std() * np.sqrt(252)

        assert realised.quantile(0.1) < 0.25
        assert realised.quantile(0.9) > 0.33


class TestMarketData:
    """The panel a client reads."""

    def test_open_high_low_close_are_coherent(self, panel):
        frame = panel.market.data
        highest = frame[["OPEN", "CLOSE"]].max(axis=1)
        lowest = frame[["OPEN", "CLOSE"]].min(axis=1)

        assert (frame["HIGH"] >= highest).all()
        assert (frame["LOW"] <= lowest).all()
        assert (frame["HIGH"] >= frame["LOW"]).all()

    def test_prices_are_positive(self, panel):
        frame = panel.market.data

        for column in ("OPEN", "HIGH", "LOW", "CLOSE"):
            assert (frame[column] > 0).all(), column

    def test_volume_is_non_negative_and_whole(self, panel):
        volume = panel.market.data["VOLUME"]

        assert (volume >= 0).all()
        assert (volume == volume.round()).all()

    def test_volume_rises_with_the_size_of_the_move(self, panel):
        """Without this, ADV is a constant with noise on it and a liquidity
        screen built on it would never bind for the right reason."""
        frame = panel.market.data
        volume = frame["VOLUME"].unstack("IDENTIFIER")
        moves = panel.returns.abs()

        aligned = volume.reindex(moves.index)
        correlations = [aligned[name].corr(moves[name]) for name in moves]

        assert np.median(correlations) > 0.15

    def test_every_field_the_fetcher_reads_is_present(self, panel):
        assert set(panel.market.columns) >= {
            "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
            "SHARES_OUTSTANDING", "FREE_FLOAT"}

    def test_shares_and_free_float_are_queryable(self, panel):
        fetcher = panel.fetcher()
        identifier = panel.universe.index[0]
        date = panel.returns.index[100].strftime("%Y-%m-%d")

        assert fetcher.fetch_shares_outstanding(identifier, date) > 0
        assert 0.0 < fetcher.fetch_free_float_factor(identifier, date) <= 1.0


class TestCorporateActions:
    """Dividends and splits, and their agreement with the price path."""

    def test_only_known_action_types_are_produced(self, panel):
        types = set(panel.actions.data["TYPE"])

        assert types <= CASH_ACTIONS | RATIO_ACTIONS

    def test_dividends_are_quarterly_for_payers(self, panel):
        actions = panel.actions.data.reset_index(drop=True)
        dividends = actions[actions["TYPE"] == "DIVIDEND"]
        payers = panel.universe.index[panel.universe["dividend_yield"] > 0]

        assert set(dividends["IDENTIFIER"]) == set(payers)

        years = panel.returns.index.year.nunique()
        per_payer = dividends.groupby("IDENTIFIER").size()

        assert per_payer.between(4 * (years - 2), 4 * years).all()

    def test_non_payers_pay_nothing(self, panel):
        actions = panel.actions.data.reset_index(drop=True)
        dividends = set(actions[actions["TYPE"] == "DIVIDEND"]["IDENTIFIER"])
        non_payers = panel.universe.index[panel.universe["dividend_yield"] == 0]

        assert not dividends & set(non_payers)

    def test_splits_happen_and_follow_a_high_price(self):
        """Generated over its own window rather than the shared calm one: a
        split needs a price to have run up past the threshold, and six calm
        years at a 9% drift do not get a single name there. Testing it on a
        panel with no splits would assert nothing."""
        panel = generate(SyntheticConfig(assets=PANEL_ASSETS, seed=PANEL_SEED,
                                         start="2010-01-04", end="2024-12-31"))

        actions = panel.actions.data.reset_index(drop=True)
        splits = actions[actions["TYPE"] == "SPLIT"]

        assert not splits.empty, "no split in the whole panel"

        closes = panel.market.data["CLOSE"].unstack("IDENTIFIER")
        for identifier, ex_date, ratio in zip(splits["IDENTIFIER"],
                                              splits["EX_DATE"],
                                              splits["VALUE"], strict=True):
            # The close on the ex-date is already divided by this split, so
            # multiplying it back gives the price the review actually saw.
            quoted = closes[identifier].loc[ex_date] * ratio

            assert quoted > SPLIT_THRESHOLD

    def test_a_split_moves_the_share_count(self, panel):
        """A split that changed the price and not the shares outstanding would
        halve the market cap, and every cap-weighted figure with it."""
        actions = panel.actions.data.reset_index(drop=True)
        splits = actions[actions["TYPE"] == "SPLIT"]
        shares = panel.market.data["SHARES_OUTSTANDING"].unstack("IDENTIFIER")

        for identifier, ex_date, ratio in zip(splits["IDENTIFIER"],
                                              splits["EX_DATE"],
                                              splits["VALUE"], strict=True):
            series = shares[identifier]
            before = series.loc[:ex_date].iloc[-2]

            assert series.loc[ex_date] == pytest.approx(before * ratio)

    def test_the_trailing_dividend_is_answerable(self, panel):
        """The point of generating actions at all: the pane has content."""
        fetcher = panel.fetcher()
        payer = panel.universe.index[panel.universe["dividend_yield"] > 0][0]
        as_of = panel.returns.index[-1].strftime("%Y-%m-%d")

        assert fetcher.fetch_trailing_dividend(payer, as_of) > 0


class TestCoherence:
    """The datasets agree with each other."""

    def test_undoing_actions_recovers_the_economic_path(self, panel):
        """Splits out, dividends back in — the stored prices must return the
        returns they were built from. "Mutually coherent" is otherwise a claim
        rather than a property."""
        closes = panel.market.data["CLOSE"].unstack("IDENTIFIER")
        actions = panel.actions.data.reset_index(drop=True)

        ratio = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)
        splits = actions[actions["TYPE"] == "SPLIT"]
        for identifier, ex_date, value in zip(splits["IDENTIFIER"],
                                              splits["EX_DATE"],
                                              splits["VALUE"], strict=True):
            ratio.loc[ex_date:, identifier] *= value

        cash = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        dividends = actions[actions["TYPE"] == "DIVIDEND"]
        for identifier, ex_date, value in zip(dividends["IDENTIFIER"],
                                              dividends["EX_DATE"],
                                              dividends["VALUE"], strict=True):
            cash.loc[ex_date, identifier] = value

        adjusted = closes * ratio
        rebuilt = (adjusted + cash * ratio) / adjusted.shift(1) - 1.0

        difference = (rebuilt.iloc[1:] - panel.returns.iloc[1:]).abs().max().max()

        # Derived from the data rather than fixed. Prices are stored at feed
        # precision, so the reconstruction inherits a half-tick of rounding on
        # the lowest-priced name — which means the achievable floor depends on
        # how low prices got, and a constant silently becomes either slack or
        # unmeetable when the window changes. A fixed 1e-4 held for a five-year
        # window and failed on a six-year one for no reason but a cheaper stock.
        #
        # The bound stays tight: a few ticks, not orders of magnitude. At 1e-3
        # this test passed while dividends recorded after a split were being
        # reported at twice the cash a holder actually received.
        tick = 10.0 ** -PRICE_DECIMALS
        floor = tick / float(closes.min().min())

        assert difference < 20 * floor, (
            f"largest discrepancy {difference:.2e}, floor {floor:.2e}")

    def test_the_datasets_cover_the_same_names(self, panel):
        assert set(panel.market.identifiers) == set(panel.reference.identifiers)
        assert set(panel.actions.identifiers) <= set(panel.market.identifiers)

    def test_the_panel_spans_the_requested_window(self):
        dataset = generate(SyntheticConfig(assets=8, start="2021-01-04",
                                           end="2021-06-30", seed=1))
        start, end = dataset.market.date_range

        assert start == pd.Timestamp("2021-01-04")
        assert end == pd.Timestamp("2021-06-30")


class TestDeterminism:
    """Same seed, same dataset — the guarantee the CLI advertises."""

    def test_the_same_seed_writes_byte_identical_files(self, tmp_path):
        config = SyntheticConfig(assets=24, start="2022-01-03",
                                 end="2023-01-03", seed=99)

        first = write(config, tmp_path / "one")
        second = write(config, tmp_path / "two")

        for name in (store.MANIFEST_NAME, store.MARKET_FILE,
                     store.REFERENCE_FILE, store.ACTIONS_FILE):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def test_a_different_seed_gives_different_data(self, tmp_path):
        """Otherwise the previous test would pass on a generator that ignored
        its seed entirely."""
        base = SyntheticConfig(assets=24, start="2022-01-03", end="2023-01-03")

        first = write(base, tmp_path / "one")
        second = write(SyntheticConfig(assets=24, start="2022-01-03",
                                       end="2023-01-03", seed=1), tmp_path / "two")

        assert ((first / store.MARKET_FILE).read_bytes()
                != (second / store.MARKET_FILE).read_bytes())

    def test_it_is_written_as_a_synthetic_store(self, tmp_path):
        path = write(SyntheticConfig(assets=8, start="2022-01-03",
                                     end="2022-06-03"), tmp_path / "store")

        assert store.read_manifest(path).source == store.SOURCE_SYNTHETIC


class TestConfiguration:
    """Arguments that do not describe a dataset are refused."""

    def test_an_empty_universe_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            SyntheticConfig(assets=0)

    def test_a_backwards_window_is_refused(self):
        with pytest.raises(ValueError, match="must fall after"):
            SyntheticConfig(start="2024-01-01", end="2023-01-01")

    def test_the_equity_premium_moves_the_market(self):
        """The single number that decides whether the generated history looks
        like a bull market or a lost decade."""
        flat = generate(SyntheticConfig(assets=40, seed=5, equity_premium=0.0))
        rich = generate(SyntheticConfig(assets=40, seed=5, equity_premium=0.15))

        assert rich.returns.mean().mean() > flat.returns.mean().mean()


class TestCommandLine:
    """The documented invocation."""

    def test_the_documented_arguments_parse(self):
        args = build_parser().parse_args(
            ["--assets", "512", "--start", "2019-12-31", "--seed", "42",
             "--out", "/tmp/store"])

        assert args.assets == 512
        assert args.start == "2019-12-31"
        assert args.seed == 42
        assert args.out == Path("/tmp/store")

    def test_the_default_window_is_five_years_ending_today(self):
        start, end = default_window(pd.Timestamp("2026-08-03").date())

        assert (start, end) == ("2021-08-03", "2026-08-03")

    def test_an_explicit_end_backs_the_start_off_it(self):
        assert resolve_window(None, "2020-01-01")[0] == "2015-01-01"

    def test_an_explicit_start_is_kept(self):
        assert resolve_window("2010-01-01", "2020-01-01") == ("2010-01-01",
                                                              "2020-01-01")

    def test_it_writes_a_loadable_store(self, tmp_path, capsys):
        code = main(["--assets", "12", "--start", "2022-01-03",
                     "--end", "2022-12-30", "--seed", "3",
                     "--out", str(tmp_path / "store")])

        assert code == 0
        assert "Wrote 12 identifiers" in capsys.readouterr().out

        fetcher = store.load(tmp_path / "store")
        assert len(fetcher.identifiers) == 12

    def test_a_bad_window_exits_two(self, tmp_path, capsys):
        code = main(["--start", "2024-01-01", "--end", "2023-01-01",
                     "--out", str(tmp_path / "store")])

        assert code == 2
        assert "error:" in capsys.readouterr().err


class TestNotTheFixture:
    """`beacon.testing.dataset` and this package are different things."""

    def test_the_frozen_fixture_is_untouched(self):
        """Chart baselines depend on the fixture's exact values. If generating
        synthetic data ever changed them, eighteen image comparisons would fail
        and the cause would not be obvious."""
        from beacon.testing import dataset as fixture

        generate(SyntheticConfig(assets=8, start="2022-01-03", end="2022-06-03"))

        assert list(fixture.UNIVERSE) == ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        assert fixture.prices().iloc[0].tolist() == pytest.approx(
            fixture.prices().iloc[0].tolist())

    def test_they_share_no_identifiers(self):
        """So a test cannot accidentally mix the two and still look sane."""
        from beacon.testing import dataset as fixture

        assert not set(fixture.UNIVERSE) & set(universe_module.identifiers(1000))


class TestServedByTheServer:
    """The end-to-end criterion: generate, spawn, and the client sees data."""

    @pytest.mark.timeout(120)
    def test_every_data_endpoint_serves_the_generated_store(self, tmp_path):
        path = write(SyntheticConfig(assets=16, start="2022-01-03",
                                     end="2023-12-29", seed=11),
                     tmp_path / "store")

        process = subprocess.Popen(
            [sys.executable, "-m", "beacon.server", "--port", "0",
             "--token", TOKEN, "--data", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            port = int(process.stdout.readline().split("=")[1].strip())
            headers = {"Authorization": f"Bearer {TOKEN}"}

            prices = _get_when_ready(port, "/data/prices/CMPA", headers)
            assert prices.status_code == 200
            assert prices.json()["prices"]["index"]

            base = f"http://127.0.0.1:{port}"
            reference = httpx.get(f"{base}/data/reference/CMPA", headers=headers)
            assert reference.status_code == 200
            assert reference.json()["fields"]["NAME"] == "Company A"

            actions = httpx.get(f"{base}/data/corporate-actions/CMPA",
                                headers=headers)
            assert actions.status_code == 200

            coverage = httpx.get(f"{base}/data/coverage", headers=headers)
            assert coverage.status_code == 200

            datasets = {entry["dataset"]: entry
                        for entry in coverage.json()["datasets"]}
            assert datasets["market"]["identifiers"] == 16
            assert datasets["reference"]["configured"] is True
        finally:
            process.terminate()
            process.wait(timeout=30)


def _get_when_ready(port: int,
                    path: str,
                    headers: dict[str, str],
                    timeout: float = 60.0):
    """Poll until the spawned server answers, then return the response."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            return httpx.get(f"http://127.0.0.1:{port}{path}",
                             headers=headers, timeout=5.0)
        except httpx.TransportError:
            time.sleep(0.1)

    raise AssertionError(f"server never answered on port {port}")
