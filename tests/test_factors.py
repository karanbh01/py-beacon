# tests/test_factors.py
"""BN-93: factor exposures, and the decomposition of active risk."""
import numpy as np
import pandas as pd
import pytest

from beacon.exceptions import CalculationError
from beacon.risk.factors import (
    MARKET_FACTOR,
    fit_factor_model,
    z_scores,
)

ASSETS = [f"S{index}" for index in range(12)]
FACTORS = ["size", "value", "momentum"]
OBSERVATIONS = 600

# The identity TE² = factor + specific is exact by construction, so anything
# above float noise means the algebra is wrong, not that it is approximate.
EXACT = 1e-15


@pytest.fixture
def exposures():
    """Standardised loadings on three factors."""
    generator = np.random.default_rng(3)

    raw = pd.DataFrame(
        {"size": generator.normal(0.0, 1.0, 12) * 3e9,
         "value": generator.normal(0.0, 1.0, 12),
         "momentum": generator.normal(0.0, 1.0, 12)},
        index=ASSETS)

    return z_scores(raw)


@pytest.fixture
def factor_model(exposures):
    """A model fitted to returns that were generated from a real factor structure.

    Built this way so the fit has something genuine to recover: the panel is
    literally B f + ε, so a correct estimator gets the factor returns back and
    an incorrect one has nowhere to hide.
    """
    generator = np.random.default_rng(3)
    loadings = np.column_stack([np.ones(12), exposures.to_numpy()])

    factor_returns = generator.normal(0.0, 0.008, (OBSERVATIONS, 4))
    noise = generator.normal(0.0, 0.004, (OBSERVATIONS, 12))

    panel = pd.DataFrame(
        factor_returns @ loadings.T + noise,
        index=pd.bdate_range("2023-01-01", periods=OBSERVATIONS),
        columns=ASSETS)

    return fit_factor_model(panel, exposures)


@pytest.fixture
def benchmark():
    return dict.fromkeys(ASSETS, 1.0 / 12.0)


@pytest.fixture
def portfolio(benchmark):
    """An active position with real factor bets in it."""
    weights = dict(benchmark)
    weights["S0"] += 0.06
    weights["S3"] += 0.04
    weights["S7"] -= 0.05
    weights["S9"] -= 0.05

    return weights


class TestZScores:

    def test_each_factor_is_centred(self,
                                    exposures):
        assert exposures.mean().abs().max() < 1e-12

    def test_each_factor_has_unit_spread(self,
                                         exposures):
        assert (exposures.std(ddof=0) - 1.0).abs().max() < 1e-12

    def test_the_shape_is_preserved(self,
                                    exposures):
        assert list(exposures.columns) == FACTORS
        assert list(exposures.index) == ASSETS

    def test_a_factor_with_no_spread_becomes_zero(self):
        """It cannot distinguish between assets, so it carries no information."""
        raw = pd.DataFrame({"flat": [2.0] * 5, "real": [1.0, 2.0, 3.0, 4.0, 5.0]},
                           index=list("ABCDE"))

        scaled = z_scores(raw)

        assert (scaled["flat"] == 0.0).all()
        assert scaled["real"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)

    def test_units_do_not_matter(self,
                                 exposures):
        """A market cap in dollars and the same in billions must score alike."""
        raw = pd.DataFrame({"size": [1e9, 2e9, 3e9]}, index=list("ABC"))
        rescaled = raw / 1e9

        pd.testing.assert_frame_equal(z_scores(raw), z_scores(rescaled))

    def test_benchmark_centring_makes_the_benchmark_score_zero(self):
        """The point of centring on weights: an index holder takes no bets."""
        raw = pd.DataFrame({"size": [1.0, 2.0, 6.0]}, index=list("ABC"))
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        scaled = z_scores(raw, weights=weights)
        exposure = sum(weights[asset] * scaled.loc[asset, "size"] for asset in weights)

        assert exposure == pytest.approx(0.0, abs=1e-12)

    def test_equal_weight_centring_matches_the_plain_mean(self):
        raw = pd.DataFrame({"size": [1.0, 2.0, 6.0]}, index=list("ABC"))
        equal = dict.fromkeys("ABC", 1.0 / 3.0)

        pd.testing.assert_frame_equal(z_scores(raw), z_scores(raw, weights=equal))

    def test_centring_weights_summing_to_zero_are_refused(self):
        raw = pd.DataFrame({"size": [1.0, 2.0, 6.0]}, index=list("ABC"))

        with pytest.raises(CalculationError, match="sum to zero"):
            z_scores(raw, weights={"A": 1.0, "B": -1.0, "C": 0.0})

    def test_empty_exposures_are_refused(self):
        with pytest.raises(CalculationError, match="no exposures"):
            z_scores(pd.DataFrame())


class TestFitting:

    def test_the_market_factor_is_added(self,
                                        factor_model):
        assert factor_model.factor_names == [MARKET_FACTOR, *FACTORS]

    def test_the_market_factor_can_be_left_out(self,
                                               exposures):
        generator = np.random.default_rng(5)
        panel = pd.DataFrame(
            generator.normal(0.0, 0.01, (100, 12)),
            index=pd.bdate_range("2024-01-01", periods=100),
            columns=ASSETS)

        model = fit_factor_model(panel, exposures, include_market=False)

        assert model.factor_names == FACTORS

    def test_it_recovers_a_known_factor_structure(self,
                                                  exposures):
        """The strongest check available: generate from known factors, refit.

        The fitted factor returns should track the ones the panel was built
        from, period by period. A wrong regression would still produce numbers
        of the right shape, but they would not correlate.
        """
        generator = np.random.default_rng(19)
        loadings = np.column_stack([np.ones(12), exposures.to_numpy()])
        truth = generator.normal(0.0, 0.008, (400, 4))

        panel = pd.DataFrame(
            truth @ loadings.T + generator.normal(0.0, 0.001, (400, 12)),
            index=pd.bdate_range("2023-01-01", periods=400),
            columns=ASSETS)

        model = fit_factor_model(panel, exposures)

        for index, name in enumerate(model.factor_names):
            correlation = np.corrcoef(truth[:, index],
                                      model.factor_returns[name].to_numpy())[0, 1]
            assert correlation > 0.99, f"{name} was not recovered"

    def test_a_real_structure_gives_a_high_r_squared(self,
                                                     factor_model):
        assert factor_model.r_squared > 0.9

    def test_noise_alone_lands_on_the_overfitting_floor(self,
                                                        exposures):
        """R² does not go to zero on pure noise, and that is not a bug.

        Four factors fitted to a twelve-asset cross-section explain about k/n
        of the variance by construction — four free parameters will always fit
        something. The floor here is 4/12 ≈ 0.33, so an R² is only evidence of
        structure when it clears that, and this is what a caller has to compare
        the 0.96 of the structured fixture against.
        """
        generator = np.random.default_rng(23)
        panel = pd.DataFrame(
            generator.normal(0.0, 0.01, (400, 12)),
            index=pd.bdate_range("2023-01-01", periods=400),
            columns=ASSETS)

        model = fit_factor_model(panel, exposures)
        floor = len(model.factor_names) / len(ASSETS)

        assert model.r_squared == pytest.approx(floor, abs=0.05)

    def test_the_factor_covariance_is_square_and_symmetric(self,
                                                            factor_model):
        matrix = factor_model.factor_covariance.to_numpy()

        assert matrix.shape == (4, 4)
        assert np.allclose(matrix, matrix.T)

    def test_specific_variances_are_positive(self,
                                             factor_model):
        assert (factor_model.specific_variance > 0).all()

    def test_a_panel_missing_an_asset_is_refused(self,
                                                  exposures):
        generator = np.random.default_rng(7)
        panel = pd.DataFrame(
            generator.normal(0.0, 0.01, (100, 11)),
            index=pd.bdate_range("2024-01-01", periods=100),
            columns=ASSETS[:11])

        with pytest.raises(CalculationError, match="does not cover"):
            fit_factor_model(panel, exposures)

    def test_too_few_observations_are_refused(self,
                                              exposures):
        panel = pd.DataFrame(
            np.zeros((1, 12)),
            index=pd.bdate_range("2024-01-01", periods=1),
            columns=ASSETS)

        with pytest.raises(CalculationError, match="at least 2"):
            fit_factor_model(panel, exposures)


class TestImpliedCovariance:

    def test_it_is_symmetric(self,
                             factor_model):
        matrix = factor_model.covariance().to_numpy()

        assert np.allclose(matrix, matrix.T)

    def test_it_is_positive_semi_definite(self,
                                          factor_model):
        """BFBᵀ is PSD and D is a positive diagonal, so the sum must be."""
        eigenvalues = np.linalg.eigvalsh(factor_model.covariance().to_numpy())

        assert eigenvalues.min() > -1e-12

    def test_its_diagonal_exceeds_the_specific_variances(self,
                                                          factor_model):
        """Total variance is common plus specific, so it cannot be less."""
        diagonal = np.diag(factor_model.covariance().to_numpy())

        assert (diagonal >= factor_model.specific_variance.to_numpy() - 1e-15).all()


class TestActiveRiskDecomposition:

    def test_the_identity_holds(self,
                                factor_model,
                                portfolio,
                                benchmark):
        """The acceptance criterion: factor + specific = total active risk².

        Exact rather than approximate, because Σ is *defined* as BFBᵀ + D. Take
        an arbitrary covariance and arbitrary exposures and there is a cross
        term; the identity belongs to the factor model, not to any pairing of a
        matrix with some loadings.
        """
        decomposition = factor_model.decompose_active_risk(portfolio, benchmark)

        assert decomposition.residual == pytest.approx(0.0, abs=EXACT)
        assert decomposition.reconciles()

    def test_the_total_matches_the_implied_covariance_directly(self,
                                                                factor_model,
                                                                portfolio,
                                                                benchmark):
        """Recomputed as aᵀΣa from the assembled matrix, not from the parts."""
        active = np.array([portfolio[asset] - benchmark[asset] for asset in ASSETS])
        covariance = factor_model.covariance().to_numpy()

        decomposition = factor_model.decompose_active_risk(portfolio, benchmark)

        assert decomposition.total_variance == pytest.approx(
            float(active @ covariance @ active), rel=1e-12)

    def test_the_factor_contributions_sum_to_the_factor_variance(self,
                                                                  factor_model,
                                                                  portfolio,
                                                                  benchmark):
        decomposition = factor_model.decompose_active_risk(portfolio, benchmark)

        assert decomposition.factor_contributions.sum() == pytest.approx(
            decomposition.factor_variance, rel=1e-12)

    def test_tracking_error_is_the_square_root_of_the_total(self,
                                                             factor_model,
                                                             portfolio,
                                                             benchmark):
        decomposition = factor_model.decompose_active_risk(portfolio, benchmark)

        assert decomposition.tracking_error == pytest.approx(
            np.sqrt(decomposition.total_variance), rel=1e-12)

    def test_holding_the_benchmark_carries_no_active_risk(self,
                                                           factor_model,
                                                           benchmark):
        decomposition = factor_model.decompose_active_risk(benchmark, benchmark)

        assert decomposition.total_variance == pytest.approx(0.0, abs=EXACT)
        assert decomposition.factor_share == 0.0

    def test_a_strong_factor_structure_puts_most_risk_in_the_factors(self,
                                                                      factor_model,
                                                                      portfolio,
                                                                      benchmark):
        decomposition = factor_model.decompose_active_risk(portfolio, benchmark)

        assert 0.0 < decomposition.factor_share < 1.0
        assert decomposition.factor_share > 0.5

    def test_the_identity_holds_for_a_purely_specific_bet(self,
                                                           factor_model,
                                                           benchmark):
        """Two assets swapped; whatever the factor part is, the split must add up."""
        weights = dict(benchmark)
        weights["S1"] += 0.10
        weights["S2"] -= 0.10

        decomposition = factor_model.decompose_active_risk(weights, benchmark)

        assert decomposition.residual == pytest.approx(0.0, abs=EXACT)
        assert decomposition.specific_variance > 0.0

    def test_both_parts_are_non_negative(self,
                                         factor_model,
                                         portfolio,
                                         benchmark):
        """F is PSD and D is positive, so neither part can come out negative."""
        decomposition = factor_model.decompose_active_risk(portfolio, benchmark)

        assert decomposition.factor_variance >= 0.0
        assert decomposition.specific_variance >= 0.0

    def test_the_frame_is_ordered_by_absolute_contribution(self,
                                                            factor_model,
                                                            portfolio,
                                                            benchmark):
        frame = factor_model.decompose_active_risk(portfolio, benchmark).to_frame()
        magnitudes = frame["contribution"].abs().tolist()

        assert magnitudes == sorted(magnitudes, reverse=True)


class TestExposures:

    def test_the_benchmark_has_near_zero_factor_exposures(self,
                                                           factor_model,
                                                           benchmark):
        """Equal weights over z-scored loadings: the tilts cancel."""
        exposures = factor_model.portfolio_exposures(benchmark)

        for name in FACTORS:
            assert abs(exposures[name]) < 1e-12

    def test_the_market_exposure_of_a_full_portfolio_is_one(self,
                                                             factor_model,
                                                             benchmark):
        """The intercept column is all ones, so it measures the amount invested."""
        exposures = factor_model.portfolio_exposures(benchmark)

        assert exposures[MARKET_FACTOR] == pytest.approx(1.0, abs=1e-12)

    def test_active_exposures_carry_no_market_bet(self,
                                                   factor_model,
                                                   portfolio,
                                                   benchmark):
        """Both sides are fully invested, so the intercept nets out."""
        active = factor_model.active_exposures(portfolio, benchmark)

        assert active[MARKET_FACTOR] == pytest.approx(0.0, abs=1e-12)

    def test_a_deliberate_tilt_shows_up_in_the_right_factor(self,
                                                             factor_model,
                                                             exposures,
                                                             benchmark):
        """Overweight the highest-momentum name; momentum exposure must rise."""
        top = exposures["momentum"].idxmax()
        weights = dict(benchmark)
        weights[top] += 0.10

        active = factor_model.active_exposures(weights, benchmark)

        assert active["momentum"] > 0.0

    def test_absent_assets_count_as_unheld(self,
                                           factor_model):
        exposures = factor_model.portfolio_exposures({"S0": 1.0})

        assert exposures[MARKET_FACTOR] == pytest.approx(1.0, abs=1e-12)

    def test_an_unknown_asset_is_refused(self,
                                         factor_model):
        """Dropping it would understate the position, the wrong way to be wrong."""
        with pytest.raises(CalculationError, match="absent from the factor model"):
            factor_model.portfolio_exposures({"NOT_COVERED": 1.0})


class TestEdgeCases:

    def test_an_existing_market_column_is_not_duplicated(self,
                                                          exposures):
        """A caller who supplied their own intercept keeps it."""
        with_market = exposures.copy()
        with_market.insert(0, MARKET_FACTOR, 1.0)

        generator = np.random.default_rng(13)
        panel = pd.DataFrame(
            generator.normal(0.0, 0.01, (100, 12)),
            index=pd.bdate_range("2024-01-01", periods=100),
            columns=ASSETS)

        model = fit_factor_model(panel, with_market)

        assert model.factor_names.count(MARKET_FACTOR) == 1
        assert model.factor_names == [MARKET_FACTOR, *FACTORS]

    def test_a_panel_with_no_variance_reports_no_explanatory_power(self,
                                                                    exposures):
        """Nothing to explain, so the honest R² is zero rather than one."""
        panel = pd.DataFrame(
            np.zeros((50, 12)),
            index=pd.bdate_range("2024-01-01", periods=50),
            columns=ASSETS)

        model = fit_factor_model(panel, exposures)

        assert model.r_squared == 0.0
