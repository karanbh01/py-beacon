# tests/test_risk_model.py
"""Unit tests for the risk model: covariance, shrinkage, correlation, PSD."""
import numpy as np
import pandas as pd
import pytest

from beacon.exceptions import CalculationError
from beacon.risk import (
    CONSTANT_CORRELATION,
    PERIODS_PER_YEAR,
    SCALED_IDENTITY,
    annualise,
    average_pairwise_correlation,
    condition_number,
    constant_correlation_target,
    correlation_from_covariance,
    eigenvalues,
    estimate_risk_model,
    heuristic_intensity,
    is_positive_semi_definite,
    nearest_positive_semi_definite,
    sample_covariance,
    scaled_identity_target,
    shrink_covariance,
)

ASSETS = ["AAA", "BBB", "CCC", "DDD"]


def returns_panel(observations: int = 500,
                  assets: int = 4,
                  seed: int = 7) -> pd.DataFrame:
    """A deterministic returns panel with genuine cross-correlation.

    Built from a shared market factor plus idiosyncratic noise, so the
    covariance has real off-diagonal structure to estimate rather than being
    diagonal by construction. Seeded, so every run sees the same panel.
    """
    generator = np.random.default_rng(seed)
    market = generator.normal(0.0, 0.01, size=observations)
    columns = {}

    for index in range(assets):
        beta = 0.5 + 0.25 * index
        idiosyncratic = generator.normal(0.0, 0.005, size=observations)
        columns[ASSETS[index] if index < len(ASSETS) else f"A{index}"] = (
            beta * market + idiosyncratic)

    dates = pd.bdate_range("2023-01-02", periods=observations)

    return pd.DataFrame(columns, index=dates)


class TestSampleCovariance:

    def test_matches_pandas(self):
        panel = returns_panel()

        computed = sample_covariance(panel.to_numpy())

        np.testing.assert_allclose(computed, panel.cov().to_numpy(), rtol=1e-12)

    def test_is_symmetric(self):
        covariance = sample_covariance(returns_panel().to_numpy())

        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=0)

    def test_is_positive_semi_definite(self):
        assert is_positive_semi_definite(sample_covariance(returns_panel().to_numpy()))

    def test_singular_when_assets_outnumber_observations(self):
        """The situation shrinkage exists for: PSD but not invertible."""
        panel = returns_panel(observations=3, assets=4)

        covariance = sample_covariance(panel.to_numpy())

        assert is_positive_semi_definite(covariance)
        assert condition_number(covariance) == float("inf")

    def test_rejects_a_single_observation(self):
        with pytest.raises(CalculationError, match="at least 2 observations"):
            sample_covariance(np.array([[0.01, 0.02]]))

    def test_rejects_non_finite_values(self):
        panel = np.array([[0.01, 0.02], [np.nan, 0.01], [0.0, 0.0]])

        with pytest.raises(CalculationError, match="NaN or infinite"):
            sample_covariance(panel)

    def test_rejects_a_one_dimensional_array(self):
        with pytest.raises(CalculationError, match="2-D array"):
            sample_covariance(np.array([0.01, 0.02]))


class TestTargets:

    def test_scaled_identity_is_diagonal_with_average_variance(self):
        covariance = np.array([[4.0, 1.0], [1.0, 2.0]])

        target = scaled_identity_target(covariance)

        np.testing.assert_allclose(target, np.eye(2) * 3.0)

    def test_constant_correlation_preserves_variances(self):
        covariance = sample_covariance(returns_panel().to_numpy())

        target = constant_correlation_target(covariance)

        np.testing.assert_allclose(np.diag(target), np.diag(covariance), rtol=1e-12)

    def test_constant_correlation_has_one_common_correlation(self):
        covariance = sample_covariance(returns_panel().to_numpy())

        correlation = correlation_from_covariance(constant_correlation_target(covariance))
        off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]

        assert off_diagonal.std() == pytest.approx(0.0, abs=1e-12)

    def test_constant_correlation_uses_the_sample_average(self):
        covariance = sample_covariance(returns_panel().to_numpy())
        expected = average_pairwise_correlation(correlation_from_covariance(covariance))

        target = constant_correlation_target(covariance)
        actual = average_pairwise_correlation(correlation_from_covariance(target))

        assert actual == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("builder",
                             [scaled_identity_target, constant_correlation_target])
    def test_targets_are_positive_semi_definite(self,
                                                builder):
        """Both targets must be PSD, or shrinkage could not preserve PSD."""
        covariance = sample_covariance(returns_panel().to_numpy())

        assert is_positive_semi_definite(builder(covariance))


class TestShrinkage:

    def test_zero_intensity_reproduces_the_sample_exactly(self):
        """An acceptance criterion: lambda = 0 changes nothing."""
        sample = sample_covariance(returns_panel().to_numpy())
        target = constant_correlation_target(sample)

        shrunk = shrink_covariance(sample, target, 0.0)

        np.testing.assert_allclose(shrunk, sample, rtol=1e-15, atol=0)

    def test_full_intensity_reproduces_the_target(self):
        sample = sample_covariance(returns_panel().to_numpy())
        target = constant_correlation_target(sample)

        np.testing.assert_allclose(shrink_covariance(sample, target, 1.0), target,
                                   rtol=1e-15, atol=0)

    def test_shrinkage_improves_conditioning(self):
        """The point of the exercise: a less extreme eigenvalue spread."""
        sample = sample_covariance(returns_panel(observations=10, assets=4).to_numpy())
        target = constant_correlation_target(sample)

        shrunk = shrink_covariance(sample, target, 0.5)

        assert condition_number(shrunk) < condition_number(sample)

    def test_rejects_an_out_of_range_intensity(self):
        sample = np.eye(2)

        with pytest.raises(CalculationError, match=r"intensity must be in \[0, 1\]"):
            shrink_covariance(sample, sample, 1.5)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(CalculationError, match="does not match target"):
            shrink_covariance(np.eye(2), np.eye(3), 0.5)

    def test_result_is_symmetric(self):
        sample = sample_covariance(returns_panel().to_numpy())
        shrunk = shrink_covariance(sample, constant_correlation_target(sample), 0.3)

        np.testing.assert_allclose(shrunk, shrunk.T, rtol=0, atol=0)


class TestHeuristicIntensity:

    def test_shrinks_harder_when_assets_outnumber_observations(self):
        few_observations = heuristic_intensity(observations=10, assets=100)
        many_observations = heuristic_intensity(observations=1000, assets=100)

        assert few_observations > many_observations

    def test_is_bounded_to_the_unit_interval(self):
        for observations, assets in [(1, 1000), (1000, 1), (50, 50)]:
            assert 0.0 < heuristic_intensity(observations, assets) < 1.0

    def test_equal_counts_give_a_half(self):
        assert heuristic_intensity(50, 50) == pytest.approx(0.5)

    def test_rejects_non_positive_counts(self):
        with pytest.raises(CalculationError, match="must both be positive"):
            heuristic_intensity(0, 5)


class TestCorrelation:

    def test_diagonal_is_exactly_one(self):
        """An acceptance criterion — and exactly, not approximately."""
        covariance = sample_covariance(returns_panel().to_numpy())

        correlation = correlation_from_covariance(covariance)

        np.testing.assert_array_equal(np.diag(correlation), np.ones(4))

    def test_is_symmetric(self):
        correlation = correlation_from_covariance(
            sample_covariance(returns_panel().to_numpy()))

        np.testing.assert_allclose(correlation, correlation.T, rtol=0, atol=0)

    def test_entries_are_within_the_valid_range(self):
        correlation = correlation_from_covariance(
            sample_covariance(returns_panel().to_numpy()))

        assert correlation.min() >= -1.0 - 1e-12
        assert correlation.max() <= 1.0 + 1e-12

    def test_known_two_asset_case(self):
        covariance = np.array([[4.0, 2.0], [2.0, 9.0]])

        correlation = correlation_from_covariance(covariance)

        # 2 / (2 * 3) = 1/3
        assert correlation[0, 1] == pytest.approx(1 / 3)

    def test_zero_variance_asset_does_not_divide_by_zero(self):
        """A constant series has no variation to correlate with anything."""
        covariance = np.array([[4.0, 0.0], [0.0, 0.0]])

        correlation = correlation_from_covariance(covariance)

        assert np.isfinite(correlation).all()
        assert correlation[0, 1] == 0.0
        assert correlation[1, 1] == 1.0

    def test_subnormal_variance_does_not_escape_the_bounds(self):
        """Regression: Hypothesis found this and it produced |rho| = 1.0485.

        Returns around 1e-162 give a variance around 1e-323 — a subnormal
        float carrying almost no mantissa — so dividing by its square root
        loses catastrophic precision.
        """
        panel = pd.DataFrame({"A0": [0.0, 0.5], "A1": [4.661305e-162, 0.0]})

        model = estimate_risk_model(panel)
        correlation = model.correlation.to_numpy()

        assert correlation.min() >= -1.0
        assert correlation.max() <= 1.0

    def test_a_negligible_variance_asset_correlates_with_nothing(self):
        covariance = np.array([[4.0, 1e-180], [1e-180, 1e-320]])

        correlation = correlation_from_covariance(covariance)

        assert correlation[0, 1] == 0.0
        assert correlation[1, 1] == 1.0

    def test_correlations_are_clipped_to_the_valid_range(self):
        """Accumulated error must never surface as a correlation above 1."""
        covariance = np.array([[1.0, 1.0 + 1e-9], [1.0 + 1e-9, 1.0]])

        correlation = correlation_from_covariance(covariance)

        assert correlation.max() <= 1.0
        assert correlation.min() >= -1.0

    def test_average_pairwise_ignores_the_diagonal(self):
        correlation = np.array([[1.0, 0.5], [0.5, 1.0]])

        assert average_pairwise_correlation(correlation) == pytest.approx(0.5)

    def test_average_pairwise_of_a_single_asset_is_zero(self):
        assert average_pairwise_correlation(np.array([[1.0]])) == 0.0


class TestPsdChecks:

    def test_identity_is_psd(self):
        assert is_positive_semi_definite(np.eye(3))

    def test_a_negative_eigenvalue_is_detected(self):
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues 3 and -1

        assert not is_positive_semi_definite(indefinite)

    def test_the_flag_is_scale_invariant(self):
        """Annualising must not change the answer."""
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])

        assert is_positive_semi_definite(indefinite) is is_positive_semi_definite(
            indefinite * PERIODS_PER_YEAR)

    def test_tiny_negative_eigenvalues_pass_within_tolerance(self):
        """A singular-but-valid matrix has zeros in theory, noise in practice."""
        nearly = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-16]])

        assert is_positive_semi_definite(nearly)

    def test_repair_removes_negative_eigenvalues(self):
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])

        repaired = nearest_positive_semi_definite(indefinite)

        assert is_positive_semi_definite(repaired)
        assert eigenvalues(repaired).min() >= -1e-12

    def test_repair_can_force_positive_definiteness(self):
        """An optimiser inverting the matrix needs strictly positive values."""
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])

        repaired = nearest_positive_semi_definite(singular, minimum_eigenvalue=1e-8)

        assert eigenvalues(repaired).min() >= 1e-9
        assert np.isfinite(condition_number(repaired))

    def test_repair_leaves_an_already_psd_matrix_alone(self):
        covariance = sample_covariance(returns_panel().to_numpy())

        np.testing.assert_allclose(nearest_positive_semi_definite(covariance),
                                   covariance, rtol=1e-12)

    def test_condition_number_of_the_identity_is_one(self):
        assert condition_number(np.eye(4)) == pytest.approx(1.0)

    def test_condition_number_is_infinite_when_singular(self):
        assert condition_number(np.array([[1.0, 1.0], [1.0, 1.0]])) == float("inf")


class TestAnnualisation:

    def test_covariance_scales_linearly(self):
        daily = np.array([[1e-4, 0.0], [0.0, 4e-4]])

        annual = annualise(daily)

        np.testing.assert_allclose(annual, daily * PERIODS_PER_YEAR)

    def test_volatility_scales_with_the_square_root(self):
        daily_variance = 1e-4
        annual = annualise(np.array([[daily_variance]]))

        assert np.sqrt(annual[0, 0]) == pytest.approx(
            np.sqrt(daily_variance) * np.sqrt(PERIODS_PER_YEAR))

    def test_annualisation_does_not_change_correlation(self):
        covariance = sample_covariance(returns_panel().to_numpy())

        np.testing.assert_allclose(correlation_from_covariance(annualise(covariance)),
                                   correlation_from_covariance(covariance),
                                   rtol=1e-12)

    def test_rejects_a_non_positive_factor(self):
        with pytest.raises(CalculationError, match="must be positive"):
            annualise(np.eye(2), periods_per_year=0)


class TestEstimateRiskModel:

    def test_produces_labelled_matrices(self):
        model = estimate_risk_model(returns_panel())

        assert model.asset_ids == ASSETS
        assert list(model.covariance.columns) == ASSETS
        assert list(model.correlation.index) == ASSETS

    def test_correlation_has_a_unit_diagonal(self):
        model = estimate_risk_model(returns_panel())

        np.testing.assert_array_equal(np.diag(model.correlation.to_numpy()),
                                      np.ones(len(ASSETS)))

    def test_diagnostics_are_reported(self):
        model = estimate_risk_model(returns_panel())
        diagnostics = model.diagnostics

        assert diagnostics.observations == 500
        assert diagnostics.assets == len(ASSETS)
        assert diagnostics.target == CONSTANT_CORRELATION
        assert 0.0 < diagnostics.intensity < 1.0
        assert diagnostics.positive_semi_definite is True
        assert diagnostics.repaired is False
        assert diagnostics.condition_number > 1.0

    def test_psd_flag_agrees_with_the_eigenvalues(self):
        """An acceptance criterion: the flag is verified, not asserted."""
        model = estimate_risk_model(returns_panel())

        assert model.diagnostics.smallest_eigenvalue == pytest.approx(
            float(model.eigenvalues()[0]))
        assert model.diagnostics.positive_semi_definite == (
            model.diagnostics.smallest_eigenvalue >= 0.0
            or abs(model.diagnostics.smallest_eigenvalue) < 1e-8)

    def test_zero_intensity_gives_the_annualised_sample_covariance(self):
        panel = returns_panel()

        model = estimate_risk_model(panel, intensity=0.0)

        expected = annualise(sample_covariance(panel.to_numpy()))
        np.testing.assert_allclose(model.covariance.to_numpy(), expected, rtol=1e-12)
        assert model.diagnostics.intensity == 0.0

    @pytest.mark.parametrize("target", [CONSTANT_CORRELATION, SCALED_IDENTITY])
    def test_both_targets_produce_a_psd_model(self,
                                              target):
        model = estimate_risk_model(returns_panel(), target=target)

        assert model.diagnostics.positive_semi_definite

    def test_unknown_target_is_rejected(self):
        with pytest.raises(CalculationError, match="unknown target"):
            estimate_risk_model(returns_panel(), target="nonsense")

    def test_rows_with_missing_values_are_dropped(self):
        panel = returns_panel()
        panel.iloc[5, 0] = np.nan

        model = estimate_risk_model(panel)

        assert model.diagnostics.observations == len(panel) - 1

    def test_an_all_missing_panel_is_rejected(self):
        panel = returns_panel(observations=5)
        panel.iloc[:, 0] = np.nan

        with pytest.raises(CalculationError, match="no complete observations"):
            estimate_risk_model(panel)

    def test_volatilities_are_the_square_root_of_the_diagonal(self):
        model = estimate_risk_model(returns_panel())

        np.testing.assert_allclose(
            model.volatilities().to_numpy(),
            np.sqrt(np.diag(model.covariance.to_numpy())), rtol=1e-12)

    def test_short_panel_still_yields_a_psd_model(self):
        """Fewer observations than assets: the case shrinkage is for."""
        model = estimate_risk_model(returns_panel(observations=3, assets=4))

        assert model.diagnostics.positive_semi_definite


class TestPortfolioRisk:

    def test_variance_of_a_single_asset_is_its_own(self):
        model = estimate_risk_model(returns_panel())
        weights = {"AAA": 1.0}

        assert model.portfolio_variance(weights) == pytest.approx(
            model.covariance.loc["AAA", "AAA"])

    def test_volatility_is_the_square_root_of_the_variance(self):
        model = estimate_risk_model(returns_panel())
        weights = dict.fromkeys(ASSETS, 0.25)

        assert model.portfolio_volatility(weights) == pytest.approx(
            np.sqrt(model.portfolio_variance(weights)))

    def test_diversification_reduces_risk(self):
        """An equal-weight basket must be no riskier than its riskiest name."""
        model = estimate_risk_model(returns_panel())
        basket = model.portfolio_volatility(dict.fromkeys(ASSETS, 0.25))

        assert basket < model.volatilities().max()

    def test_variance_matches_the_explicit_quadratic_form(self):
        model = estimate_risk_model(returns_panel())
        weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}
        vector = np.array([weights[a] for a in model.asset_ids])

        expected = float(vector @ model.covariance.to_numpy() @ vector)

        assert model.portfolio_variance(weights) == pytest.approx(expected)

    def test_missing_assets_count_as_zero_weight(self):
        model = estimate_risk_model(returns_panel())

        assert model.portfolio_variance({"AAA": 1.0}) == pytest.approx(
            model.portfolio_variance({"AAA": 1.0, "BBB": 0.0}))

    def test_unknown_asset_is_an_error_not_a_silent_drop(self):
        """Dropping it would understate the position's risk."""
        model = estimate_risk_model(returns_panel())

        with pytest.raises(CalculationError, match="absent from the risk model"):
            model.portfolio_variance({"ZZZ": 1.0})

    def test_variance_is_never_negative(self):
        model = estimate_risk_model(returns_panel())

        assert model.portfolio_variance(dict.fromkeys(ASSETS, 0.0)) >= 0.0

    def test_tracking_error_is_zero_when_matching_the_benchmark(self):
        model = estimate_risk_model(returns_panel())
        weights = dict.fromkeys(ASSETS, 0.25)

        assert model.tracking_error(weights, weights) == pytest.approx(0.0, abs=1e-12)

    def test_tracking_error_grows_with_the_active_position(self):
        model = estimate_risk_model(returns_panel())
        benchmark = dict.fromkeys(ASSETS, 0.25)
        small = {**benchmark, "AAA": 0.30, "BBB": 0.20}
        large = {**benchmark, "AAA": 0.45, "BBB": 0.05}

        assert (model.tracking_error(large, benchmark)
                > model.tracking_error(small, benchmark))

    def test_tracking_error_handles_a_name_absent_from_the_benchmark(self):
        model = estimate_risk_model(returns_panel())
        benchmark = {"AAA": 0.5, "BBB": 0.5}
        portfolio = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}

        assert model.tracking_error(portfolio, benchmark) > 0.0


class TestDataBinding:

    def test_with_data_returns_self_for_chaining(self):
        from unittest.mock import MagicMock

        model = estimate_risk_model(returns_panel())
        fetcher = MagicMock()

        assert model.with_data(fetcher) is model
        assert model._data_fetcher is fetcher
