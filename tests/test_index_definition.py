# tests/test_index_definition.py
"""Unit tests for IndexDefinition."""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from beacon.index.constructor import IndexDefinition


@pytest.fixture
def mock_eligibility_rule():
    return MagicMock()


@pytest.fixture
def mock_weighting_scheme():
    return MagicMock()


@pytest.fixture
def default_kwargs(mock_eligibility_rule, mock_weighting_scheme):
    return dict(
        index_id="TEST_IDX",
        index_name="Test Index",
        base_date="2020-01-01",
        base_value=1000.0,
        currency="usd",
        eligibility_rules=[mock_eligibility_rule],
        weighting_scheme=mock_weighting_scheme,
        rebalancing_frequency="QUARTERLY",
    )


# ── Construction & Validation ──────────────────────────────────────────


class TestConstruction:
    def test_valid_instantiation(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)
        assert idx.index_id == "TEST_IDX"
        assert idx.index_name == "Test Index"
        assert idx.base_date == pd.Timestamp("2020-01-01")
        assert idx.base_value == 1000.0
        assert idx.currency == "USD"
        assert idx.rebalancing_frequency == "QUARTERLY"
        assert idx.description is None
        assert idx.universe_identifiers is None

    def test_valid_with_optional_params(self, default_kwargs):
        idx = IndexDefinition(
            **default_kwargs,
            description="A test index",
            universe_identifiers=["AAPL", "MSFT"],
        )
        assert idx.description == "A test index"
        assert idx.universe_identifiers == ["AAPL", "MSFT"]

    def test_currency_uppercased(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)
        assert idx.currency == "USD"

    def test_rebalancing_frequency_uppercased(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "monthly"
        idx = IndexDefinition(**default_kwargs)
        assert idx.rebalancing_frequency == "MONTHLY"

    def test_empty_index_id_raises(self, default_kwargs):
        default_kwargs["index_id"] = ""
        with pytest.raises(ValueError, match="index_id cannot be empty"):
            IndexDefinition(**default_kwargs)

    def test_empty_index_name_raises(self, default_kwargs):
        default_kwargs["index_name"] = ""
        with pytest.raises(ValueError, match="index_name cannot be empty"):
            IndexDefinition(**default_kwargs)

    def test_empty_base_date_raises(self, default_kwargs):
        default_kwargs["base_date"] = ""
        with pytest.raises(ValueError, match="base_date cannot be empty"):
            IndexDefinition(**default_kwargs)

    def test_zero_base_value_raises(self, default_kwargs):
        default_kwargs["base_value"] = 0
        with pytest.raises(ValueError, match="base_value must be positive"):
            IndexDefinition(**default_kwargs)

    def test_negative_base_value_raises(self, default_kwargs):
        default_kwargs["base_value"] = -100
        with pytest.raises(ValueError, match="base_value must be positive"):
            IndexDefinition(**default_kwargs)

    def test_empty_currency_raises(self, default_kwargs):
        default_kwargs["currency"] = ""
        with pytest.raises(ValueError, match="currency cannot be empty"):
            IndexDefinition(**default_kwargs)

    def test_empty_rebalancing_frequency_raises(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = ""
        with pytest.raises(ValueError, match="rebalancing_frequency cannot be empty"):
            IndexDefinition(**default_kwargs)

    def test_none_weighting_scheme_raises(self, default_kwargs):
        default_kwargs["weighting_scheme"] = None
        with pytest.raises(ValueError, match="weighting_scheme must be provided"):
            IndexDefinition(**default_kwargs)

    def test_empty_eligibility_rules_logs_warning(self, default_kwargs, caplog):
        default_kwargs["eligibility_rules"] = []
        with caplog.at_level("WARNING"):
            IndexDefinition(**default_kwargs)
        assert "defined with no eligibility rules" in caplog.text


# ── Universe Identifiers ───────────────────────────────────────────────


class TestUniverseIdentifiers:
    def test_default_is_none(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)
        assert idx.universe_identifiers is None

    def test_valid_list(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs, universe_identifiers=["AAPL", "GOOG"])
        assert idx.universe_identifiers == ["AAPL", "GOOG"]

    def test_empty_list_raises(self, default_kwargs):
        with pytest.raises(ValueError, match="universe_identifiers.*must be a non-empty list"):
            IndexDefinition(**default_kwargs, universe_identifiers=[])


# ── Rebalance Dates ────────────────────────────────────────────────────


class TestGetRebalanceDates:
    def test_monthly(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "MONTHLY"
        idx = IndexDefinition(**default_kwargs)
        dates = idx.get_rebalance_dates("2025-01-01", "2025-06-30")
        assert len(dates) == 6
        months = [d.month for d in dates]
        assert months == [1, 2, 3, 4, 5, 6]

    def test_quarterly(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)  # QUARTERLY
        dates = idx.get_rebalance_dates("2025-01-01", "2025-12-31")
        assert len(dates) == 4
        months = [d.month for d in dates]
        assert months == [1, 4, 7, 10]

    def test_semi_annual(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "SEMI-ANNUAL"
        idx = IndexDefinition(**default_kwargs)
        dates = idx.get_rebalance_dates("2025-01-01", "2025-12-31")
        assert len(dates) == 2
        months = [d.month for d in dates]
        assert months == [1, 7]

    def test_annual(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "ANNUAL"
        idx = IndexDefinition(**default_kwargs)
        dates = idx.get_rebalance_dates("2025-01-01", "2026-12-31")
        assert len(dates) == 2
        assert dates[0].year == 2025
        assert dates[1].year == 2026

    def test_dates_are_business_days(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "MONTHLY"
        idx = IndexDefinition(**default_kwargs)
        dates = idx.get_rebalance_dates("2025-01-01", "2025-12-31")
        for d in dates:
            assert d.dayofweek < 5, f"{d} is a weekend"

    def test_dates_are_sorted(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)
        dates = idx.get_rebalance_dates("2024-01-01", "2026-12-31")
        assert dates == sorted(dates)

    def test_unsupported_frequency_raises(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "WEEKLY"
        idx = IndexDefinition(**default_kwargs)
        with pytest.raises(ValueError, match="Unsupported rebalancing frequency"):
            idx.get_rebalance_dates("2025-01-01", "2025-12-31")

    def test_same_start_end_date_on_business_day(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "MONTHLY"
        idx = IndexDefinition(**default_kwargs)
        # 2025-01-01 is a Wednesday and the first business day of Jan 2025
        dates = idx.get_rebalance_dates("2025-01-01", "2025-01-01")
        assert len(dates) == 1

    def test_no_dates_in_narrow_range(self, default_kwargs):
        default_kwargs["rebalancing_frequency"] = "ANNUAL"
        idx = IndexDefinition(**default_kwargs)
        # Mid-month range that won't contain a first-business-day-of-month
        dates = idx.get_rebalance_dates("2025-03-10", "2025-03-20")
        assert len(dates) == 0

    def test_multi_year_range(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)  # QUARTERLY
        dates = idx.get_rebalance_dates("2023-01-01", "2025-12-31")
        assert len(dates) >= 12  # At least 4 per year * 3 years


# ── __repr__ ───────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_core_fields(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)
        r = repr(idx)
        assert "TEST_IDX" in r
        assert "Test Index" in r
        assert "2020-01-01" in r
        assert "1000.0" in r
        assert "USD" in r
        assert "QUARTERLY" in r

    def test_repr_shows_universe_size(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs, universe_identifiers=["A", "B", "C"])
        assert "universe_size=3" in repr(idx)

    def test_repr_shows_zero_universe_when_none(self, default_kwargs):
        idx = IndexDefinition(**default_kwargs)
        assert "universe_size=0" in repr(idx)
