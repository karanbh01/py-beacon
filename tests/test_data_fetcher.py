# tests/test_data_fetcher.py
"""Tests for DataFetcher's auxiliary accessors: shares outstanding, free-float,
and FX rates — the single home for these series, sourced from market data."""
import pandas as pd
import pytest

from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher


def _make_fetcher(include_aux=True):
    base = {"IDENTIFIER": [], "DATE": [], "CLOSE": []}
    rows = [
        {"IDENTIFIER": "AAA", "DATE": "2024-01-02", "CLOSE": 100.0,
         "SHARES_OUTSTANDING": 1000.0, "FREE_FLOAT": 0.8, "RATE": float("nan")},
        {"IDENTIFIER": "AAA", "DATE": "2024-01-03", "CLOSE": 101.0,
         "SHARES_OUTSTANDING": 1000.0, "FREE_FLOAT": 0.8, "RATE": float("nan")},
        # FX pair stored as its own identifier
        {"IDENTIFIER": "GBPUSD", "DATE": "2024-01-02", "CLOSE": float("nan"),
         "SHARES_OUTSTANDING": float("nan"), "FREE_FLOAT": float("nan"), "RATE": 1.25},
        {"IDENTIFIER": "GBPUSD", "DATE": "2024-01-03", "CLOSE": float("nan"),
         "SHARES_OUTSTANDING": float("nan"), "FREE_FLOAT": float("nan"), "RATE": 1.26},
    ]
    df = pd.DataFrame(rows)
    if not include_aux:
        df = df[["IDENTIFIER", "DATE", "CLOSE"]]
    return DataFetcher(MarketData.from_dataframe(df))


class TestSharesOutstanding:

    def test_returns_value(self):
        assert _make_fetcher().fetch_shares_outstanding("AAA", "2024-01-02") == 1000.0

    def test_none_when_column_absent(self):
        assert _make_fetcher(include_aux=False).fetch_shares_outstanding("AAA", "2024-01-02") is None

    def test_none_when_identifier_absent(self):
        assert _make_fetcher().fetch_shares_outstanding("ZZZ", "2024-01-02") is None

    def test_none_when_no_row_for_date(self):
        assert _make_fetcher().fetch_shares_outstanding("AAA", "2024-06-01") is None


class TestFreeFloatFactor:

    def test_returns_value(self):
        assert _make_fetcher().fetch_free_float_factor("AAA", "2024-01-02") == pytest.approx(0.8)

    def test_none_when_column_absent(self):
        assert _make_fetcher(include_aux=False).fetch_free_float_factor("AAA", "2024-01-02") is None


class TestFxRates:

    def test_returns_rate_series_for_pair(self):
        fx = _make_fetcher().fetch_fx_rates("GBP", "USD", "2024-01-02", "2024-01-03")
        assert isinstance(fx, pd.Series)
        assert list(fx) == pytest.approx([1.25, 1.26])
        assert fx.iloc[0] == pytest.approx(1.25)

    def test_empty_when_pair_absent(self):
        fx = _make_fetcher().fetch_fx_rates("EUR", "USD")
        assert isinstance(fx, pd.Series)
        assert fx.empty
