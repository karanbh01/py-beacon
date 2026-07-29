# tests/test_derivatives_base.py
"""Unit tests for beacon.derivatives.base.DerivativeBase."""
import pandas as pd
import pytest

from beacon.derivatives.base import DerivativeBase

# ---------------------------------------------------------------------------
# Minimal concrete subclass for exercising the base class
# ---------------------------------------------------------------------------

class _ConcreteDerivative(DerivativeBase):
    def fair_value(self,
                   spot_price,
                   valuation_date,
                   market_data):
        return spot_price * self.notional

    def mark_to_market(self,
                       market_price,
                       spot_price,
                       valuation_date,
                       market_data):
        fv = self.fair_value(spot_price, valuation_date, market_data)
        return {"fair_value": fv, "pnl": (market_price - spot_price) * self.notional}


def _make(**overrides):
    kwargs = {
        "derivative_id": "D1",
        "underlying_id": "SPX",
        "underlying_type": "INDEX",
        "currency": "usd",
        "expiry_date": "2025-01-01",
        "notional": 1_000_000.0,
    }
    kwargs.update(overrides)
    return _ConcreteDerivative(**kwargs)


# ---------------------------------------------------------------------------
# Abstract behaviour
# ---------------------------------------------------------------------------

class TestAbstract:

    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            DerivativeBase("D1", "SPX", "INDEX", "USD", "2025-01-01", 1.0)

    def test_subclass_missing_methods_cannot_instantiate(self):
        class Incomplete(DerivativeBase):
            def fair_value(self,
                           spot_price,
                           valuation_date,
                           market_data):
                return 0.0
            # mark_to_market not implemented

        with pytest.raises(TypeError):
            Incomplete("D1", "SPX", "INDEX", "USD", "2025-01-01", 1.0)

    def test_concrete_subclass_instantiates(self):
        d = _make()
        assert isinstance(d, DerivativeBase)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_stores_fields(self):
        d = _make()
        assert d.derivative_id == "D1"
        assert d.underlying_id == "SPX"
        assert d.underlying_type == "INDEX"
        assert d.notional == 1_000_000.0

    def test_currency_uppercased(self):
        assert _make(currency="usd").currency == "USD"

    def test_underlying_type_uppercased(self):
        assert _make(underlying_type="etf").underlying_type == "ETF"

    def test_expiry_parsed_to_timestamp(self):
        d = _make(expiry_date="2025-06-30")
        assert d.expiry_date == pd.Timestamp("2025-06-30")

    @pytest.mark.parametrize("field", [
        "derivative_id", "underlying_id", "underlying_type", "currency", "expiry_date",
    ])
    def test_empty_required_field_raises(self,
                                         field):
        with pytest.raises(ValueError, match=field):
            _make(**{field: ""})

    def test_invalid_underlying_type_raises(self):
        with pytest.raises(ValueError, match="underlying_type"):
            _make(underlying_type="BOND")

    def test_valid_underlying_types_accepted(self):
        for t in ("INDEX", "ETF", "EQUITY"):
            assert _make(underlying_type=t).underlying_type == t

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError, match="notional"):
            _make(notional=0.0)

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError, match="notional"):
            _make(notional=-100.0)


# ---------------------------------------------------------------------------
# time_to_expiry
# ---------------------------------------------------------------------------

class TestTimeToExpiry:

    def test_one_year_act365(self):
        # 2023-01-01 -> 2024-01-01 is 365 days (2023 not a leap year).
        d = _make(expiry_date="2024-01-01")
        assert d.time_to_expiry(pd.Timestamp("2023-01-01")) == pytest.approx(1.0)

    def test_half_year(self):
        # 2024-01-01 -> 2024-07-01 is 182 days.
        d = _make(expiry_date="2024-07-01")
        expected = 182.0 / 365.0
        assert d.time_to_expiry(pd.Timestamp("2024-01-01")) == pytest.approx(expected)

    def test_at_expiry_is_zero(self):
        d = _make(expiry_date="2025-01-01")
        assert d.time_to_expiry(pd.Timestamp("2025-01-01")) == 0.0

    def test_after_expiry_clamped_to_zero(self):
        d = _make(expiry_date="2025-01-01")
        assert d.time_to_expiry(pd.Timestamp("2025-06-01")) == 0.0

    def test_accepts_string_valuation_date(self):
        d = _make(expiry_date="2024-01-01")
        assert d.time_to_expiry("2023-01-01") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Concrete method wiring
# ---------------------------------------------------------------------------

class TestConcreteMethods:

    def test_fair_value_and_mtm(self):
        d = _make(notional=10.0)
        vd = pd.Timestamp("2024-06-01")
        assert d.fair_value(100.0, vd, {}) == 1000.0
        mtm = d.mark_to_market(105.0, 100.0, vd, {})
        assert mtm["fair_value"] == 1000.0
        assert mtm["pnl"] == pytest.approx(50.0)
