# tests/test_asset.py
"""Unit tests for the asset module."""
import pytest

from beacon.asset.base import Asset
from beacon.asset.equity import Equity
from beacon.asset.bond import Bond
from beacon.asset.commodity import Commodity


# ── Asset (Base Class) ─────────────────────────────────────────────────


class TestAsset:
    def test_valid_construction(self):
        a = Asset(name="Test", currency="USD", asset_id="T1", asset_type="EQUITY")
        assert a.name == "Test"
        assert a.currency == "USD"
        assert a.asset_id == "T1"
        assert a.asset_type == "EQUITY"

    @pytest.mark.parametrize("field", ["asset_id", "asset_type", "name", "currency"])
    def test_empty_field_raises(self, field):
        kwargs = dict(name="N", currency="C", asset_id="ID", asset_type="T")
        kwargs[field] = ""
        with pytest.raises(ValueError, match=f"{field} cannot be empty"):
            Asset(**kwargs)

    def test_frozen(self):
        a = Asset(name="N", currency="C", asset_id="ID", asset_type="T")
        with pytest.raises(AttributeError):
            a.name = "X"

    def test_equality(self):
        a1 = Asset(name="N", currency="C", asset_id="ID", asset_type="T")
        a2 = Asset(name="N", currency="C", asset_id="ID", asset_type="T")
        assert a1 == a2

    def test_inequality(self):
        a1 = Asset(name="N", currency="C", asset_id="ID1", asset_type="T")
        a2 = Asset(name="N", currency="C", asset_id="ID2", asset_type="T")
        assert a1 != a2

    def test_hash_equal_objects(self):
        a1 = Asset(name="N", currency="C", asset_id="ID", asset_type="T")
        a2 = Asset(name="N", currency="C", asset_id="ID", asset_type="T")
        assert hash(a1) == hash(a2)

    def test_usable_in_set_and_dict(self):
        a = Asset(name="N", currency="C", asset_id="ID", asset_type="T")
        s = {a}
        assert a in s
        d = {a: 1.0}
        assert d[a] == 1.0


# ── Equity ─────────────────────────────────────────────────────────────


class TestEquity:
    def test_full_construction(self):
        eq = Equity(
            name="Apple Inc", currency="USD", ticker="AAPL",
            exchange="NASDAQ", isin="US0378331005", sector="Technology",
            country="US", asset_id="AAPL_US", asset_type="EQUITY",
        )
        assert eq.ticker == "AAPL"
        assert eq.exchange == "NASDAQ"
        assert eq.isin == "US0378331005"
        assert eq.sector == "Technology"
        assert eq.country == "US"
        assert eq.asset_id == "AAPL_US"
        assert eq.asset_type == "EQUITY"

    def test_asset_type_defaults_to_equity(self):
        eq = Equity(name="Apple", currency="USD", ticker="AAPL", exchange="NASDAQ")
        assert eq.asset_type == "EQUITY"

    def test_asset_id_defaults_to_ticker(self):
        eq = Equity(name="Apple", currency="USD", ticker="AAPL", exchange="NASDAQ")
        assert eq.asset_id == "AAPL"

    def test_optional_fields_default_none(self):
        eq = Equity(name="Apple", currency="USD", ticker="AAPL", exchange="NASDAQ")
        assert eq.isin is None
        assert eq.sector is None
        assert eq.country is None

    def test_empty_ticker_raises(self):
        with pytest.raises(ValueError, match="ticker cannot be empty"):
            Equity(name="N", currency="C", ticker="", exchange="E")

    def test_empty_exchange_raises(self):
        with pytest.raises(ValueError, match="exchange cannot be empty"):
            Equity(name="N", currency="C", ticker="T", exchange="")

    def test_frozen(self):
        eq = Equity(name="N", currency="C", ticker="T", exchange="E")
        with pytest.raises(AttributeError):
            eq.ticker = "X"

    def test_equality(self):
        eq1 = Equity(name="N", currency="C", ticker="T", exchange="E")
        eq2 = Equity(name="N", currency="C", ticker="T", exchange="E")
        assert eq1 == eq2
        assert hash(eq1) == hash(eq2)

    def test_is_asset(self):
        eq = Equity(name="N", currency="C", ticker="T", exchange="E")
        assert isinstance(eq, Asset)


# ── Bond ───────────────────────────────────────────────────────────────


class TestBond:
    def test_valid_construction(self):
        b = Bond(
            name="US Treasury 10Y", currency="USD", asset_id="UST10Y",
            coupon=4.5, maturity_date="2035-03-01", issuer="US Treasury",
            credit_rating="AAA", face_value=1000.0,
        )
        assert b.coupon == 4.5
        assert b.maturity_date == "2035-03-01"
        assert b.issuer == "US Treasury"
        assert b.credit_rating == "AAA"
        assert b.face_value == 1000.0

    def test_asset_type_defaults_to_bond(self):
        b = Bond(
            name="Bond", currency="USD", asset_id="B1",
            coupon=3.0, maturity_date="2030-01-01", issuer="Issuer",
        )
        assert b.asset_type == "BOND"

    def test_empty_maturity_date_raises(self):
        with pytest.raises(ValueError, match="maturity_date cannot be empty"):
            Bond(name="B", currency="C", asset_id="B1", coupon=3.0,
                 maturity_date="", issuer="I")

    def test_empty_issuer_raises(self):
        with pytest.raises(ValueError, match="issuer cannot be empty"):
            Bond(name="B", currency="C", asset_id="B1", coupon=3.0,
                 maturity_date="2030-01-01", issuer="")

    def test_optional_credit_rating_default_none(self):
        b = Bond(name="B", currency="C", asset_id="B1", coupon=3.0,
                 maturity_date="2030-01-01", issuer="I")
        assert b.credit_rating is None

    def test_frozen(self):
        b = Bond(name="B", currency="C", asset_id="B1", coupon=3.0,
                 maturity_date="2030-01-01", issuer="I")
        with pytest.raises(AttributeError):
            b.coupon = 5.0

    def test_is_asset(self):
        b = Bond(name="B", currency="C", asset_id="B1", coupon=3.0,
                 maturity_date="2030-01-01", issuer="I")
        assert isinstance(b, Asset)


# ── Commodity ──────────────────────────────────────────────────────────


class TestCommodity:
    def test_valid_construction(self):
        c = Commodity(
            name="Gold", currency="USD", asset_id="GOLD",
            commodity_type="PRECIOUS_METAL", contract_unit="troy_oz",
        )
        assert c.commodity_type == "PRECIOUS_METAL"
        assert c.contract_unit == "troy_oz"

    def test_asset_type_defaults_to_commodity(self):
        c = Commodity(name="Gold", currency="USD", asset_id="GOLD",
                      commodity_type="PRECIOUS_METAL", contract_unit="troy_oz")
        assert c.asset_type == "COMMODITY"

    def test_empty_commodity_type_raises(self):
        with pytest.raises(ValueError, match="commodity_type cannot be empty"):
            Commodity(name="G", currency="C", asset_id="G1",
                      commodity_type="", contract_unit="oz")

    def test_empty_contract_unit_raises(self):
        with pytest.raises(ValueError, match="contract_unit cannot be empty"):
            Commodity(name="G", currency="C", asset_id="G1",
                      commodity_type="METAL", contract_unit="")

    def test_frozen(self):
        c = Commodity(name="G", currency="C", asset_id="G1",
                      commodity_type="METAL", contract_unit="oz")
        with pytest.raises(AttributeError):
            c.commodity_type = "X"

    def test_is_asset(self):
        c = Commodity(name="G", currency="C", asset_id="G1",
                      commodity_type="METAL", contract_unit="oz")
        assert isinstance(c, Asset)
