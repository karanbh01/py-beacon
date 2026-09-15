# tests/test_chaining.py
"""BN-184: the chained level path refuses rather than substituting.

`chaining.py` reimplements the calculator's arithmetic over identifiers, and
it had reimplemented two of the catalogue of #197 along with it: a name the
schedule allocates to but the data has never priced took zero units, and the
FX conversion never ran at all. The second is not a substitution so much as
its consequence — the branch that would have applied a rate compared a
variable against itself and was therefore always false — so a chained index
over foreign names added prices in different currencies as though every unit
were the same size, which is BN-188's defect on a second path.

These drive `chain_levels` directly: it takes an identity, a parent
calculation and a solved schedule as plain arguments, so none of the
derivation machinery is needed to exercise it.
"""
import pandas as pd
import pytest

from beacon.exceptions import CalculationError
from beacon.index.chaining import chain_levels
from beacon.index.result import IndexResult

DAYS = pd.bdate_range("2024-01-02", periods=4)
BASE = 1000.0


class Fetcher:
    """An in-memory stand-in: prices, currencies and one FX pair."""

    def __init__(self,
                 prices: dict[str, list[float] | None],
                 currencies: dict[str, str] | None = None,
                 rates: list[float] | None = None):
        self.prices = prices
        self.currencies = currencies or {}
        self.rates = rates
        self.reference_error: Exception | None = None

    def fetch_market_data(self,
                          identifier,
                          start_date=None,
                          end_date=None,
                          columns=None):
        series = self.prices.get(identifier)

        if series is None:
            return pd.DataFrame()

        return pd.DataFrame({"CLOSE": series}, index=DAYS)

    def fetch_reference_data(self,
                             identifier,
                             date=None,
                             columns=None):
        if self.reference_error is not None:
            raise self.reference_error

        currency = self.currencies.get(identifier)

        if currency is None:
            return pd.DataFrame()

        return pd.DataFrame({"CURRENCY": [currency]},
                            index=pd.Index([identifier], name="IDENTIFIER"))

    def fetch_fx_rates(self,
                       from_currency,
                       to_currency,
                       start_date=None,
                       end_date=None,
                       column="RATE"):
        if self.rates is None:
            return pd.Series(dtype=float)

        return pd.Series(self.rates, index=DAYS)


def parent() -> IndexResult:
    """A parent calculation whose only role here is to supply the calendar."""
    return IndexResult(index_id="PARENT",
                       index_levels=pd.Series(BASE, index=DAYS),
                       divisor_history=pd.Series(1.0, index=DAYS),
                       constituent_snapshots={},
                       weight_snapshots={})


def chain(solved,
          fetcher) -> IndexResult:
    return chain_levels(index_id="CHAINED",
                        base_value=BASE,
                        currency="USD",
                        parent=parent(),
                        solved=solved,
                        data_provider=fetcher,
                        price_column="CLOSE")


class TestAnUnpricedHoldingIsRefused:
    """A name the schedule allocates to and the data cannot price."""

    def test_it_is_refused_rather_than_held_at_zero(self):
        """Zero units leaves the index short by that name's whole weight.

        The old answer logged a warning and carried on, so the level came out
        low by the missing name's share and was published as the index's own.
        """
        solved = {DAYS[0]: {"AAA": 0.5, "MISSING": 0.5}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "MISSING": None})

        with pytest.raises(CalculationError, match="schedule allocates to it"):
            chain(solved, fetcher)

    def test_the_refusal_names_the_name_and_the_column(self):
        solved = {DAYS[0]: {"MISSING": 1.0}}

        with pytest.raises(CalculationError) as raised:
            chain(solved, Fetcher({"MISSING": None}))

        message = str(raised.value)

        assert "MISSING" in message
        assert "CLOSE" in message

    def test_a_name_never_held_is_tolerated(self):
        """Weight zero throughout is not a holding, so no price is needed.

        The refusal keys on allocation rather than on membership of the
        schedule, so a name carried at zero does not fail a whole calculation
        over prices nothing will ever read.
        """
        solved = {DAYS[0]: {"AAA": 1.0, "SPARE": 0.0}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "SPARE": None})

        result = chain(solved, fetcher)

        assert result.index_levels.iloc[0] == pytest.approx(BASE)


class TestForeignPricesAreConverted:
    """The FX branch that never ran (BN-184)."""

    def test_a_moving_rate_moves_the_level(self):
        """A flat GBP price under a doubling rate doubles a USD index.

        This is the assertion the bug could not pass: with the conversion
        skipped the level stayed at its base value all four days, because the
        only thing moving was the rate.
        """
        solved = {DAYS[0]: {"BP": 1.0}}
        fetcher = Fetcher({"BP": [100.0] * 4},
                          currencies={"BP": "GBP"},
                          rates=[1.0, 1.0, 2.0, 2.0])

        levels = chain(solved, fetcher).index_levels

        assert levels.iloc[0] == pytest.approx(BASE)
        assert levels.iloc[1] == pytest.approx(BASE)
        assert levels.iloc[2] == pytest.approx(2 * BASE)
        assert levels.iloc[3] == pytest.approx(2 * BASE)

    def test_a_name_in_the_index_currency_is_untouched(self):
        """The conversion did not start firing where it should not."""
        solved = {DAYS[0]: {"AAA": 1.0}}
        fetcher = Fetcher({"AAA": [100.0, 110.0, 110.0, 121.0]},
                          currencies={"AAA": "USD"},
                          rates=[7.0] * 4)

        levels = chain(solved, fetcher).index_levels

        assert levels.iloc[1] == pytest.approx(1.1 * BASE)
        assert levels.iloc[3] == pytest.approx(1.21 * BASE)

    def test_each_name_is_converted_at_its_own_rate(self):
        """The rebinding also leaked one name's currency into the next.

        Two names in two currencies is the case that exposed it: whichever
        sorted last was compared against the previous name's currency rather
        than against the index's.
        """
        solved = {DAYS[0]: {"AAA": 0.5, "BP": 0.5}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "BP": [100.0] * 4},
                          currencies={"AAA": "USD", "BP": "GBP"},
                          rates=[1.0, 1.0, 2.0, 2.0])

        levels = chain(solved, fetcher).index_levels

        # Half the book is flat, half of it doubles.
        assert levels.iloc[0] == pytest.approx(BASE)
        assert levels.iloc[2] == pytest.approx(1.5 * BASE)


class TestAFailingCurrencyLookupPropagates:
    """The bare `except Exception` around the reference read is gone."""

    def test_it_is_not_turned_into_the_index_currency(self):
        """A lookup that fails is not a name quoted in the index's own money.

        The old handler logged and returned the default, which is the
        BN-188 defect wearing an error for a cause — and it would have
        absorbed any refusal the data layer raised.
        """
        fetcher = Fetcher({"BP": [100.0] * 4}, currencies={"BP": "GBP"})
        fetcher.reference_error = ConnectionError("no route")

        with pytest.raises(ConnectionError, match="no route"):
            chain({DAYS[0]: {"BP": 1.0}}, fetcher)

    def test_reference_data_without_currency_still_defaults(self):
        """Triaged as leave: a dataset that does not model currency.

        There is no value being substituted for — a reference frame with no
        CURRENCY column is single-currency by construction, and reading it as
        the index's own currency is the only interpretation available.
        """
        fetcher = Fetcher({"AAA": [100.0, 110.0, 110.0, 110.0]})

        levels = chain({DAYS[0]: {"AAA": 1.0}}, fetcher).index_levels

        assert levels.iloc[1] == pytest.approx(1.1 * BASE)


# -- BN-191: the chained path takes the calculated path's decision ------------


class TestAnUnconvertibleHoldingIsRefused:
    """BN-191: the twin of `MarketValuesMixin._fx_rate`, moved together.

    A name quoted where no rate reaches the index currency used to yield a
    column of NaN, which `_units_for` turned into zero units, which left the
    chained index short by that name's whole weight and published the
    shortfall. The calculated path refuses it; if this one substituted, an
    optimised index and its parent would disagree about what an unvaluable
    name means — the two-layer disagreement of #195 and #198 in a new place.
    """

    def test_a_held_foreign_name_with_no_rate_refuses(self):
        solved = {DAYS[0]: {"AAA": 0.5, "BP": 0.5}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "BP": [100.0] * 4},
                          currencies={"AAA": "USD", "BP": "GBP"})

        with pytest.raises(CalculationError) as raised:
            chain(solved, fetcher)

        message = str(raised.value)

        assert "GBP/USD" in message
        assert "BP" in message

    def test_a_foreign_name_never_held_is_tolerated(self):
        """The same carve-out the missing-price refusal makes.

        A name carried at a weight of zero is never valued, so the pair it
        would need is not needed either.
        """
        solved = {DAYS[0]: {"AAA": 1.0, "BP": 0.0}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "BP": [100.0] * 4},
                          currencies={"AAA": "USD", "BP": "GBP"})

        assert chain(solved, fetcher).index_levels.iloc[0] == pytest.approx(BASE)

    def test_the_refusal_does_not_depend_on_which_name_came_first(self):
        """The rate cache holds None for an unknown pair, not a NaN column.

        Two names in the same unconvertible currency, one held and one not: had
        the cache stored the unheld name's NaN series the held name would have
        read a cached "conversion" and never reached the refusal. `SPARE`
        sorts before `BP` is reached, which is the ordering that would have hid
        it.
        """
        solved = {DAYS[0]: {"AAA": 0.5, "BP": 0.5, "SPARE": 0.0}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "BP": [100.0] * 4,
                           "SPARE": [100.0] * 4},
                          currencies={"AAA": "USD", "BP": "GBP",
                                      "SPARE": "GBP"})

        with pytest.raises(CalculationError, match="GBP/USD"):
            chain(solved, fetcher)


class TestPricedAtZeroIsNotTheSameAsUnvaluable:
    """BN-191: `isna or <= 0.0` covered two different events."""

    def test_a_name_quoted_at_zero_holds_zero_units_rather_than_refusing(self):
        """A price of zero is an observation, not the absence of one.

        No finite position in a worthless name carries a weight, so zero units
        is the only answer arithmetic allows — and half the book is then
        uninvested, which is why the level is half the base. That is the
        honest consequence of a real quote, so it computes; the case BN-191
        refuses is the one where the shortfall was manufactured by data that
        was simply not there.
        """
        solved = {DAYS[0]: {"AAA": 0.5, "ZERO": 0.5}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "ZERO": [0.0] * 4})

        levels = chain(solved, fetcher).index_levels

        assert levels.iloc[0] == pytest.approx(0.5 * BASE)

    def test_a_negative_price_refuses(self):
        """A negative equity price is not a price."""
        solved = {DAYS[0]: {"AAA": 0.5, "NEG": 0.5}}
        fetcher = Fetcher({"AAA": [100.0] * 4, "NEG": [-5.0] * 4})

        with pytest.raises(CalculationError, match="NEG"):
            chain(solved, fetcher)
