# tests/test_index_batch_reads.py
"""BN-190: a market-cap methodology reads its session once, for everyone.

The optimisation these cover is invisible by design — every answer below was
already the answer before the batch read existed. That is exactly why they are
worth writing: a batched read is one join away from attributing name A's shares
to name B, and a substituted number in a weight is the class of fault BN-179,
BN-182 and BN-188 have spent the week removing. So the tests here are about
identity and refusal first, and about the read count second.
"""
import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.calculation.selection import select_with_provenance
from beacon.index.context import IndexContext
from beacon.index.methodology import MarketCapRule, MarketCapWeighted

DATES = pd.bdate_range("2024-01-01", "2024-03-29")
SESSION = pd.Timestamp("2024-02-15")

# Deliberately anti-correlated: the name with the highest price has the lowest
# share count, and both orders run against the alphabetical one the identifiers
# sort into. A read that lined a column up by position rather than by name
# would produce caps that are plausible, ordered, and wrong.
PRICES = {"AAA": 10.0, "BBB": 40.0, "CCC": 90.0, "DDD": 160.0, "EEE": 250.0}
SHARES = {"AAA": 50_000.0, "BBB": 40_000.0, "CCC": 30_000.0,
          "DDD": 20_000.0, "EEE": 10_000.0}
# AAA 500k, BBB 1.6m, CCC 2.7m, DDD 3.2m, EEE 2.5m — no two alike, and the
# ordering is neither the price ordering nor the share ordering.
CAPS = {name: PRICES[name] * SHARES[name] for name in PRICES}

CURRENCIES = dict.fromkeys(PRICES, "USD")


def market_rows(prices: dict[str, float] | None = None,
                shares: dict[str, float] | None = None) -> list[dict[str, object]]:
    """One row per name per date, at flat prices and share counts."""
    close = prices if prices is not None else PRICES
    counts = shares if shares is not None else SHARES

    return [{"IDENTIFIER": name,
             "DATE": date,
             "CLOSE": close.get(name),
             "VOLUME": 1_000_000.0,
             "SHARES_OUTSTANDING": counts.get(name),
             "FREE_FLOAT": 0.5}
            for name in close
            for date in DATES]


def build_fetcher(rows: list[dict[str, object]] | None = None,
                  currencies: dict[str, str] | None = None) -> DataFetcher:
    """A fetcher over the five names, or over whatever rows are supplied."""
    market = pd.DataFrame(rows if rows is not None else market_rows())
    quoted = currencies if currencies is not None else CURRENCIES

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": quoted.get(name, "USD"), "EXCHANGE": "NYSE"}
        for name in quoted
    ])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def assets(names: list[str] | None = None,
           currencies: dict[str, str] | None = None) -> list[Equity]:
    """Equity objects for the universe, in the given order."""
    quoted = currencies if currencies is not None else CURRENCIES

    return [Equity(name=name, currency=quoted.get(name, "USD"),
                   ticker=name, exchange="NYSE")
            for name in (names if names is not None else list(PRICES))]


class CountingMarket(MarketData):
    """A MarketData that records every query made of it."""

    def __init__(self,
                 inner: MarketData):
        self._df = inner._df
        self.calls: list[object] = []

    def get(self,
            identifier,
            start_date=None,
            end_date=None,
            columns=None):
        self.calls.append(identifier)

        return super().get(identifier, start_date, end_date, columns)


def counting_fetcher(rows: list[dict[str, object]] | None = None) -> DataFetcher:
    """A fetcher whose market reads are counted."""
    fetcher = build_fetcher(rows)
    fetcher._market = CountingMarket(fetcher._market)

    return fetcher


class TestPerNameIdentity:
    """A batched read must not move a number from one name onto another."""

    def test_weights_are_each_name_s_own_cap(self):
        fetcher = build_fetcher()
        total = sum(CAPS.values())

        weights = MarketCapWeighted().calculate_weights(
            list(assets()), SESSION, fetcher, IndexContext(currency="USD"))

        by_name = {asset.ticker: weight for asset, weight in weights.items()}

        assert by_name == pytest.approx({name: cap / total
                                         for name, cap in CAPS.items()})

    def test_weights_do_not_depend_on_the_order_asked_in(self):
        """The join is by name, so reversing the universe changes nothing."""
        fetcher = build_fetcher()
        forward = MarketCapWeighted().calculate_weights(
            list(assets()), SESSION, fetcher, IndexContext(currency="USD"))
        backward = MarketCapWeighted().calculate_weights(
            list(assets(sorted(PRICES, reverse=True))), SESSION, fetcher,
            IndexContext(currency="USD"))

        assert ({asset.ticker: weight for asset, weight in forward.items()}
                == pytest.approx({asset.ticker: weight
                                  for asset, weight in backward.items()}))

    def test_the_rule_screens_each_name_on_its_own_cap(self):
        """A bound between two adjacent caps must cut exactly there."""
        fetcher = build_fetcher()
        rule = MarketCapRule(min_market_cap=2_000_000.0)

        result = select_with_provenance(list(assets()), [rule], SESSION,
                                        fetcher, IndexContext(currency="USD"))

        # CCC 2.7m, DDD 3.2m, EEE 2.5m clear it; AAA 500k and BBB 1.6m do not.
        assert sorted(result.survivor_ids) == ["CCC", "DDD", "EEE"]

    def test_free_float_is_read_per_name(self):
        """A per-name float factor must weight that name and no other."""
        rows = market_rows()
        for row in rows:
            row["FREE_FLOAT"] = 1.0 if row["IDENTIFIER"] == "AAA" else 0.25

        fetcher = build_fetcher(rows)
        floated = {name: CAPS[name] * (1.0 if name == "AAA" else 0.25)
                   for name in CAPS}
        total = sum(floated.values())

        weights = MarketCapWeighted(use_free_float=True).calculate_weights(
            list(assets()), SESSION, fetcher, IndexContext(currency="USD"))

        assert ({asset.ticker: weight for asset, weight in weights.items()}
                == pytest.approx({name: value / total
                                  for name, value in floated.items()}))


class TestReadsOnce:
    """The reads a universe makes must not grow with the universe."""

    def test_selection_and_weighting_share_one_session_read(self):
        fetcher = counting_fetcher()
        market = fetcher._market
        assert isinstance(market, CountingMarket)

        universe = list(assets())
        rule = MarketCapRule(min_market_cap=1.0)
        context = IndexContext(currency="USD")

        result = select_with_provenance(universe, [rule], SESSION, fetcher,
                                        context)
        MarketCapWeighted().calculate_weights(result.survivors, SESSION,
                                              fetcher, context)

        # One batch for the rule's candidates; the weighting's names are a
        # subset of those, on the same session, so it reads nothing again.
        assert len(market.calls) == 1
        assert market.calls[0] == list(PRICES)

    def test_read_count_is_flat_in_the_universe_size(self):
        """Two names or two hundred: the same number of frame reads."""
        counts = []

        for size in (2, 200):
            names = [f"N{index:04d}" for index in range(size)]
            prices = {name: 10.0 + index for index, name in enumerate(names)}
            shares = dict.fromkeys(names, 1_000.0)

            fetcher = counting_fetcher(market_rows(prices, shares))
            market = fetcher._market
            assert isinstance(market, CountingMarket)

            universe = assets(names, dict.fromkeys(names, "USD"))
            context = IndexContext(currency="USD")
            result = select_with_provenance(universe,
                                            [MarketCapRule(min_market_cap=1.0)],
                                            SESSION, fetcher, context)
            MarketCapWeighted().calculate_weights(result.survivors, SESSION,
                                                  fetcher, context)

            counts.append(len(market.calls))

        assert counts[0] == counts[1]


class TestRefusalsStillFire:
    """Every refusal the batched paths replaced must still be reachable."""

    def test_a_name_with_no_price_is_excluded_not_dropped_silently(self):
        rows = [row for row in market_rows()
                if not (row["IDENTIFIER"] == "CCC"
                        and row["DATE"] == SESSION)]
        fetcher = build_fetcher(rows)

        result = select_with_provenance(list(assets()),
                                        [MarketCapRule(min_market_cap=1.0)],
                                        SESSION, fetcher,
                                        IndexContext(currency="USD"))

        assert result.excluded_by("CCC") is not None
        assert "CCC" not in result.survivor_ids

    def test_a_name_with_no_shares_is_excluded_by_the_rule(self):
        shares = dict(SHARES)
        shares["DDD"] = float("nan")
        fetcher = build_fetcher(market_rows(shares=shares))

        result = select_with_provenance(list(assets()),
                                        [MarketCapRule(min_market_cap=1.0)],
                                        SESSION, fetcher,
                                        IndexContext(currency="USD"))

        assert "DDD" not in result.survivor_ids

    def test_a_name_with_no_shares_refuses_the_weighting(self):
        shares = dict(SHARES)
        shares["DDD"] = float("nan")
        fetcher = build_fetcher(market_rows(shares=shares))

        with pytest.raises(CalculationError, match="SHARES_OUTSTANDING"):
            MarketCapWeighted().calculate_weights(
                list(assets()), SESSION, fetcher, IndexContext(currency="USD"))

    def test_an_unconvertible_currency_refuses_the_rule(self):
        currencies = dict(CURRENCIES)
        currencies["EEE"] = "JPY"
        fetcher = build_fetcher(currencies=currencies)

        with pytest.raises(CalculationError, match="JPY/USD"):
            select_with_provenance(list(assets(currencies=currencies)),
                                   [MarketCapRule(min_market_cap=1.0)],
                                   SESSION, fetcher,
                                   IndexContext(currency="USD"))

    def test_an_unconvertible_currency_refuses_the_weighting(self):
        currencies = dict(CURRENCIES)
        currencies["EEE"] = "JPY"
        fetcher = build_fetcher(currencies=currencies)

        with pytest.raises(CalculationError, match="JPY/USD"):
            MarketCapWeighted().calculate_weights(
                list(assets(currencies=currencies)), SESSION, fetcher,
                IndexContext(currency="USD"))

    def test_a_date_past_the_data_refuses_the_rule(self):
        fetcher = build_fetcher()
        beyond = pd.Timestamp("2025-06-01")

        with pytest.raises(CalculationError, match="outside it"):
            select_with_provenance(list(assets()),
                                   [MarketCapRule(min_market_cap=1.0)],
                                   beyond, fetcher,
                                   IndexContext(currency="USD"))

    def test_a_date_past_the_data_refuses_the_weighting(self):
        fetcher = build_fetcher()

        with pytest.raises(CalculationError, match="outside it"):
            MarketCapWeighted().calculate_weights(
                list(assets()), pd.Timestamp("2025-06-01"), fetcher,
                IndexContext(currency="USD"))

    def test_a_name_priceable_nowhere_refuses_the_weighting(self):
        rows = [row for row in market_rows() if row["IDENTIFIER"] != "BBB"]
        rows += [{"IDENTIFIER": "BBB", "DATE": date, "CLOSE": None,
                  "VOLUME": 1.0, "SHARES_OUTSTANDING": SHARES["BBB"],
                  "FREE_FLOAT": 0.5}
                 for date in DATES]
        fetcher = build_fetcher(rows)

        with pytest.raises(CalculationError, match="no CLOSE on or before"):
            MarketCapWeighted().calculate_weights(
                list(assets()), SESSION, fetcher, IndexContext(currency="USD"))

    def test_a_quiet_day_still_walks_back_to_the_last_print(self):
        """A warmed session that holds no close for a name must not end the walk."""
        rows = market_rows()
        quiet = pd.Timestamp("2024-02-14")

        for row in rows:
            if row["IDENTIFIER"] == "CCC" and row["DATE"] >= quiet:
                row["CLOSE"] = None

        fetcher = build_fetcher(rows)
        total = sum(CAPS.values())

        weights = MarketCapWeighted().calculate_weights(
            list(assets()), SESSION, fetcher, IndexContext(currency="USD"))

        # CCC last printed at its flat price, so its cap is unchanged.
        assert ({asset.ticker: weight for asset, weight in weights.items()}
                == pytest.approx({name: cap / total
                                  for name, cap in CAPS.items()}))


class TestTheReadIsScopedToItsSession:
    """A warmed session must answer for itself and for nothing else."""

    def test_a_later_date_reads_its_own_prices(self):
        """Warming one session must not answer the next one out of it."""
        rising = pd.DataFrame(market_rows())
        rising.loc[rising["DATE"] > SESSION, "CLOSE"] *= 2.0

        fetcher = build_fetcher(rising.to_dict("records"))
        later = pd.Timestamp("2024-03-01")

        MarketCapWeighted().calculate_weights(list(assets()), SESSION, fetcher,
                                              IndexContext(currency="USD"))

        assert fetcher.fetch_price("AAA", SESSION.strftime("%Y-%m-%d")) == (
            pytest.approx(PRICES["AAA"]))
        assert fetcher.fetch_price("AAA", later.strftime("%Y-%m-%d")) == (
            pytest.approx(PRICES["AAA"] * 2.0))

    def test_a_name_outside_the_warmed_set_is_still_read(self):
        fetcher = build_fetcher()
        stamp = SESSION.strftime("%Y-%m-%d")

        fetcher.warm_session(["AAA", "BBB"], SESSION)

        assert fetcher.fetch_price("EEE", stamp) == pytest.approx(PRICES["EEE"])
        assert fetcher.fetch_shares_outstanding("EEE", stamp) == (
            pytest.approx(SHARES["EEE"]))

    def test_merged_rows_are_not_answered_out_of_a_stale_session(self):
        fetcher = build_fetcher()
        stamp = SESSION.strftime("%Y-%m-%d")

        fetcher.warm_session(list(PRICES), SESSION)
        assert fetcher.fetch_price("AAA", stamp) == pytest.approx(PRICES["AAA"])

        fetcher.merge_market_data(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": SESSION, "CLOSE": 999.0,
             "VOLUME": 1.0, "SHARES_OUTSTANDING": SHARES["AAA"],
             "FREE_FLOAT": 0.5}]))

        assert fetcher.fetch_price("AAA", stamp) == pytest.approx(999.0)

    def test_warming_is_invisible_to_the_answers(self):
        """Every read must give the same value warmed or cold."""
        cold = build_fetcher()
        warm = build_fetcher()
        stamp = SESSION.strftime("%Y-%m-%d")

        warm.warm_session(list(PRICES), SESSION)

        for name in PRICES:
            assert warm.fetch_price(name, stamp) == cold.fetch_price(name, stamp)
            assert (warm.fetch_shares_outstanding(name, stamp)
                    == cold.fetch_shares_outstanding(name, stamp))
            assert (warm.fetch_free_float_factor(name, stamp)
                    == cold.fetch_free_float_factor(name, stamp))
