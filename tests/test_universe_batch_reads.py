# tests/test_universe_batch_reads.py
"""BN-192: a universe resolves in one reference read, and stays itself.

The batch read these cover changes no answer — every assertion below held
before it existed. That is the point. `_get_universe` builds an `Equity` per
identifier out of a row, and since BN-188 the `CURRENCY` on that row decides
which FX rate the name's market cap is converted at. So a batched read that
lined the frame up by position rather than by name would not produce an obvious
failure: it would produce a universe of plausible companies wearing each
other's currencies, and therefore an index with wrong weights that sum to one.

The identity tests below use names whose alphabetical order, universe order and
attribute values all disagree, so a positional join is wrong in a way the
assertions can see. The refusal tests are the BN-184 pair the batch had to
carry across unchanged: a name the reference data has never heard of is skipped
and counted, and a definition with no universe at all is refused before any
read happens.
"""
import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted

DATES = pd.bdate_range("2024-01-01", "2024-03-29")
AS_OF = pd.Timestamp("2024-02-15")

# The universe in the order the definition names it, which is neither
# alphabetical nor the order any attribute below sorts into.
UNIVERSE = ["ZED", "ACME", "MIDCO", "BRAVO", "YOKO"]

# Every field distinct, and deliberately anti-correlated with the ordering:
# alphabetically the names run ACME, BRAVO, MIDCO, YOKO, ZED, so a read that
# joined the returned block to the universe by position would hand ZED's GBP to
# ACME and ACME's USD to BRAVO — plausible, ordered, and wrong.
PROFILES = {
    "ZED": ("Zed Holdings", "GBP", "XLON"),
    "ACME": ("Acme Industrial", "USD", "XNYS"),
    "MIDCO": ("Midco Group", "JPY", "XTKS"),
    "BRAVO": ("Bravo Media", "EUR", "XETR"),
    "YOKO": ("Yoko Systems", "CHF", "XSWX"),
}


def reference_rows(profiles: dict[str, tuple[str, str, str]] | None = None
                   ) -> list[dict[str, object]]:
    """One open-ended record per name."""
    held = profiles if profiles is not None else PROFILES

    return [{"IDENTIFIER": identifier,
             "DATE_FROM": "2020-01-01",
             "DATE_TO": pd.NaT,
             "NAME": name,
             "CURRENCY": currency,
             "EXCHANGE": exchange}
            for identifier, (name, currency, exchange) in held.items()]


# One per currency a profile above quotes in, so a name resolved with its own
# currency can actually be valued. They are distinct and nowhere near parity:
# a name wearing a neighbour's currency would be converted at the wrong one of
# these, which is what makes a misattributed CURRENCY a wrong weight rather
# than a cosmetic error (BN-188).
RATES = {"GBP": 1.25, "EUR": 1.10, "JPY": 0.0067, "CHF": 1.15}


def market_rows(identifiers: list[str]) -> list[dict[str, object]]:
    """Flat prices, so nothing here depends on the market data."""
    rows: list[dict[str, object]] = [
        {"IDENTIFIER": identifier,
         "DATE": date,
         "CLOSE": 100.0,
         "SHARES_OUTSTANDING": 1_000.0}
        for identifier in identifiers
        for date in DATES]

    rows += [{"IDENTIFIER": f"{currency}USD", "DATE": date, "RATE": rate}
             for currency, rate in RATES.items()
             for date in DATES]

    return rows


def build_fetcher(rows: list[dict[str, object]] | None = None,
                  identifiers: list[str] | None = None) -> DataFetcher:
    """A fetcher over the five names, or over whatever records are supplied."""
    reference = pd.DataFrame(rows if rows is not None else reference_rows())
    market = pd.DataFrame(market_rows(identifiers if identifiers is not None
                                      else UNIVERSE))

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def build_calculator(fetcher: DataFetcher,
                     universe: list[str] | None = None) -> IndexCalculator:
    """A calculator over *universe*, in the order given."""
    definition = IndexDefinition(
        index_id="BATCH",
        index_name="Batch Read Index",
        base_date="2024-01-02",
        base_value=1000.0,
        currency="USD",
        eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="QUARTERLY",
        calendar="XNYS",
        universe_identifiers=list(universe if universe is not None
                                  else UNIVERSE))

    return IndexCalculator(definition, fetcher)


def profile_of(asset: Equity) -> tuple[str, str, str]:
    """What the calculator decided this name is."""
    return (asset.name, asset.currency, asset.exchange)


class CountingReference(ReferenceData):
    """A ReferenceData that records every query made of it."""

    def __init__(self,
                 inner: ReferenceData):
        self._df = inner._df
        self.calls: list[object] = []

    def get(self,
            identifier,
            date=None,
            columns=None):
        self.calls.append(identifier)

        return super().get(identifier, date, columns)


def counting_fetcher(rows: list[dict[str, object]] | None = None,
                     identifiers: list[str] | None = None) -> DataFetcher:
    """A fetcher whose reference reads are counted."""
    fetcher = build_fetcher(rows, identifiers)
    fetcher._reference = CountingReference(fetcher._reference)

    return fetcher


class TestPerNameIdentity:
    """A batched read must not move a row from one name onto another."""

    def test_each_name_carries_its_own_reference_row(self):
        universe = build_calculator(build_fetcher()).resolve_universe(AS_OF)

        assert {asset.ticker: profile_of(asset) for asset in universe} == PROFILES

    def test_the_universe_comes_back_in_the_order_it_was_named(self):
        universe = build_calculator(build_fetcher()).resolve_universe(AS_OF)

        assert [asset.ticker for asset in universe] == UNIVERSE

    def test_the_join_does_not_depend_on_the_order_asked_in(self):
        """Reversing the universe reverses the list and moves no attribute."""
        fetcher = build_fetcher()
        reversed_names = list(reversed(UNIVERSE))

        forward = build_calculator(fetcher).resolve_universe(AS_OF)
        backward = build_calculator(fetcher,
                                    reversed_names).resolve_universe(AS_OF)

        assert [asset.ticker for asset in backward] == reversed_names
        assert ({asset.ticker: profile_of(asset) for asset in backward}
                == {asset.ticker: profile_of(asset) for asset in forward})

    def test_a_subset_reads_only_its_own_names(self):
        """Asking for three of five must not pull in the other two's rows."""
        wanted = ["MIDCO", "ZED", "YOKO"]

        universe = build_calculator(build_fetcher(), wanted).resolve_universe(AS_OF)

        assert [asset.ticker for asset in universe] == wanted
        assert ({asset.ticker: profile_of(asset) for asset in universe}
                == {name: PROFILES[name] for name in wanted})

    def test_a_repeated_identifier_resolves_to_itself_twice(self):
        """A universe naming a name twice gets that name twice, not a neighbour."""
        universe = build_calculator(build_fetcher(),
                                    ["MIDCO", "ACME", "MIDCO"]).resolve_universe(AS_OF)

        assert [asset.ticker for asset in universe] == ["MIDCO", "ACME", "MIDCO"]
        assert [profile_of(asset) for asset in universe] == [
            PROFILES["MIDCO"], PROFILES["ACME"], PROFILES["MIDCO"]]

    def test_defaults_fill_in_per_name_where_a_column_is_absent(self):
        """A frame with no NAME or EXCHANGE still falls back per identifier."""
        rows = [{"IDENTIFIER": identifier,
                 "DATE_FROM": "2020-01-01",
                 "DATE_TO": pd.NaT,
                 "CURRENCY": currency}
                for identifier, (_, currency, _) in PROFILES.items()]

        universe = build_calculator(build_fetcher(rows)).resolve_universe(AS_OF)

        assert {asset.ticker: profile_of(asset) for asset in universe} == {
            identifier: (identifier, currency, "UNKNOWN")
            for identifier, (_, currency, _) in PROFILES.items()}


# ZED and ACME each change hands, on different dates and in different
# directions. The records are listed youngest-first for ZED and oldest-first for
# ACME, so "the first row in the block" and "the last row in the block" are each
# wrong for one of them.
HISTORIC: list[dict[str, object]] = [
    {"IDENTIFIER": "ZED", "DATE_FROM": "2024-01-01", "DATE_TO": pd.NaT,
     "NAME": "Zed Global", "CURRENCY": "USD", "EXCHANGE": "XNYS"},
    {"IDENTIFIER": "ZED", "DATE_FROM": "2020-01-01",
     "DATE_TO": "2023-12-31",
     "NAME": "Zed Holdings", "CURRENCY": "GBP", "EXCHANGE": "XLON"},
    {"IDENTIFIER": "ACME", "DATE_FROM": "2020-01-01",
     "DATE_TO": "2024-06-30",
     "NAME": "Acme Industrial", "CURRENCY": "USD", "EXCHANGE": "XNYS"},
    {"IDENTIFIER": "ACME", "DATE_FROM": "2024-07-01", "DATE_TO": pd.NaT,
     "NAME": "Acme Global", "CURRENCY": "EUR", "EXCHANGE": "XETR"},
]


class TestPointInTime:
    """The batch must resolve each name's record for the date, not the frame's."""

    def resolved(self,
                 date: pd.Timestamp) -> dict[str, tuple[str, str, str]]:
        fetcher = build_fetcher(HISTORIC, ["ZED", "ACME"])
        universe = build_calculator(fetcher,
                                    ["ZED", "ACME"]).resolve_universe(date)

        return {asset.ticker: profile_of(asset) for asset in universe}

    def test_each_name_resolves_its_own_record_for_the_date(self):
        assert self.resolved(pd.Timestamp("2024-02-15")) == {
            "ZED": ("Zed Global", "USD", "XNYS"),
            "ACME": ("Acme Industrial", "USD", "XNYS")}

    def test_an_earlier_date_reads_the_earlier_records(self):
        assert self.resolved(pd.Timestamp("2022-06-15")) == {
            "ZED": ("Zed Holdings", "GBP", "XLON"),
            "ACME": ("Acme Industrial", "USD", "XNYS")}

    def test_a_later_date_reads_the_later_records(self):
        assert self.resolved(pd.Timestamp("2024-09-15")) == {
            "ZED": ("Zed Global", "USD", "XNYS"),
            "ACME": ("Acme Global", "EUR", "XETR")}

    def test_a_name_with_no_record_on_the_date_is_skipped_not_borrowed(self):
        """ACME's row must not stand in for a ZED that did not exist yet."""
        rows = [row for row in HISTORIC
                if not (row["IDENTIFIER"] == "ZED"
                        and row["DATE_FROM"] == "2020-01-01")]
        fetcher = build_fetcher(rows, ["ZED", "ACME"])

        universe = build_calculator(fetcher, ["ZED", "ACME"]).resolve_universe(
            pd.Timestamp("2022-06-15"))

        assert [asset.ticker for asset in universe] == ["ACME"]
        assert profile_of(universe[0]) == ("Acme Industrial", "USD", "XNYS")


class TestSkipsAndRefusals:
    """BN-184's two answers must survive the batch unchanged."""

    def test_an_unknown_name_is_skipped_and_the_rest_are_computed(self,
                                                                 caplog):
        fetcher = build_fetcher()
        calculator = build_calculator(fetcher,
                                      ["ZED", "NOSUCH", "BRAVO"])

        with caplog.at_level("INFO"):
            universe = calculator.resolve_universe(AS_OF)

        assert [asset.ticker for asset in universe] == ["ZED", "BRAVO"]
        assert ({asset.ticker: profile_of(asset) for asset in universe}
                == {"ZED": PROFILES["ZED"], "BRAVO": PROFILES["BRAVO"]})
        assert "No reference data for 'NOSUCH'" in caplog.text

    def test_the_skip_is_counted_in_the_log(self,
                                            caplog):
        calculator = build_calculator(build_fetcher(),
                                      ["ZED", "NOSUCH", "BRAVO"])

        with caplog.at_level("INFO"):
            calculator.resolve_universe(AS_OF)

        assert "Resolved 2/3 identifiers" in caplog.text

    def test_a_universe_none_of_which_resolves_is_empty_not_a_refusal(self):
        universe = build_calculator(build_fetcher(),
                                    ["NOSUCH", "NEITHER"]).resolve_universe(AS_OF)

        assert universe == []

    def test_no_universe_at_all_is_refused_before_any_read(self):
        fetcher = counting_fetcher()
        calculator = build_calculator(fetcher)
        calculator.definition.universe_identifiers = None
        reference = fetcher._reference
        assert isinstance(reference, CountingReference)

        with pytest.raises(CalculationError, match="no universe_identifiers"):
            calculator.resolve_universe(AS_OF)

        assert reference.calls == []

    def test_a_failing_read_is_not_absorbed_as_a_smaller_universe(self):
        """A broken data layer is not a universe that happens to be empty."""
        class BrokenReference(ReferenceData):
            def __init__(self,
                         inner: ReferenceData):
                self._df = inner._df

            def get(self,
                    identifier,
                    date=None,
                    columns=None):
                raise ConnectionError("no route to reference store")

        fetcher = build_fetcher()
        fetcher._reference = BrokenReference(fetcher._reference)

        with pytest.raises(ConnectionError, match="no route"):
            build_calculator(fetcher).resolve_universe(AS_OF)


class TestReadsOnce:
    """The reads a universe makes must not grow with the universe."""

    def test_one_read_for_the_whole_universe(self):
        fetcher = counting_fetcher()
        reference = fetcher._reference
        assert isinstance(reference, CountingReference)

        build_calculator(fetcher).resolve_universe(AS_OF)

        assert reference.calls == [UNIVERSE]

    def test_read_count_is_flat_in_the_universe_size(self):
        """Two names or two hundred: the same number of reference reads."""
        counts = []

        for size in (2, 200):
            names = [f"N{index:04d}" for index in range(size)]
            profiles = {name: (f"{name} Corp", "USD", "XNYS") for name in names}

            fetcher = counting_fetcher(reference_rows(profiles), names)
            reference = fetcher._reference
            assert isinstance(reference, CountingReference)

            universe = build_calculator(fetcher, names).resolve_universe(AS_OF)

            assert [asset.ticker for asset in universe] == names
            counts.append(len(reference.calls))

        assert counts[0] == counts[1] == 1


class TestTheRunAgrees:
    """`run` and `resolve_universe` resolve the same names the same way."""

    def test_the_base_composition_is_the_resolved_universe(self):
        fetcher = build_fetcher()
        calculator = build_calculator(fetcher)

        result = calculator.run(end_date="2024-01-31")
        base = min(result.constituent_snapshots)

        assert result.constituent_snapshots[base] == UNIVERSE
