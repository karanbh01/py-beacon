# tests/test_synthetic_extend.py
"""BN-238: extending a generated store to a later date without changing it.

The base store turns its universe over quickly (40% listing and delisting
hazards) so a one-year extension has names joining and leaving to check.
"""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beacon.data import store
from beacon.synthetic import SyntheticConfig, extend, write
from beacon.synthetic import state as state_module
from beacon.synthetic.__main__ import main
from beacon.synthetic.universe import ticker_suffix

BASE = SyntheticConfig(assets=80, start="2023-01-02", end="2024-05-31", seed=11,
                       listing_rate=0.4, delisting_rate=0.4)
LAST = "2024-05-31"
EXTENDED_TO = "2025-06-30"
FILES = (store.MARKET_FILE, store.REFERENCE_FILE, store.ACTIONS_FILE,
         store.FEATURES_FILE)


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    return write(BASE, tmp_path_factory.mktemp("base") / "store")


@pytest.fixture(scope="module")
def extended(base,
             tmp_path_factory) -> Path:
    path = copy(base, tmp_path_factory.mktemp("extended"))
    extend(path, EXTENDED_TO)

    return path


def copy(source: Path,
         folder: Path) -> Path:
    return Path(shutil.copytree(source, folder / "store"))


def raw(path: Path,
        name: str) -> bytes:
    return (path / name).read_bytes()


def table(path: Path,
          name: str) -> pd.DataFrame:
    """A stored file as text, exactly as written."""
    with gzip.open(path / name, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, dtype=str, keep_default_na=False)


def market(path: Path) -> pd.DataFrame:
    frame = table(path, store.MARKET_FILE)

    for column in ("CLOSE", "SHARES_OUTSTANDING"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


class TestHistoryIsKept:

    @pytest.mark.parametrize("name", [store.MARKET_FILE, store.FEATURES_FILE])
    def test_the_stored_bytes_are_untouched(self,
                                            base,
                                            extended,
                                            name):
        """New rows are appended, so the file begins with exactly the bytes
        it held before."""
        before = raw(base, name)

        assert raw(extended, name)[:len(before)] == before

    def test_every_market_row_up_to_the_join_is_unchanged(self,
                                                          base,
                                                          extended):
        before = table(base, store.MARKET_FILE)
        after = table(extended, store.MARKET_FILE)
        kept = after.loc[after["DATE"] <= LAST].sort_values(
            ["IDENTIFIER", "DATE"], ignore_index=True)

        pd.testing.assert_frame_equal(
            kept, before.sort_values(["IDENTIFIER", "DATE"], ignore_index=True))

    def test_every_new_row_is_after_the_join(self,
                                             base,
                                             extended):
        for name in (store.MARKET_FILE, store.FEATURES_FILE):
            added = len(table(extended, name)) - len(table(base, name))
            dates = table(extended, name)["DATE"].iloc[-added:]

            assert added > 0
            assert (dates > LAST).all()

    def test_actions_change_only_from_announced_to_paid(self,
                                                        base,
                                                        extended):
        before = table(base, store.ACTIONS_FILE)
        after = table(extended, store.ACTIONS_FILE)
        key = ["IDENTIFIER", "EX_DATE", "TYPE"]
        matched = before.merge(after, on=key, suffixes=("", "_after"))

        assert len(matched) == len(before)
        assert (matched["VALUE"] == matched["VALUE_after"]).all()
        assert (matched["PAY_DATE"] == matched["PAY_DATE_after"]).all()

        changed = matched.loc[matched["STATUS"] != matched["STATUS_after"]]
        assert not changed.empty
        assert set(changed["STATUS"]) == {"announced"}
        assert set(changed["STATUS_after"]) == {"paid"}

    def test_reference_changes_only_listing_state_and_earnings(self,
                                                               base,
                                                               extended):
        before = table(base, store.REFERENCE_FILE).set_index("IDENTIFIER")
        after = table(extended, store.REFERENCE_FILE).set_index("IDENTIFIER")
        moving = {"DATE_TO", "TRADING_STATUS", "NEXT_EARNINGS"}
        fixed = [column for column in before.columns if column not in moving]

        pd.testing.assert_frame_equal(after.loc[before.index, fixed],
                                      before[fixed])


class TestReproducible:

    def test_the_same_extension_gives_the_same_bytes(self,
                                                     base,
                                                     extended,
                                                     tmp_path):
        again = copy(base, tmp_path)
        extend(again, EXTENDED_TO)

        for name in (*FILES, "synthetic/universe.csv.gz",
                     "synthetic/settings.json"):
            assert raw(again, name) == raw(extended, name), name

    def test_a_weekend_end_date_changes_nothing(self,
                                                base,
                                                tmp_path):
        """Extending to Friday or to the Sunday after adds the same sessions
        and draws the same numbers."""
        friday, sunday = copy(base, tmp_path / "a"), copy(base, tmp_path / "b")
        extend(friday, "2024-09-06")
        extend(sunday, "2024-09-08")

        assert raw(friday, store.MARKET_FILE) == raw(sunday, store.MARKET_FILE)

    def test_nothing_to_add_leaves_the_store_alone(self,
                                                   base,
                                                   tmp_path):
        path = copy(base, tmp_path)
        before = {name: raw(path, name) for name in FILES}

        added = extend(path, LAST)

        assert added.sessions == 0 and added.first is None
        assert {name: raw(path, name) for name in FILES} == before


class TestCarriesOn:

    def test_prices_continue_from_the_last_close(self,
                                                 extended):
        frame = market(extended)
        equities = frame.loc[frame["IDENTIFIER"].str.startswith("CMP")]
        last = equities.loc[equities["DATE"] == LAST].set_index("IDENTIFIER")
        first = equities.loc[equities["DATE"] == "2024-06-03"].set_index(
            "IDENTIFIER")
        both = last.index.intersection(first.index)

        moves = np.log(first.loc[both, "CLOSE"] / last.loc[both, "CLOSE"])

        assert len(both) > 40
        assert moves.abs().max() < 0.25

    def test_share_counts_carry_on(self,
                                   extended):
        frame = market(extended)
        last = frame.loc[frame["DATE"] == LAST].set_index("IDENTIFIER")
        first = frame.loc[frame["DATE"] == "2024-06-03"].set_index("IDENTIFIER")
        both = last.index.intersection(first.index)

        pd.testing.assert_series_equal(
            first.loc[both, "SHARES_OUTSTANDING"],
            last.loc[both, "SHARES_OUTSTANDING"])

    def test_exchange_rates_continue_from_the_last_rate(self,
                                                        extended):
        frame = market(extended)
        pairs = frame.loc[~frame["IDENTIFIER"].str.startswith("CMP")]
        last = pairs.loc[pairs["DATE"] == LAST].set_index("IDENTIFIER")["CLOSE"]
        first = pairs.loc[pairs["DATE"] == "2024-06-03"].set_index(
            "IDENTIFIER")["CLOSE"]

        assert np.log(first / last.loc[first.index]).abs().max() < 0.05

    def test_no_month_gets_a_second_dividend(self,
                                             extended):
        actions = table(extended, store.ACTIONS_FILE)
        dividends = actions.loc[actions["TYPE"] == "DIVIDEND"]
        months = dividends["EX_DATE"].str[:7]

        assert not dividends.assign(month=months).duplicated(
            ["IDENTIFIER", "month"]).any()
        assert (dividends["EX_DATE"] > LAST).any()

    def test_realised_volatility_matches_the_targets(self,
                                                     extended):
        """Across the new year, the median realised volatility sits near the
        median target, as in a freshly generated panel.

        Measured over seeds 1-6: realised over target ran 1.02-1.05 for the
        extension and 0.94-1.07 for a fresh panel over the same dates. The
        bound sits outside both.
        """
        _, names = state_module.load(extended)
        frame = market(extended)
        new = frame.loc[frame["DATE"] > LAST].pivot(index="DATE",
                                                    columns="IDENTIFIER",
                                                    values="CLOSE")
        carried = [name for name in new.columns if name in names.index
                   and new[name].notna().sum() > 200]
        moves = np.log(new[carried]).diff()
        realised = moves[moves.abs() < 0.5].std() * np.sqrt(252)  # not splits

        assert realised.median() == pytest.approx(
            names.loc[carried, "volatility"].median(), rel=0.15)


class TestListings:

    def test_names_that_leave_stop_trading_and_say_so(self,
                                                      extended):
        reference = table(extended, store.REFERENCE_FILE).set_index("IDENTIFIER")
        left = reference.loc[(reference["DATE_TO"] > LAST)]
        frame = table(extended, store.MARKET_FILE)

        assert not left.empty
        assert (left["TRADING_STATUS"] == "Delisted").all()

        for identifier, row in left.iterrows():
            dates = frame.loc[frame["IDENTIFIER"] == identifier, "DATE"]
            assert dates.max() <= row["DATE_TO"]

    def test_new_names_continue_the_ticker_sequence(self,
                                                    base,
                                                    extended):
        before = set(table(base, store.REFERENCE_FILE)["IDENTIFIER"])
        reference = table(extended, store.REFERENCE_FILE).set_index("IDENTIFIER")
        joined = reference.loc[~reference.index.isin(before)]
        expected = {f"CMP{ticker_suffix(position)}"
                    for position in range(80, 80 + len(joined))}

        assert len(joined) > 0
        assert set(joined.index) == expected
        assert (joined["DATE_FROM"] > LAST).all()
        assert (joined["TRADING_STATUS"] == "Active").all()
        assert (joined["ISIN"] != "").all()

    def test_new_names_trade_only_from_their_listing(self,
                                                     base,
                                                     extended):
        before = set(table(base, store.REFERENCE_FILE)["IDENTIFIER"])
        reference = table(extended, store.REFERENCE_FILE).set_index("IDENTIFIER")
        frame = table(extended, store.MARKET_FILE)

        for identifier in reference.index.difference(list(before)):
            dates = frame.loc[frame["IDENTIFIER"] == identifier, "DATE"]
            assert dates.min() >= reference.loc[identifier, "DATE_FROM"]

    def test_the_state_records_both(self,
                                    extended):
        settings, names = state_module.load(extended)

        assert settings.end == EXTENDED_TO
        assert settings.extensions == [{"from": "2024-06-03",
                                        "to": EXTENDED_TO}]
        assert len(names) > 80
        assert (names["listed_to"] > pd.Timestamp(LAST)).any()


class TestFeatures:

    def test_no_quarter_is_reported_twice(self,
                                          extended):
        features = table(extended, store.FEATURES_FILE)

        assert not features.duplicated(["IDENTIFIER", "FIELD", "DETAIL"]).any()

    def test_a_quarter_ending_before_the_join_is_reported_after_it(self,
                                                                   base,
                                                                   extended):
        before = len(table(base, store.FEATURES_FILE))
        added = table(extended, store.FEATURES_FILE).iloc[before:]

        assert added["DETAIL"].str.contains("period ending 2024-03-31").any()

    def test_ratios_agree_with_the_prices(self,
                                          base,
                                          extended):
        """eps x pe is the close at the quarter end, as in generated data."""
        before = len(table(base, store.FEATURES_FILE))
        added = table(extended, store.FEATURES_FILE).iloc[before:]
        frame = market(extended)

        row = added.loc[(added["FIELD"] == "eps")
                        & added["DETAIL"].str.contains("2024-09-30")].iloc[0]
        pe = added.loc[(added["IDENTIFIER"] == row["IDENTIFIER"])
                       & (added["DETAIL"] == row["DETAIL"])
                       & (added["FIELD"] == "pe_ratio"), "VALUE"].iloc[0]
        close = frame.loc[(frame["IDENTIFIER"] == row["IDENTIFIER"])
                          & (frame["DATE"] == "2024-09-30"), "CLOSE"].iloc[0]

        assert float(row["VALUE"]) * float(pe) == pytest.approx(close)


class TestUsingIt:

    def test_the_extended_store_loads(self,
                                      extended):
        fetcher = store.load(extended)
        dates = fetcher.market.data.index.get_level_values("DATE")

        assert dates.max() == pd.Timestamp(EXTENDED_TO)

    def test_it_extends_again(self,
                              extended,
                              tmp_path):
        path = copy(extended, tmp_path)
        before = raw(path, store.MARKET_FILE)

        added = extend(path, "2025-09-30")
        settings, _ = state_module.load(path)

        assert added.first == "2025-07-01"
        assert raw(path, store.MARKET_FILE)[:len(before)] == before
        assert len(settings.extensions) == 2

    def test_a_store_without_generator_state_is_refused(self,
                                                        base,
                                                        tmp_path):
        path = copy(base, tmp_path)
        shutil.rmtree(path / state_module.STATE_DIRECTORY)

        with pytest.raises(ValueError, match="cannot be extended"):
            extend(path, EXTENDED_TO)

    def test_generation_saves_the_settings(self,
                                           base):
        settings = json.loads((base / "synthetic" / "settings.json").read_text())

        assert settings["seed"] == 11
        assert settings["end"] == "2024-05-31"
        assert settings["extensions"] == []

    def test_from_the_command_line(self,
                                   base,
                                   tmp_path,
                                   capsys):
        path = copy(base, tmp_path)

        assert main(["--extend", str(path), "--end", "2024-07-31"]) == 0
        assert "Extended" in capsys.readouterr().out
        assert main(["--extend", str(tmp_path / "nothing")]) == 2
