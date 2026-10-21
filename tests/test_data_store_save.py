# tests/test_data_store_save.py
"""BN-332: a store is written a chunk at a time, compressed as it goes, and
says how far it has got.

The chunked write must read back exactly as one `to_csv` call would, stay
byte-for-byte reproducible, and report progress that only rises and ends at 1.
"""
import gzip

import pandas as pd
import pytest

from beacon.data import store
from beacon.synthetic.dataset import SAVE_END, SAVE_START, SyntheticConfig, write
from test_data_store import build_fetcher


@pytest.fixture
def small_chunks(monkeypatch):
    """Chunks of three rows, so a small frame is written in several."""
    monkeypatch.setattr(store, "CHUNK_ROWS", 3)


def frame() -> pd.DataFrame:
    return pd.DataFrame({"IDENTIFIER": [f"N{i:02d}" for i in range(10)],
                         "CLOSE": [100.0 + i / 3 for i in range(10)],
                         "VOLUME": [float(i * 1000) for i in range(10)]})


class TestTheChunkedWrite:

    def test_reads_back_as_one_call_would_write_it(self,
                                                   tmp_path,
                                                   small_chunks):
        written = tmp_path / "chunked.csv.gz"
        store._write_frame(frame(), written)

        assert gzip.decompress(written.read_bytes()).decode("utf-8") == \
            frame().to_csv(index=False, lineterminator="\n")

    def test_an_empty_frame_still_writes_its_header(self,
                                                    tmp_path):
        written = tmp_path / "empty.csv.gz"
        store._write_frame(frame().iloc[:0], written)

        assert gzip.decompress(written.read_bytes()) == b"IDENTIFIER,CLOSE,VOLUME\n"

    def test_is_byte_for_byte_reproducible(self,
                                           tmp_path,
                                           small_chunks):
        store._write_frame(frame(), tmp_path / "one.csv.gz")
        store._write_frame(frame(), tmp_path / "two.csv.gz")

        assert (tmp_path / "one.csv.gz").read_bytes() == (tmp_path / "two.csv.gz").read_bytes()

    def test_the_header_names_no_file(self,
                                      tmp_path):
        """Flag bit 3 of a gzip header marks a file name; a staged write and an
        in-place one must be the same bytes."""
        store._write_frame(frame(), tmp_path / "staged.csv.gz")

        assert (tmp_path / "staged.csv.gz").read_bytes()[3] & 0x08 == 0

    def test_reports_each_chunk(self,
                                tmp_path,
                                small_chunks):
        seen: list[float] = []
        store._write_frame(frame(), tmp_path / "x.csv.gz", seen.append)

        assert seen == [0.3, 0.6, 0.9, 1.0]


class TestTheSavesProgress:

    def test_rises_to_one_through_every_dataset(self,
                                                tmp_path,
                                                small_chunks):
        seen: list[tuple[float, str]] = []
        store.save(build_fetcher(), tmp_path / "store",
                   progress=lambda fraction, message: seen.append((fraction, message)))
        fractions = [fraction for fraction, _ in seen]

        assert fractions == sorted(fractions)
        assert fractions[-1] == pytest.approx(1.0)
        assert {message for _, message in seen} == {
            "Saving market data", "Saving reference data", "Saving corporate actions"}

    def test_the_store_reads_back_unchanged(self,
                                            tmp_path,
                                            small_chunks):
        fetcher = build_fetcher()
        loaded = store.load(store.save(fetcher, tmp_path / "store"))

        pd.testing.assert_frame_equal(loaded.market.data, fetcher.market.data)


class TestASyntheticWrite:

    def test_moves_through_the_save(self,
                                    tmp_path):
        seen: list[tuple[float, str]] = []
        write(SyntheticConfig(assets=8, start="2022-01-03", end="2022-06-03"),
              tmp_path / "store", lambda fraction, message: seen.append((fraction, message)))
        fractions = [fraction for fraction, _ in seen]
        saving = [fraction for fraction, message in seen if message.startswith("Saving ")
                  and message != "Saving the store"]

        assert fractions == sorted(fractions)
        assert seen[-1] == (1.0, "Done")
        assert saving and min(saving) >= SAVE_START and max(saving) <= SAVE_END
