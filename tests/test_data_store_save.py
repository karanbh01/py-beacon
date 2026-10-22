# tests/test_data_store_save.py
"""BN-332: a store is written a chunk at a time, and says how far it has got.

The chunked write must read back as the frame it was, stay byte-for-byte
reproducible, and report progress that only rises and ends at 1. Since
BN-354 a chunk is a Parquet row group.
"""
import pandas as pd
import pyarrow.parquet as pq
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

    def test_reads_back_as_the_frame_it_was(self,
                                            tmp_path,
                                            small_chunks):
        written = tmp_path / "chunked.parquet"
        store._write_frame(frame(), written)

        pd.testing.assert_frame_equal(store._read_frame(written), frame())
        assert pq.ParquetFile(written).num_row_groups == 4

    def test_an_empty_frame_still_writes_its_columns(self,
                                                     tmp_path):
        written = tmp_path / "empty.parquet"
        store._write_frame(frame().iloc[:0], written)

        assert list(store._read_frame(written).columns) == ["IDENTIFIER", "CLOSE", "VOLUME"]

    def test_is_byte_for_byte_reproducible(self,
                                           tmp_path,
                                           small_chunks):
        store._write_frame(frame(), tmp_path / "one.parquet")
        store._write_frame(frame(), tmp_path / "two.parquet")

        assert (tmp_path / "one.parquet").read_bytes() == (tmp_path / "two.parquet").read_bytes()

    def test_reports_each_chunk(self,
                                tmp_path,
                                small_chunks):
        seen: list[float] = []
        store._write_frame(frame(), tmp_path / "x.parquet", seen.append)

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
