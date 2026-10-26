# tests/test_analysis_panels.py
"""BN-361: a run's large tables, kept as Parquet files beside its record.

Synthetic tables only: wide frames of weights with gaps, written to a
temporary folder and read back.
"""
import json

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.panels import (
    FUND_WEIGHTS,
    MANIFEST_FILE,
    PANELS_VERSION,
    read_manifest,
    read_panels,
    run_panels,
    write_panels,
)
from beacon.exceptions import ConfigurationError, DataNotFoundError

DATES = pd.bdate_range("2024-01-02", periods=60)
NAMES = ["AAA", "BBB", "CCC", "DDD"]


def weights(seed: int = 3,
            index_name: str | None = None) -> pd.DataFrame:
    """Daily weights in four names, with some names not held on some days."""
    random = np.random.default_rng(seed)
    values = random.random((len(DATES), len(NAMES)))
    values[random.random(values.shape) < 0.3] = np.nan
    frame = pd.DataFrame(values, index=DATES, columns=NAMES)
    frame.index.name = index_name

    return frame


class TestRoundTrip:

    def test_a_table_reads_back_exactly(self,
                                        tmp_path):
        table = weights()
        write_panels(tmp_path / "run", {FUND_WEIGHTS: table})

        back = read_panels(tmp_path / "run")[FUND_WEIGHTS]

        np.testing.assert_array_equal(back.to_numpy(), table.to_numpy())
        assert list(back.columns) == NAMES
        assert back.index.equals(table.index)
        assert back.index.name is None

    def test_the_index_keeps_its_name(self,
                                      tmp_path):
        write_panels(tmp_path / "run", {"named": weights(index_name="DATE")})

        assert read_panels(tmp_path / "run")["named"].index.name == "DATE"

    def test_several_tables_and_a_chosen_few(self,
                                             tmp_path):
        write_panels(tmp_path / "run", {"first": weights(1), "second": weights(2)})

        assert set(read_panels(tmp_path / "run")) == {"first", "second"}
        assert set(read_panels(tmp_path / "run", ["second"])) == {"second"}

    def test_the_manifest_says_what_is_there(self,
                                             tmp_path):
        write_panels(tmp_path / "run", {FUND_WEIGHTS: weights()})

        manifest = read_manifest(tmp_path / "run")

        assert manifest == {"version": PANELS_VERSION,
                            "panels": {FUND_WEIGHTS: {"rows": len(DATES), "columns": 4,
                                                      "index": None}}}

    def test_writing_again_replaces_every_table(self,
                                                tmp_path):
        write_panels(tmp_path / "run", {"old": weights(1)})
        write_panels(tmp_path / "run", {"new": weights(2)})

        assert set(read_panels(tmp_path / "run")) == {"new"}
        assert not (tmp_path / "run" / "old.parquet").exists()
        assert [path.name for path in tmp_path.iterdir()] == ["run"]


class TestWithoutTables:

    def test_a_record_without_a_folder_has_no_tables(self,
                                                     tmp_path):
        assert read_manifest(tmp_path / "never") is None
        assert read_panels(tmp_path / "never") == {}

    def test_a_table_not_there_is_named(self,
                                        tmp_path):
        write_panels(tmp_path / "run", {FUND_WEIGHTS: weights()})

        with pytest.raises(DataNotFoundError, match="name_returns"):
            read_panels(tmp_path / "run", ["name_returns"])


class TestRefusals:

    @pytest.mark.parametrize("name", ["Fund", "fund weights", "../x", ""])
    def test_a_name_that_is_not_a_plain_file_name(self,
                                                  tmp_path,
                                                  name):
        with pytest.raises(ConfigurationError):
            write_panels(tmp_path / "run", {name: weights()})

    def test_columns_must_be_text(self,
                                  tmp_path):
        with pytest.raises(ConfigurationError, match="text"):
            write_panels(tmp_path / "run", {"numbers": weights().set_axis([1, 2, 3, 4], axis=1)})

    def test_a_refused_write_leaves_the_tables_as_they_were(self,
                                                            tmp_path):
        write_panels(tmp_path / "run", {FUND_WEIGHTS: weights()})

        with pytest.raises(ConfigurationError):
            write_panels(tmp_path / "run", {"Bad": weights()})

        assert set(read_panels(tmp_path / "run")) == {FUND_WEIGHTS}

    def test_a_newer_layout_is_refused(self,
                                       tmp_path):
        write_panels(tmp_path / "run", {FUND_WEIGHTS: weights()})
        manifest = tmp_path / "run" / MANIFEST_FILE
        manifest.write_text(json.dumps({"version": PANELS_VERSION + 1, "panels": {}}),
                            encoding="utf-8")

        with pytest.raises(ConfigurationError, match="newer"):
            read_panels(tmp_path / "run")


class TestARunsTables:

    def test_a_run_keeps_its_fund_s_daily_weights(self,
                                                  tmp_path):
        from test_strategy_futures import index, run

        result = run(index())
        write_panels(tmp_path / "run", run_panels(result))

        back = read_panels(tmp_path / "run")[FUND_WEIGHTS]
        expected = result.portfolio.weights

        np.testing.assert_array_equal(back.to_numpy(), expected.to_numpy())
        assert list(back.columns) == [str(name) for name in expected.columns]
        assert back.index.equals(expected.index)
