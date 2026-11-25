# tests/test_catalogue_columns.py
"""BN-455: every table statistic says what its columns are measured in:
those in different units list each column with its unit, and the rest give
one unit for them all. The columns listed are the ones each table gives.

The runs are test_strategy_futures' index held physically and through
futures, test_backtest_flows' ETF with money arriving and leaving,
test_cash_management's fund whose cash leaves its band, and
test_risk_contributions_over_time's thirty names for the risk model.
"""
import pandas as pd
import pytest

from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.catalogue.columns import COLUMNS
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.options import AnalysisOptions, RiskModelOptions
from beacon.analysis.reports.statistics import TABLE, Summarised

UNITS = {"fraction", "bps", "days", "money", "count", "ratio", "date", None}


@pytest.fixture(scope="module")
def contexts():
    from beacon.backtest import Backtest, DatedFlows
    from beacon.index.calculation import IndexCalculator
    from test_backtest_etf import etf
    from test_backtest_flows import DAYS, QUIET, run
    from test_cash_management import banded
    from test_factor_analytics import SOURCE, factors
    from test_risk_contributions_over_time import noisy
    from test_strategy_futures import DATA, END, START, futures, index
    from test_strategy_futures import run as futures_run

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)
    plain = Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                     transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                        end=END)
    listed = run(vehicle=etf(in_kind=True),
                 flows=DatedFlows({QUIET: 500_000.0, DAYS[60]: -300_000.0}))
    results = [plain, futures_run(futures()), listed, banded(0.03)]
    found = [Context(result.run_data, None, getattr(result, "_data_fetcher", None))
             for result in results]
    found.append(Context(noisy(), AnalysisOptions(risk_model=RiskModelOptions(window=100)),
                         SOURCE, factors()))

    return found


def test_every_table_says_its_units():
    vague = [statistic.id for statistic in CATALOGUE
             if statistic.kind == TABLE and not statistic.columns and statistic.unit is None]

    assert vague == []


def test_each_column_s_unit_is_one_known():
    for statistic in CATALOGUE:
        for column in statistic.columns:
            assert column.unit in UNITS, (statistic.id, column.name)


@pytest.mark.parametrize("statistic_id", sorted(COLUMNS))
def test_the_columns_listed_are_the_table_s(contexts,
                                             statistic_id):
    statistic = CATALOGUE[statistic_id]

    for context in contexts:
        try:
            value = statistic.compute(context)
        except Exception:  # This run cannot give it; another may.
            continue

        table = value.table if isinstance(value, Summarised) else value

        if isinstance(table, pd.DataFrame) and not table.empty:
            assert list(table.columns) == [column.name for column in statistic.columns]
            return

    # Share classes are measured in their own tests (test_dealing).
    assert statistic_id == "class_costs"
