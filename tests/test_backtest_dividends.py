# tests/test_backtest_dividends.py
"""BN-263: a backtest is paid its holdings' cash distributions.

AAA grows 0.1% a day in total return. It pays 2.00 a share on EX, and its
price drops by exactly that on the ex-date, as an unadjusted feed's does. So
a total-return index of AAA grows 0.1% every day, and a book that reinvests
the dividend the day it arrives grows with it exactly.
"""
import numpy as np
import pandas as pd
import pytest

from beacon import ModellingAssumptions
from beacon.backtest import BacktestEngine
from beacon.data import pair_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.calculation.total_return import TOTAL_RETURN
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.portfolio.cash_flows import DISTRIBUTION, DIVIDEND
from beacon.testing.weights import index_result_from_weights

DAYS = pd.bdate_range("2024-01-02", "2024-03-28")
EX = DAYS[20]
CAPITAL = 1_000_000.0
DIVIDEND_PER_SHARE = 2.0


def total_return_path() -> np.ndarray:
    return 100.0 * 1.001 ** np.arange(len(DAYS))


def prices(flat: bool = False) -> np.ndarray:
    """The close: the total-return path, less the dividend from EX on.

    *flat* holds the price at 100 until the dividend and 98 after it.
    """
    if flat:
        return np.where(DAYS >= EX, 100.0 - DIVIDEND_PER_SHARE, 100.0)

    path = total_return_path()
    drop = 1.0 - DIVIDEND_PER_SHARE / path[DAYS.get_loc(EX)]

    return np.where(DAYS >= EX, path * drop, path)


def fetcher(flat: bool = False,
            pay_date: pd.Timestamp | None = None,
            status: str | None = None,
            currency: str = "USD") -> DataFetcher:
    rows = [{"IDENTIFIER": "AAA", "DATE": day, "CLOSE": price,
             "SHARES_OUTSTANDING": 1e6}
            for day, price in zip(DAYS, prices(flat), strict=True)]
    rows += [{"IDENTIFIER": "BBB", "DATE": day, "CLOSE": 50.0,
              "SHARES_OUTSTANDING": 1e6} for day in DAYS]

    if currency != "USD":
        rows += [{"IDENTIFIER": f"{currency}USD", "DATE": day,
                  "CLOSE": 1.25 + 0.001 * step}
                 for step, day in enumerate(DAYS)]

    action = {"IDENTIFIER": "AAA", "EX_DATE": EX, "TYPE": "DIVIDEND",
              "VALUE": DIVIDEND_PER_SHARE}
    if pay_date is not None:
        action["PAY_DATE"] = pay_date
    if status is not None:
        action["STATUS"] = status

    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01", "NAME": "A",
         "CURRENCY": currency, "EXCHANGE": "XNYS"},
        {"IDENTIFIER": "BBB", "DATE_FROM": "2020-01-01", "NAME": "B",
         "CURRENCY": "USD", "EXCHANGE": "XNYS"}])

    if currency != "USD":
        reference = pd.concat([reference,
                               pd.DataFrame([pair_reference(f"{currency}USD")])],
                              ignore_index=True)

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       CorporateActions.from_dataframe(pd.DataFrame([action])))


def total_return_index(data: DataFetcher):
    definition = IndexDefinition(index_id="tr", index_name="TR",
                                 base_date=str(DAYS[0].date()),
                                 base_value=100.0, currency="USD",
                                 eligibility_rules=[],
                                 weighting_scheme=EqualWeighted(),
                                 rebalancing_frequency="MONTHLY",
                                 calendar="XNYS", universe_identifiers=["AAA"],
                                 return_type=TOTAL_RETURN)

    return IndexCalculator(definition, data).run(end_date=str(DAYS[-1].date()))


def run(data: DataFetcher | None = None,
        index=None,
        **engine_args):
    data = data if data is not None else fetcher()
    index = index if index is not None else total_return_index(data)

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=CAPITAL, data_provider=data,
                          index_result=index, calendar="XNYS",
                          **engine_args).run()


def shares_at_ex() -> float:
    return CAPITAL / 100.0


def flows(result,
          kind: str) -> list:
    return [flow for flow in result.portfolio.cash_flows if flow.kind == kind]


class TestTrackingATotalReturnIndex:

    def test_reinvesting_the_day_it_arrives_tracks_it_exactly(self):
        data = fetcher()
        index = total_return_index(data)
        result = run(data, index, dividends="reinvest")

        levels = index.index_levels / index.index_levels.iloc[0]
        nav = result.trading_nav / CAPITAL

        np.testing.assert_allclose(nav.reindex(levels.index).to_numpy(),
                                   levels.to_numpy(), rtol=1e-9)

    def test_reinvesting_is_the_default(self):
        data = fetcher()
        index = total_return_index(data)

        assert run(data, index).trading_nav.equals(
            run(data, index, dividends="reinvest").trading_nav)

    def test_without_dividends_the_book_trails_by_the_yield(self):
        data = fetcher(status="cancelled")
        index = total_return_index(fetcher())
        result = run(data, index)

        book = result.trading_nav.iloc[-1] / CAPITAL
        level = index.index_levels.iloc[-1] / index.index_levels.iloc[0]

        # The price alone ends short of the total return by the dividend's
        # share of the price it was paid from.
        yield_at_ex = DIVIDEND_PER_SHARE / total_return_path()[DAYS.get_loc(EX)]

        assert book / level == pytest.approx(1.0 - yield_at_ex, rel=1e-9)


class TestReceivingTheCash:

    def test_the_holding_at_the_start_of_the_ex_date_is_paid(self):
        result = run()
        paid = flows(result, DIVIDEND)

        assert len(paid) == 1
        assert paid[0].date == EX
        assert paid[0].asset_id == "AAA"
        assert paid[0].amount == pytest.approx(shares_at_ex()
                                               * DIVIDEND_PER_SHARE)

    def test_kept_as_cash_there_is_no_trade_on_the_pay_date(self):
        result = run(dividends="cash")
        trades_on_ex = [trade for trade in result.portfolio.transactions
                        if trade.transaction_date == EX]

        assert trades_on_ex == []
        assert result.portfolio.cash.loc[EX] == pytest.approx(
            shares_at_ex() * DIVIDEND_PER_SHARE)

    def test_cash_arrives_on_the_pay_date(self):
        pay = DAYS[25]
        result = run(fetcher(pay_date=pay))

        assert [flow.date for flow in flows(result, DIVIDEND)] == [pay]
        assert result.portfolio.cash.loc[DAYS[22]] == pytest.approx(0.0,
                                                                   abs=1e-6)

    def test_a_holding_sold_before_the_pay_date_is_still_paid(self):
        """Entitled on the ex-date, whatever is held when the cash comes."""
        data = fetcher(pay_date=DAYS[30])
        index = index_result_from_weights({DAYS[0]: {"AAA": 1.0},
                                           DAYS[25]: {"BBB": 1.0}})
        result = run(data, index)

        assert "AAA" not in result.portfolio.holdings
        assert flows(result, DIVIDEND)[0].date == DAYS[30]

    def test_withholding_keeps_back_its_share(self):
        result = run(modelling_assumptions=ModellingAssumptions(
            withholding_tax_rate=0.15))

        assert flows(result, DIVIDEND)[0].amount == pytest.approx(
            shares_at_ex() * DIVIDEND_PER_SHARE * 0.85)

    def test_a_cancelled_dividend_is_not_paid(self):
        assert flows(run(fetcher(status="cancelled")), DIVIDEND) == []

    def test_a_foreign_dividend_is_converted_on_the_day_it_arrives(self):
        data = fetcher(currency="GBP")
        index = index_result_from_weights({DAYS[0]: {"AAA": 1.0}})
        result = run(data, index)
        rate = data.fx_rate_on("GBP", "USD", EX)
        bought = result.portfolio.transactions[0].quantity

        assert flows(result, DIVIDEND)[0].amount == pytest.approx(
            bought * DIVIDEND_PER_SHARE * rate)


class TestDistributing:

    def test_it_is_paid_out_of_the_book(self):
        result = run(dividends="distribute")
        paid_out = flows(result, DISTRIBUTION)

        assert len(paid_out) == 1
        assert paid_out[0].amount == pytest.approx(
            -shares_at_ex() * DIVIDEND_PER_SHARE)

    def test_performance_still_counts_it(self):
        """Flat prices but for the drop: the price alone falls 2%, the total
        return does not move, whichever way the dividend is handled."""
        data = fetcher(flat=True)
        index = index_result_from_weights({DAYS[0]: {"AAA": 1.0}})

        distributed = run(data, index, dividends="distribute").summary()
        accumulated = run(data, index).summary()

        assert distributed["total_return"] == pytest.approx(0.0, abs=1e-9)
        assert accumulated["total_return"] == pytest.approx(0.0, abs=1e-9)
        assert distributed["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="dividend policy"):
        run(dividends="sometimes")


class TestWhereReinvestmentGoes:
    """BN-343: a dividend is reinvested in the name that paid it by default,
    or across the holdings in proportion to value."""

    def bought_on_the_payment_day(self,
                                  **engine_args) -> dict[str, float]:
        index = index_result_from_weights({DAYS[0]: {"AAA": 0.5, "BBB": 0.5}})
        result = run(index=index, **engine_args)

        return {trade.asset_id: trade.quantity * trade.price
                for trade in result.portfolio.transactions
                if trade.transaction_date == EX and trade.transaction_type == "BUY"}

    def test_the_payer_by_default(self):
        bought = self.bought_on_the_payment_day()

        assert list(bought) == ["AAA"]
        assert bought["AAA"] == pytest.approx(shares_at_ex() / 2 * DIVIDEND_PER_SHARE)

    def test_the_holdings_when_asked(self):
        bought = self.bought_on_the_payment_day(reinvest_into="holdings")

        assert sorted(bought) == ["AAA", "BBB"]
        assert sum(bought.values()) == pytest.approx(shares_at_ex() / 2 * DIVIDEND_PER_SHARE)

    def test_an_unknown_target_is_refused(self):
        with pytest.raises(ValueError, match="reinvestment"):
            run(reinvest_into="everything")


class TestWhatTheBookIsOwed:
    """BN-356: from the ex-date the NAV includes the dividend the book is
    owed, so it does not dip and recover against an index that reinvests it
    on the ex-date."""

    PAY = DAYS[25]

    def test_on_the_ex_date_the_book_is_the_index(self):
        data = fetcher(pay_date=self.PAY)
        index = total_return_index(data)
        result = run(data, index)

        levels = index.index_levels / index.index_levels.iloc[0]
        nav = result.trading_nav / CAPITAL

        assert nav.loc[EX] == pytest.approx(levels.loc[EX], rel=1e-9)

    def test_it_does_not_dip_before_the_pay_date(self):
        """Until the cash comes the owed dividend earns nothing, where the
        index has it reinvested, so the two part only by that."""
        data = fetcher(pay_date=self.PAY)
        index = total_return_index(data)
        nav = run(data, index).trading_nav / CAPITAL
        levels = (index.index_levels / index.index_levels.iloc[0]).reindex(nav.index)

        gap = (nav / levels - 1.0).loc[EX:self.PAY]

        assert gap.abs().max() < 1e-3

    def test_it_is_owed_from_the_ex_date_until_paid(self):
        result = run(fetcher(pay_date=self.PAY))
        owed = result.dividends_receivable

        assert owed.loc[:DAYS[19]].eq(0.0).all()
        assert owed.loc[EX:DAYS[24]].to_numpy() == pytest.approx(
            shares_at_ex() * DIVIDEND_PER_SHARE)
        assert owed.loc[self.PAY:].eq(0.0).all()

    def test_net_of_withholding(self):
        result = run(fetcher(pay_date=self.PAY), modelling_assumptions=ModellingAssumptions(
            withholding_tax_rate=0.15))

        assert result.dividends_receivable.loc[EX] == pytest.approx(
            shares_at_ex() * DIVIDEND_PER_SHARE * 0.85)

    def test_nothing_is_bought_with_it_before_it_is_paid(self):
        result = run(fetcher(pay_date=self.PAY))
        buys = {trade.transaction_date for trade in result.portfolio.transactions
                if trade.transaction_type == "BUY" and trade.transaction_date > DAYS[0]}

        assert buys == {self.PAY}
