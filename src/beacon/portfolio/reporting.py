# src/beacon/portfolio/reporting.py
"""
Module for generating reports from portfolio data, such as holdings reports
and performance reports, potentially in formats like Excel.
"""
import logging

import pandas as pd

from .._optional import require
from ..exceptions import ReportingError
from .base import Portfolio

logger = logging.getLogger(__name__)

# Re-exported so `from beacon.portfolio.reporting import ReportingError` keeps
# working; the class itself lives in beacon.exceptions with the rest.
__all__ = ["ReportGenerator", "ReportingError"]

class ReportGenerator:
    """
    Generates various reports for a portfolio or backtest results.

    Excel output needs the 'excel' extra; the dependency is checked when a
    report is generated, so the class itself is always constructible.
    """
    def generate_holdings_report_excel(self,
                                       portfolio: Portfolio,
                                       report_path: str,
                                       valuation_date: pd.Timestamp) -> None:
        """
        Generates an Excel report summarizing the current portfolio holdings.

        The report is built from the portfolio's own state, so the caller must
        have called ``portfolio.update_prices(...)`` beforehand for the holdings
        to carry current prices (and therefore current market values/weights).

        Args:
            portfolio: The Portfolio object to report on.
            report_path: The file path (including .xlsx extension) where the Excel
                         report will be saved.
            valuation_date: The date the holdings are reported as of; used for
                            logging and report labelling only.

        Raises:
            MissingDependencyError: If openpyxl is not installed.
            ReportingError: If there's an issue writing the file.
            ValueError: If portfolio is None.
        """
        require("openpyxl", "Excel reporting")

        if portfolio is None:
            raise ValueError("Portfolio object must be provided.")
        if not report_path.endswith(".xlsx"):
            logger.warning(f"Report path '{report_path}' does not end with .xlsx. Appending it.")
            report_path += ".xlsx"

        logger.info(
            f"Generating holdings report for portfolio '{portfolio.portfolio_id}' as of "
            f"{valuation_date.strftime('%Y-%m-%d')} to '{report_path}'.")

        try:
            holdings_summary_df = portfolio.get_holdings_summary()

            if holdings_summary_df.empty:
                logger.warning(
                    f"No holdings data to report for portfolio "
                    f"'{portfolio.portfolio_id}'. Excel file will be empty or not created.")
                # Create an empty sheet or just return
                # For now, let's write an empty DataFrame if that's the case.

            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                holdings_summary_df.to_excel(writer, sheet_name='HoldingsSummary', index=False)

                # You could add more sheets here, e.g., transaction history
                transactions_df = pd.DataFrame([vars(tx) for tx in portfolio.transactions])
                if not transactions_df.empty:
                    transactions_df = self._normalise_transactions_df(transactions_df)
                    transactions_df.to_excel(writer, sheet_name='TransactionHistory', index=False)

            logger.info(f"Holdings report successfully saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate or save holdings report to {report_path}: {e}")
            raise ReportingError(f"Error generating holdings report: {e}") from e

    def _normalise_transactions_df(self,
                                   transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Asset objects in transactions_df to string representations if needed."""
        if 'asset' in transactions_df.columns:
            transactions_df['asset_id'] = transactions_df['asset'].apply(
                lambda x: x.asset_id if hasattr(x, 'asset_id') else str(x)
            )
            # Drop original asset object column
            transactions_df.drop(columns=['asset'], inplace=True)
        return transactions_df


    def generate_performance_report_excel(self,
                                          # Output from BacktestEngine or analysis
                                          performance_data: pd.DataFrame,
                                          report_path: str,
                                          report_title: str | None = "Performance Report") -> None:
        """
        Generates an Excel report from a DataFrame of performance data.
        The performance_data DataFrame is typically the output of a backtest
        (e.g., daily portfolio values, returns) or specific analysis results.

        Args:
            performance_data: A pandas DataFrame containing performance metrics over time.
                              Expected to have a DatetimeIndex.
            report_path: The file path (including .xlsx extension) for the report.
            report_title: An optional title for the report (used as sheet name or in header).

        Raises:
            MissingDependencyError: If openpyxl is not installed.
            ReportingError: If there's an issue writing the file.
            ValueError: If performance_data is not a non-empty DataFrame.
        """
        require("openpyxl", "Excel reporting")

        if not isinstance(performance_data, pd.DataFrame) or performance_data.empty:
            raise ValueError("performance_data must be a non-empty pandas DataFrame.")
        if not report_path.endswith(".xlsx"):
            logger.warning(f"Report path '{report_path}' does not end with .xlsx. Appending it.")
            report_path += ".xlsx"

        # Excel sheet name limits
        sheet_name = report_title.replace(" ", "_")[:30] if report_title else "PerformanceData"
        logger.info(f"Generating performance report '{sheet_name}' to '{report_path}'.")

        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Assuming DatetimeIndex should be written
                performance_data.to_excel(writer, sheet_name=sheet_name, index=True)
            logger.info(f"Performance report successfully saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate or save performance report to {report_path}: {e}")
            raise ReportingError(f"Error generating performance report: {e}") from e
