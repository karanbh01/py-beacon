# beacon/portfolio/reporting.py
"""
Module for generating reports from portfolio data, such as holdings reports
and performance reports, potentially in formats like Excel.
"""
import pandas as pd
from typing import TYPE_CHECKING, Optional
import logging

# Try importing openpyxl for Excel writing
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    logging.warning("openpyxl library not found. Excel reporting will be disabled.")
    OPENPYXL_AVAILABLE = False

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from .base import Portfolio
    from ..data.fetcher import DataFetcher # For portfolio methods needing it

logger = logging.getLogger(__name__)

class ReportingError(Exception):
    """Custom exception for errors during report generation."""
    pass

class ReportGenerator:
    """
    Generates various reports for a portfolio or backtest results.
    """
    def __init__(self):
        """Initializes the ReportGenerator."""
        if not OPENPYXL_AVAILABLE:
            logger.warning("ReportGenerator initialized, but openpyxl is missing. Excel output will fail.")

    def generate_holdings_report_excel(self,
                                       portfolio: 'Portfolio',
                                       report_path: str,
                                       valuation_date: pd.Timestamp,
                                       data_provider: 'DataFetcher') -> None:
        """
        Generates an Excel report summarizing the current portfolio holdings.

        Args:
            portfolio: The Portfolio object to report on.
            report_path: The file path (including .xlsx extension) where the Excel report will be saved.
            valuation_date: The date for which to value the holdings.
            data_provider: DataFetcher instance needed by portfolio methods to get current prices.

        Raises:
            ReportingError: If openpyxl is not available or if there's an issue writing the file.
            ValueError: If portfolio or data_provider is None.
        """
        if not OPENPYXL_AVAILABLE:
            raise ReportingError("openpyxl library is required for Excel reports but not installed.")
        if portfolio is None:
            raise ValueError("Portfolio object must be provided.")
        if data_provider is None:
            raise ValueError("DataFetcher object must be provided for holdings valuation.")
        if not report_path.endswith(".xlsx"):
            logger.warning(f"Report path '{report_path}' does not end with .xlsx. Appending it.")
            report_path += ".xlsx"

        logger.info(f"Generating holdings report for portfolio '{portfolio.portfolio_id}' as of {valuation_date.strftime('%Y-%m-%d')} to '{report_path}'.")

        try:
            holdings_summary_df = portfolio.get_holdings_summary(data_provider, valuation_date)

            if holdings_summary_df.empty:
                logger.warning(f"No holdings data to report for portfolio '{portfolio.portfolio_id}'. Excel file will be empty or not created.")
                # Create an empty sheet or just return
                # For now, let's write an empty DataFrame if that's the case.
            
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                holdings_summary_df.to_excel(writer, sheet_name='HoldingsSummary', index=False)
                
                # You could add more sheets here, e.g., transaction history
                transactions_df = pd.DataFrame([vars(tx) for tx in portfolio.transactions])
                if not transactions_df.empty:
                    # Convert Asset objects in transactions_df to string representations if needed
                    if 'asset' in transactions_df.columns:
                         transactions_df['asset_id'] = transactions_df['asset'].apply(lambda x: x.asset_id if hasattr(x, 'asset_id') else str(x))
                         transactions_df.drop(columns=['asset'], inplace=True) # Drop original asset object column
                    transactions_df.to_excel(writer, sheet_name='TransactionHistory', index=False)

            logger.info(f"Holdings report successfully saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate or save holdings report to {report_path}: {e}")
            raise ReportingError(f"Error generating holdings report: {e}")


    def generate_performance_report_excel(self,
                                          performance_data: pd.DataFrame, # Output from BacktestEngine or analysis
                                          report_path: str,
                                          report_title: Optional[str] = "Performance Report") -> None:
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
            ReportingError: If openpyxl is not available or if there's an issue writing the file.
            ValueError: If performance_data is not a non-empty DataFrame.
        """
        if not OPENPYXL_AVAILABLE:
            raise ReportingError("openpyxl library is required for Excel reports but not installed.")
        if not isinstance(performance_data, pd.DataFrame) or performance_data.empty:
            raise ValueError("performance_data must be a non-empty pandas DataFrame.")
        if not report_path.endswith(".xlsx"):
            logger.warning(f"Report path '{report_path}' does not end with .xlsx. Appending it.")
            report_path += ".xlsx"

        sheet_name = report_title.replace(" ", "_")[:30] if report_title else "PerformanceData" # Excel sheet name limits
        logger.info(f"Generating performance report '{sheet_name}' to '{report_path}'.")

        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                performance_data.to_excel(writer, sheet_name=sheet_name, index=True) # Assuming DatetimeIndex should be written
            logger.info(f"Performance report successfully saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate or save performance report to {report_path}: {e}")
            raise ReportingError(f"Error generating performance report: {e}")