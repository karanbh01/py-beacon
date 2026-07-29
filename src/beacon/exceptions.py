# src/beacon/exceptions.py
"""
Custom exceptions for the beacon package.
This helps in categorizing errors originating from the beacon package.
"""

class BeaconError(Exception):
    """Base exception class for all custom exceptions in the beacon package."""
    def __init__(self,
                 message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message

class DataNotFoundError(BeaconError):
    """Raised when specific financial data cannot be found or is unavailable."""
    def __init__(self,
                 data_description: str,
                 source: str = "N/A"):
        message = f"Data not found: {data_description}. (Source: {source})"
        super().__init__(message)
        self.data_description = data_description
        self.source = source

class InvalidRuleError(BeaconError):
    """Raised when an index methodology rule or backtest rule is invalid or
    improperly configured."""
    def __init__(self,
                 rule_description: str,
                 reason: str):
        message = f"Invalid rule: {rule_description}. Reason: {reason}"
        super().__init__(message)
        self.rule_description = rule_description
        self.reason = reason

class ConfigurationError(BeaconError):
    """Raised for errors related to package or module configuration."""
    def __init__(self,
                 config_param: str,
                 details: str):
        message = f"Configuration error for '{config_param}': {details}"
        super().__init__(message)
        self.config_param = config_param
        self.details = details

class MissingDependencyError(BeaconError, ImportError):
    """Raised when a feature is used without its optional dependency installed.

    Subclasses ImportError as well as BeaconError so that callers already
    handling a missing import keep working.
    """
    def __init__(self,
                 module_name: str,
                 feature: str,
                 extra: str):
        message = (f"{feature} requires the '{module_name}' package, which is "
                   f"not installed. Install it with: "
                   f'pip install "py-beacon[{extra}]"')
        super().__init__(message)
        self.module_name = module_name
        self.feature = feature
        self.extra = extra

class CalculationError(BeaconError):
    """Raised during financial calculations if an error occurs (e.g., division by
    zero, bad inputs)."""
    def __init__(self,
                 calculation_name: str,
                 details: str):
        message = f"Error in calculation '{calculation_name}': {details}"
        super().__init__(message)
        self.calculation_name = calculation_name
        self.details = details

# For the main __init__.py, they can be exposed directly:
# from .beacon_exceptions import DataNotFoundError, InvalidRuleError
# if beacon_exceptions.py is in root
