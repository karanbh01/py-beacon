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

class ExpressionError(BeaconError):
    """Raised when an expression is built or used in a way that cannot work.

    Most often `bool(expression)` — Python evaluates `and`, `or` and `not` by
    calling `__bool__`, and an expression has no truth value until it is
    resolved against an instrument and a date. Returning `True` there would
    make `(a == 1) and (b > 2)` silently discard half the expression, so it
    raises instead.
    """


class UnknownDatasetError(ExpressionError, AttributeError):
    """Raised when an expression names a dataset that does not exist.

    Also an `AttributeError`, because it is raised from `__getattr__` and the
    language builds on that: `hasattr` and `getattr(..., default)` catch
    `AttributeError` and nothing else, so raising only a `BeaconError` would
    make `hasattr(data, "typo")` blow up instead of answering False.
    """


class InvalidIdentifierError(BeaconError, ValueError):
    """Raised when a caller supplies an identifier that cannot be used.

    Subclasses `ValueError` as well as `BeaconError`, on the same principle as
    `MissingDependencyError` above: a caller already writing
    ``except ValueError`` around a store operation keeps working, because a
    rejected identifier *is* a value error. The API still answers 422 rather
    than the generic argument handler, because `BeaconError` precedes
    `ValueError` in the MRO and the handler lookup walks it in order.

    Distinct from `DataNotFoundError`, which means the identifier was fine and
    nothing was stored under it. This means the identifier itself is
    unusable — empty, or containing path separators — so there is nothing to
    look for.

    The distinction matters because it decides the status code. A document id
    arrives from a URL path parameter, so rejecting one is a statement about
    the *request*, and answering 500 would tell a client the server had
    broken when in fact it had correctly refused bad input. That is exactly
    what happened until BN-131: the path-traversal guard below worked, said so
    clearly, and returned it as an internal error.
    """
    def __init__(self,
                 identifier: str,
                 reason: str):
        # Truncated rather than echoed whole. The value came from a URL and
        # may be long or hostile, and a client needs enough to recognise which
        # id it sent rather than the entire string returned to it.
        shown = identifier if len(identifier) <= 40 else f"{identifier[:40]}..."

        super().__init__(f"Invalid identifier '{shown}': {reason}")
        self.identifier = shown
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

class ReportingError(BeaconError):
    """Raised when a report cannot be generated or written.

    A BeaconError rather than a bare Exception so it reaches a client through
    the API's error envelope with a stable code, like every other library
    failure, instead of as an unlabelled 500.
    """
    def __init__(self,
                 details: str):
        super().__init__(f"Reporting failed: {details}")
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
