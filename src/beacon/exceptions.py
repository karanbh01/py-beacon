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

    Most often `bool(expression)`. Python evaluates `and`, `or` and `not` by
    calling `__bool__`, and an expression has no truth value until it is
    resolved against an instrument and a date. Returning `True` there would
    make `(a == 1) and (b > 2)` silently discard half the expression, so it
    raises instead. Combine expressions with `&`, `|` and `~`.
    """


class UnknownDatasetError(ExpressionError, AttributeError):
    """Raised when an expression names a dataset that does not exist.

    Also an `AttributeError`, because it is raised from `__getattr__` and the
    language builds on that: `hasattr` and `getattr(..., default)` catch
    `AttributeError` and nothing else, so raising only a `BeaconError` would
    make `hasattr(data, "typo")` blow up instead of answering False.
    """


class DataSourceError(BeaconError):
    """Raised when a read needs a data source and the process has none.

    The message always names both fixes, `beacon.use(fetcher)` and
    generating the default store, because "no data" discovered deep inside a
    price lookup is useless without being told what to do about it.
    """


# Until BN-131 a rejected document id reached the client as a 500: the
# path-traversal guard worked, said so clearly, and was returned as an
# internal error. This class exists so the API answers 422 instead.
class InvalidIdentifierError(BeaconError, ValueError):
    """Raised when a caller supplies an identifier that cannot be used.

    Subclasses `ValueError` as well as `BeaconError`, on the same principle as
    `MissingDependencyError`: a caller already writing ``except ValueError``
    around a store operation keeps working, because a rejected identifier *is*
    a value error. The API still answers 422 rather than using the generic
    argument handler, because `BeaconError` precedes `ValueError` in the MRO
    and the handler lookup walks it in order.

    Distinct from `DataNotFoundError`, which means the identifier was fine and
    nothing was stored under it. This means the identifier itself is
    unusable (empty, or containing path separators), so there is nothing to
    look for.

    The distinction decides the status code. A document id arrives from a
    URL path parameter, so rejecting one is a statement about the *request*,
    not a server fault.

    The identifier is truncated to 40 characters in the message and in the
    `identifier` attribute.
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

# BN-236.
class NoDataLoadedError(BeaconError):
    """Raised when something needs market data and none is loaded.

    The engine can run with no data: it starts empty until a data store is
    loaded, and everything that does not read data keeps working. This is the
    answer for everything that does. It is a state the caller can change by
    loading a store, not a server fault, so it maps to 409, not 500.

    Args:
        purpose: What could not be done, completing "No data is loaded, so
            ...", e.g. "a backtest cannot be run".
    """
    def __init__(self,
                 purpose: str):
        self.purpose = purpose

        super().__init__(f"No data is loaded, so {purpose}. Load a data "
                         f"store first.")


# BN-201. Before it, a document from a newer build and a damaged file were
# reported as one number, "could not be read", which invites restoring a file
# that has nothing wrong with it.
class DocumentFromNewerBuildError(ConfigurationError):
    """A stored document written by a newer py-beacon than this one.

    A `ConfigurationError`, so every handler for that still catches it, and a
    subclass, so a listing can tell it apart from a damaged file. The two call
    for opposite responses: nothing is wrong with this file, and the remedy is
    to upgrade the engine reading it. Two installs reach this state when the
    engine and the app are updated on different machines.
    """


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
                   f'pip install "py-beacon-kit[{extra}]"')
        super().__init__(message)
        self.module_name = module_name
        self.feature = feature
        self.extra = extra

class FrozenPortfolioError(BeaconError):
    """Raised when something tries to write to a portfolio that is closed.

    A finished backtest freezes its portfolio, because that portfolio *is* the
    record of the run: applying another trade to it would quietly restate a
    result someone has already read. Continuing a strategy means seeding a new
    run from the old end state, not mutating the record.

    Frozen is a state, not a subclass: a hand-built portfolio is never frozen
    unless its owner freezes it.
    """
    def __init__(self,
                 portfolio_id: str,
                 operation: str):
        message = (f"Portfolio '{portfolio_id}' is frozen: it is the record of a "
                   f"finished backtest, so '{operation}' cannot change it. Seed a "
                   f"new run from its end state to continue.")
        super().__init__(message)
        self.portfolio_id = portfolio_id
        self.operation = operation

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

# The two shared one published code until BN-194, so a client heading
# `CALCULATION_ERROR` with "the engine refused to answer" was told a decision
# had been made when in fact something broke. That is a wrong remedy, which
# costs more than no remedy: the reader goes looking for what to change,
# there is nothing, and the real signal (a stack trace worth reporting) is
# disguised as a considered answer.
#
# What separates them on the wire is the published code alone, never the
# `WeightingScheme-` prefix that one `except` block happens to put in
# `calculation_name`, which is an implementation detail.
class UnexpectedCalculationError(CalculationError):
    """Raised when a calculation *crashed*, as opposed to refusing.

    Every other `CalculationError` is a deliberate refusal: a guard that names
    what was missing and what to do about it. This one is the opposite: an
    exception nobody anticipated, caught at a boundary and re-raised so it
    still reaches a client inside the error envelope, with its own published
    code, instead of as an unlabelled 500. It means there is nothing in the
    request to change, and the failure is worth reporting.

    A subclass rather than a sibling, because a crash during a calculation
    genuinely *is* a calculation error: anything already written as
    ``except CalculationError`` keeps catching it.

    `original_type` carries the class name of the exception that actually
    failed, so a reader can tell a `ZeroDivisionError` from a `KeyError`
    without a server log. Like every other attribute of a `BeaconError`, it
    reaches the client in the envelope's `detail`.
    """
    def __init__(self,
                 calculation_name: str,
                 cause: BaseException):
        self.original_type = type(cause).__name__

        # `BeaconError.__init__` rather than the parent's, because the parent's
        # "Error in calculation '<name>': <details>" reads as a considered
        # answer, which is the confusion this class exists to end. The two
        # attributes it would have set are set below instead, so the shape a
        # client sees stays the parent's plus one field.
        BeaconError.__init__(
            self,
            f"Calculation '{calculation_name}' failed with an unexpected "
            f"{self.original_type}: {cause}. This is a fault rather than a "
            f"refusal: there is nothing in the request to change, and it is "
            f"worth reporting.")

        self.calculation_name = calculation_name
        self.details = str(cause)
