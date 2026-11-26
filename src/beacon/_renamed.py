# src/beacon/_renamed.py
"""
Names renamed in 0.11.0, which keep working, warning, until 0.12.0.

    __getattr__ = renamed_names(__name__, {"Book": ("PortfolioRecord", PortfolioRecord)})

    @renamed_keyword("book", "of")
    def weights(self, of: str = "fund"): ...

A module's old names are served by its `__getattr__`, so importing the
module does not warn and reading an old name does. A function's old keyword
is passed on as its new one.
"""
# BN-462 (Karan, 2026-11-24): prose says portfolio, not book, and the public
# names follow.
import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

REMOVED_IN = "0.12.0"


def renamed_names(module: str,
                  names: dict[str, tuple[str, Any]]) -> Callable[[str], Any]:
    """A `__getattr__` for *module* serving each old name in *names* as the
    object it was renamed to, warning."""
    def read(name: str) -> Any:
        if name not in names:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")

        new, found = names[name]
        warnings.warn(f"{module}.{name} is deprecated and will be removed in {REMOVED_IN}; "
                      f"use {new}.", DeprecationWarning, stacklevel=2)

        return found

    return read


def renamed_keyword(old: str,
                    new: str) -> Callable[[F], F]:
    """A function whose keyword *old* is now *new*: given as *old*, it warns
    and is passed as *new*."""
    def wrap(function: F) -> F:
        @functools.wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{function.__qualname__}() takes {new}= or {old}=, not "
                                    f"both.")

                warnings.warn(f"{function.__qualname__}({old}=) is deprecated and will be "
                              f"removed in {REMOVED_IN}; use {new}=.", DeprecationWarning,
                              stacklevel=2)
                kwargs[new] = kwargs.pop(old)

            return function(*args, **kwargs)

        return call  # type: ignore[return-value]

    return wrap


__all__ = ["renamed_keyword", "renamed_names"]
