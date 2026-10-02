# src/beacon/__init__.py
"""
py-beacon, an end-to-end toolkit for index, ETF, and Delta-1 derivatives
development.
"""
__version__ = "0.4.0"

from . import derivatives
from .assumptions import (
    ModellingAssumptions,
    current_modelling_assumptions,
    use_modelling_assumptions,
)
from .derivatives import (
    DerivativeBase,
    ETFFuture,
    IndexFuture,
    TotalReturnSwap,
)
from .sources import use

__all__ = [
    "DerivativeBase",
    "ETFFuture",
    "IndexFuture",
    "ModellingAssumptions",
    "TotalReturnSwap",
    "__version__",
    "current_modelling_assumptions",
    "derivatives",
    "use",
    "use_modelling_assumptions",
]
