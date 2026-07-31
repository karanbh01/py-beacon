# src/beacon/plot/base.py
"""
The lazy `.plot` accessor.

Every result object carries one, and none of them cost anything until it is
touched. That is the whole point of this module: it imports nothing beyond the
standard library, so `import beacon` stays as fast without matplotlib installed
as it was before plotting existed, and a user who never draws a chart never
pays for the ability to.

The mechanism is a descriptor. `IndexResult.plot` is an attribute lookup that
resolves, on first access, to a class in `beacon.plot.accessors` — which is
where matplotlib is required. Naming the accessor class as a *string* here is
what keeps the import out of the core: a real import at module scope would drag
matplotlib into `beacon.index.result`, and the guard would be decorative.
"""
from typing import Any


class PlotAccessor:
    """Descriptor that resolves to a result's plotting methods on first use.

    Attributes:
        accessor_name: Class in `beacon.plot.accessors` to instantiate, named
            rather than imported so this module stays free of matplotlib.
    """

    def __init__(self,
                 accessor_name: str):
        self.accessor_name = accessor_name

    def __set_name__(self,
                     owner: type,
                     name: str) -> None:
        self._attribute = name

    def __get__(self,
                instance: Any,
                owner: type | None = None) -> Any:
        # Accessed on the class rather than an instance — help(), Sphinx, a
        # `hasattr` probe. Returning the descriptor keeps those working without
        # importing matplotlib to answer them.
        if instance is None:
            return self

        from . import accessors

        accessor: type[Any] = getattr(accessors, self.accessor_name)

        return accessor(instance)


class ChartMethods:
    """Base for the accessor classes, providing the listing repr.

    A caller who types `result.plot` and presses enter should be told what they
    can do with it. Without this they get the default object repr, which
    answers a question nobody asked.
    """

    def __init__(self,
                 result: Any):
        self._result = result

    def methods(self) -> list[str]:
        """Chart methods this accessor offers, alphabetically."""
        return sorted(name for name in dir(type(self))
                      if not name.startswith("_") and name != "methods"
                      and callable(getattr(type(self), name, None)))

    def __repr__(self) -> str:
        """List the available charts."""
        available = ", ".join(f"{name}()" for name in self.methods())

        return f"<{type(self).__name__}: {available}>"
