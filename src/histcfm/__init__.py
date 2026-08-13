"""Public HistCFM package interface with a lazy model export."""

from typing import TYPE_CHECKING

from ._version import __version__


if TYPE_CHECKING:
    from .models import HistCFM as HistCFM


__all__ = ["HistCFM", "__version__"]


def __getattr__(name):
    if name == "HistCFM":
        from .models import HistCFM

        return HistCFM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
