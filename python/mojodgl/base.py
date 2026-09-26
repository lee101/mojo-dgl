"""Base data types shared across the API, mirroring `dgl/base.py`."""

from __future__ import annotations

__all__ = ["ALL", "EID", "ETYPE", "NID", "NTYPE", "DGLError", "is_all"]


class DGLError(RuntimeError):
    """The error DGL raises; kept so callers can catch it by name."""


class _AllType:
    """Sentinel meaning "every edge" / "every node"."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ALL"

    def __bool__(self) -> bool:
        return False


ALL = _AllType()
EID = ALL
ETYPE = ALL
NID = ALL
NTYPE = ALL


def is_all(data) -> bool:
    """True when `data` is the ALL sentinel rather than an index array."""
    return isinstance(data, _AllType)
