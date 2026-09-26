"""mojodgl — DGL's graph message-passing kernels, implemented in Mojo.

The names are DGL's: `dgl.graph`, `dgl.ops`, `dgl.function`, `g.update_all`
with `fn.copy_u('h', 'm')` and `fn.sum('m', 'h')` all mean what they mean in
upstream. Features cross into Mojo by address: a C-contiguous float64 numpy
array is never copied.
"""

from . import function, nn, ops
from ._lib import build
from .base import ALL, EID, ETYPE, NID, NTYPE, DGLError
from .convert import graph
from .function import *  # noqa: F401,F403
from .heterograph import DGLGraph
from .ops import *  # noqa: F401,F403
from .udf import EdgeBatch, NodeBatch

__version__ = "0.1.0"

__all__ = [
    "DGLGraph", "graph", "DGLError", "ALL", "EID", "ETYPE", "NID", "NTYPE",
    "EdgeBatch", "NodeBatch", "build",
    "function", "ops", "nn",
] + ops.__all__
