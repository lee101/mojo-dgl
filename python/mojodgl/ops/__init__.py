"""dgl operator module."""

from . import edge_softmax as _es_mod
from . import gather_mm as _gm_mod
from . import sddmm as _sddmm_mod
from . import segment as _seg_mod
from . import spmm as _spmm_mod

# The star imports below shadow the `edge_softmax` and `gather_mm` module names
# with the same-named functions, exactly as upstream does.
from .edge_softmax import *
from .gather_mm import *
from .sddmm import *
from .segment import *
from .spmm import *

__all__ = (
    _es_mod.__all__
    + _gm_mod.__all__
    + _sddmm_mod.__all__
    + _seg_mod.__all__
    + _spmm_mod.__all__
)
