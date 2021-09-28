import sys

from .autograd import *
from .autograd import __all__ as _autograd_all
from .comparison import *
from .comparison import __all__ as _comparison_all
from .construct import *
from .construct import __all__ as _construct_all
from .math import *
from .math import __all__ as _math_all
from .matrix import *
from .matrix import __all__ as _matrix_all
from .operation import *
from .operation import __all__ as _operation_all
from .reduction import *
from .reduction import __all__ as _reduction_all
from ...utils import module_autoremove

__all__ = [
    *_autograd_all,
    *_comparison_all,
    *_construct_all,
    *_math_all,
    *_matrix_all,
    *_operation_all,
    *_reduction_all,
]

_current_module = sys.modules[__name__]
_current_module = module_autoremove(_current_module)
sys.modules[__name__] = _current_module
