import builtins
from types import ModuleType, FunctionType, BuiltinFunctionType
from typing import Iterable

import numpy as np

from .array import *
from .array import __all__ as _array_all
from .funcs import *
from .funcs import __all__ as _funcs_all
from .funcs import get_func_from_numpy
from ..config.meta import __VERSION__

__all__ = [
    *_funcs_all,
    *_array_all,
]

_basic_types = (
    builtins.bool, builtins.bytearray, builtins.bytes, builtins.complex, builtins.dict,
    builtins.float, builtins.frozenset, builtins.int, builtins.list, builtins.range, builtins.set,
    builtins.slice, builtins.str, builtins.tuple,
)
_np_all = set(np.__all__)


class _Module(ModuleType):
    def __init__(self, module):
        ModuleType.__init__(self, module.__name__)

        for name in filter(lambda x: x.startswith('__') and x.endswith('__'), dir(module)):
            setattr(self, name, getattr(module, name))
        self.__origin__ = module
        self.__numpy_version__ = np.__version__
        self.__version__ = __VERSION__

    def __getattr__(self, name):
        if (name in self.__all__) or \
                (hasattr(self.__origin__, name) and isinstance(getattr(self.__origin__, name), ModuleType)):
            return getattr(self.__origin__, name)
        else:
            item = getattr(np, name)
            if isinstance(item, (FunctionType, BuiltinFunctionType)) and not name.startswith('_'):
                return get_func_from_numpy(name)
            elif isinstance(item, _basic_types) and name in _np_all:
                return item
            else:
                raise AttributeError(f'Attribute {repr(name)} not found in {repr(__name__)}.')

    def __dir__(self) -> Iterable[str]:
        return self.__all__


import sys

sys.modules[__name__] = _Module(sys.modules[__name__])
