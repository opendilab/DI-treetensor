import builtins
from functools import lru_cache
from types import ModuleType, FunctionType, BuiltinFunctionType
from typing import Iterable

import torch

from .funcs import *
from .funcs import __all__ as _funcs_all
from .funcs.base import get_func_from_torch
from .size import *
from .size import __all__ as _size_all
from .stream import *
from .stream import __all__ as _stream_all
from .tensor import *
from .tensor import __all__ as _tensor_all
from ..config.meta import __VERSION__

__all__ = [
    *_funcs_all,
    *_size_all,
    *_tensor_all,
    *_stream_all,
]

_basic_types = (
    builtins.bool, builtins.bytearray, builtins.bytes, builtins.complex, builtins.dict,
    builtins.float, builtins.frozenset, builtins.int, builtins.list, builtins.range, builtins.set,
    builtins.slice, builtins.str, builtins.tuple,
)
_torch_all = set(torch.__all__)


class _Module(ModuleType):
    def __init__(self, module):
        ModuleType.__init__(self, module.__name__)

        for name in filter(lambda x: x.startswith('__') and x.endswith('__'), dir(module)):
            setattr(self, name, getattr(module, name))
        self.__origin__ = module
        self.__torch_version__ = torch.__version__
        self.__version__ = __VERSION__

    @lru_cache()
    def __getattr__(self, name):
        if (name in self.__all__) or \
                (hasattr(self.__origin__, name) and isinstance(getattr(self.__origin__, name), ModuleType)):
            return getattr(self.__origin__, name)
        else:
            item = getattr(torch, name)
            if isinstance(item, (FunctionType, BuiltinFunctionType)) and not name.startswith('_'):
                return get_func_from_torch(name)
            elif (isinstance(item, torch.dtype)) or \
                    isinstance(item, _basic_types) and name in _torch_all:
                return item
            else:
                raise AttributeError(f'Attribute {repr(name)} not found in {repr(__name__)}.')

    def __dir__(self) -> Iterable[str]:
        return self.__all__


import sys

sys.modules[__name__] = _Module(sys.modules[__name__])
