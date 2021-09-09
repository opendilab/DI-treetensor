from functools import partial, wraps
from typing import Tuple

import torch
from treevalue import func_treelize, TreeValue

from .tensor import TreeTensor
from ..common import vreduce

_treelize = partial(func_treelize, return_type=TreeTensor)
_python_all = all


def _size_based_treelize(*args_, prefix: bool = False, tuple_: bool = False, **kwargs_):
    def _decorator(func):
        @_treelize(*args_, **kwargs_)
        def _sub_func(size: Tuple[int, ...], *args, **kwargs):
            _size_args = (size,) if tuple_ else size
            _args = (*args, *_size_args) if prefix else (*_size_args, *args)
            return func(*_args, **kwargs)

        @wraps(func)
        def _new_func(size, *args, **kwargs):
            if isinstance(size, (TreeValue, dict)):
                size = TreeTensor(size)
            return _sub_func(size, *args, **kwargs)

        return _new_func

    return _decorator


# Tensor generation based on shapes
zeros = _size_based_treelize()(torch.zeros)
randn = _size_based_treelize()(torch.randn)
randint = _size_based_treelize(prefix=True, tuple_=True)(torch.randint)
ones = _size_based_treelize()(torch.ones)
full = _size_based_treelize(tuple_=True)(torch.full)
empty = _size_based_treelize()(torch.empty)

# Tensor generation based on another tensor
zeros_like = _treelize()(torch.zeros_like)
randn_like = _treelize()(torch.randn_like)
randint_like = _treelize()(torch.randint_like)
ones_like = _treelize()(torch.ones_like)
full_like = _treelize()(torch.full_like)
empty_like = _treelize()(torch.empty_like)

# Tensor operators
all = vreduce(all)(_treelize()(torch.all))
eq = _treelize()(torch.eq)
equal = _treelize()(torch.equal)
