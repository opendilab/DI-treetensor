import builtins

import numpy as np
from treevalue import func_treelize as original_func_treelize

from .numpy import TreeNumpy
from ..common import ireduce, TreeObject
from ..utils import replaceable_partial

__all__ = [
    'all', 'any',
    'equal', 'array_equal',
]

func_treelize = replaceable_partial(original_func_treelize, return_type=TreeNumpy)


@ireduce(builtins.all)
@func_treelize(return_type=TreeObject)
def all(a, *args, **kwargs):
    return np.all(a, *args, **kwargs)


@ireduce(builtins.any)
@func_treelize()
def any(a, *args, **kwargs):
    return np.any(a, *args, **kwargs)


@func_treelize()
def equal(x1, x2, *args, **kwargs):
    return np.equal(x1, x2, *args, **kwargs)


@func_treelize()
def array_equal(a1, a2, *args, **kwargs):
    return np.array_equal(a1, a2, *args, **kwargs)
