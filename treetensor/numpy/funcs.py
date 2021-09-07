from functools import partial

import numpy as np
from treevalue import func_treelize, TreeValue

from .numpy import TreeNumpy

_treelize = partial(func_treelize, return_type=TreeNumpy)

equal = _treelize()(np.equal)
array_equal = _treelize()(np.array_equal)


def all_array_equal(tx, ty, *args, **kwargs) -> bool:
    _result = array_equal(tx, ty, *args, **kwargs)
    if isinstance(tx, TreeValue) and isinstance(ty, TreeValue):
        return _result.reduce(lambda **kws: all(list(kws.values())))
    else:
        return _result
