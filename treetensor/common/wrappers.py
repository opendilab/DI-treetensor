from functools import wraps

from treevalue import TreeValue
from treevalue import reduce_ as treevalue_reduce


def kwreduce(reduce_func):
    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            _result = func(*args, **kwargs)
            if isinstance(_result, TreeValue):
                return treevalue_reduce(_result, reduce_func)
            else:
                return _result

        return _new_func

    return _decorator


def vreduce(vreduce_func):
    return kwreduce(lambda **kws: vreduce_func(kws.values()))
