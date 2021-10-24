from functools import wraps
from operator import itemgetter

from treevalue import TreeValue, walk

__all__ = [
    'ireduce',
    'return_self',
]


def ireduce(rfunc, piter=None):
    piter = piter or (lambda x: x)

    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, TreeValue):
                it = map(itemgetter(1), walk(result, include_nodes=False))
                return rfunc(piter(it))
            else:
                return result

        return _new_func

    return _decorator


def return_self(func):
    @wraps(func)
    def _new_func(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return _new_func
