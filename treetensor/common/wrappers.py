from functools import wraps

from treevalue import TreeValue, flatten_values

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
                it = flatten_values(result)
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
