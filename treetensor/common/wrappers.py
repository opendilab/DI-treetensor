from collections import namedtuple
from functools import wraps
from itertools import chain

from treevalue import TreeValue
from treevalue import reduce_ as treevalue_reduce

__all__ = [
    'kwreduce', 'ireduce', 'vreduce',
    'return_self',
]


def kwreduce(rfunc):
    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            _result = func(*args, **kwargs)
            if isinstance(_result, TreeValue):
                return treevalue_reduce(_result, rfunc)
            else:
                return _result

        return _new_func

    return _decorator


def vreduce(rfunc):
    return kwreduce(lambda **kws: rfunc(kws.values()))


def ireduce(rfunc, piter=None):
    _IterReduceWrapper = namedtuple("_IterReduceWrapper", ['v'])
    piter = piter or (lambda x: x)

    def _reduce_func(values):
        _list = []
        for item in values:
            if isinstance(item, _IterReduceWrapper):
                _list.append(item.v)
            else:
                _list.append([item])
        return _IterReduceWrapper(chain(*_list))

    def _decorator(func):
        rifunc = vreduce(_reduce_func)(func)

        @wraps(func)
        def _new_func(*args, **kwargs):
            _iw = rifunc(*args, **kwargs)
            if isinstance(_iw, _IterReduceWrapper):
                return rfunc(piter(_iw.v))
            else:
                return _iw

        return _new_func

    return _decorator


def return_self(func):
    @wraps(func)
    def _new_func(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    return _new_func
