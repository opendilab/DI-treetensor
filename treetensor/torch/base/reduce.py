import warnings
from functools import wraps
from typing import Optional

import torch

from ...common import ireduce

__all__ = ['rmreduce', 'post_reduce', 'auto_reduce']


def _reduce_func(rfunc):
    rfunc = rfunc or (lambda x: x)

    def _new_func(ts):
        return rfunc(torch.cat(tuple(map(lambda x: x.view((-1,)), ts))))

    return _new_func


def rmreduce(rfunc=None):
    return ireduce(_reduce_func(rfunc))


def post_reduce(rfunc=None, prefunc=None):
    rfunc = rfunc or (lambda x, *args, **kwargs: x)

    def _decorator(func):
        func = rmreduce(prefunc)(func)

        # noinspection PyUnusedLocal,PyShadowingBuiltins
        @wraps(func)
        def _new_func(input, *args, **kwargs):
            result = func(input, *args, **kwargs)
            return rfunc(result, *args, **kwargs)

        return _new_func

    return _decorator


# noinspection PyUnusedLocal
def _default_auto_determine(*args, out=None, **kwargs):
    return False if args or kwargs else None


# noinspection PyUnusedLocal
def _default_auto_condition(*args, out=None, **kwargs):
    return not args and not kwargs


def auto_reduce(rfunc, nrfunc, determine=None, condition=None):
    determine = determine or _default_auto_determine
    condition = condition or _default_auto_condition

    def _decorator(func):
        # noinspection PyUnusedLocal,PyShadowingBuiltins
        @wraps(func)
        def _new_func(input, *args, reduce: Optional[bool] = None, **kwargs):
            _determine = determine(*args, **kwargs)
            if _determine is not None:
                if reduce is not None:
                    if not _determine and reduce:
                        warnings.warn(UserWarning(
                            f'Reduce forbidden for this case of function {func.__name__}, '
                            f'enablement of reduce option will be ignored.'), stacklevel=2)
                    elif _determine and not reduce:
                        warnings.warn(UserWarning(
                            f'Reduce must be processed for this case of function {func.__name__}, '
                            f'disablement of reduce option will be ignored.'), stacklevel=2)
                reduce = not not _determine

            _reduce = condition(*args, **kwargs) if reduce is None else not not reduce
            return (rfunc if _reduce else nrfunc)(input, *args, **kwargs)

        return _new_func

    return _decorator
