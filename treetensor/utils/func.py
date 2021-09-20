"""
Functional Decorators.
"""
from functools import wraps
from typing import Callable, Union, Any

__all__ = [
    'replaceable_partial',
    'args_mapping',
]


def replaceable_partial(func, **kws):
    @wraps(func)
    def _new_func(*args, **kwargs):
        return func(*args, **{**kws, **kwargs})

    return _new_func


def args_mapping(mapper: Callable[[Union[int, str], Any], Any]):
    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            return func(
                *(mapper(i, x) for i, x in enumerate(args)),
                **{k: mapper(k, v) for k, v in kwargs.items()},
            )

        return _new_func

    return _decorator
