"""
Documentation Decorators.
"""
from .reflection import removed

__all__ = [
    'doc_from', 'doc_from_base',
]

_DOC_FROM_TAG = '__doc_from__'


def doc_from(src):
    def _decorator(obj):
        setattr(obj, _DOC_FROM_TAG, src)
        return obj

    return _decorator


def doc_from_base(base, name: str = None):
    def _decorator(func):
        _name = name or func.__name__
        if hasattr(base, _name):
            func = doc_from(getattr(base, _name))(func)
        else:
            func = removed(func)
        return func

    return _decorator
