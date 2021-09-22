import pytest

_TEST_PREFIX = 'test_'


def choose_mark_with_existence_check(base, name: str = None):
    def _decorator(func):
        _name = name or func.__name__[len(_TEST_PREFIX):]
        _mark = pytest.mark.unittest if hasattr(base, _name) else pytest.mark.ignore

        func = _mark(func)
        return func

    return _decorator
