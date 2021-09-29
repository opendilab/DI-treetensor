import pytest

_TEST_PREFIX = 'test_'


def get_mark_with_existence_check(base, name):
    _mark = pytest.mark.unittest if hasattr(base, name) else pytest.mark.ignore
    return _mark


def choose_mark_with_existence_check(base, name: str = None):
    def _decorator(func):
        _name = name or func.__name__[len(_TEST_PREFIX):]
        _mark = get_mark_with_existence_check(base, _name)
        return _mark(func)

    return _decorator
