import os
from typing import List, Optional, Callable, Any

__all__ = [
    'inherit_doc', 'direct_doc',
]


def _strip_lines(doc: Optional[str]):
    _lines = (doc or '').splitlines()
    _first_line_id, _ = sorted(filter(lambda t: t[1].strip(), enumerate(_lines)))[0]
    _lines = _lines[_first_line_id:]
    _exist_lines = list(filter(str.strip, _lines))

    if not _exist_lines:
        _indent = ''
    else:
        l, r = 0, min(map(len, _exist_lines))
        while l < r:
            m = (l + r + 1) // 2
            _prefixes = set(map(lambda x: x[:m], _exist_lines))
            l, r = (m, r) if len(_prefixes) <= 1 else (l, m - 1)
        _indent = list(map(lambda x: x[:l], _exist_lines))[0]

    _stripped_lines = list(map(lambda x: x[len(_indent):] if x.strip() else '', _lines))
    return _indent, _stripped_lines


def _unstrip_lines(indent: str, stripped_lines: List[str]) -> str:
    return os.linesep.join(map(lambda x: indent + x, stripped_lines))


def inherit_doc(src, stripper: Optional[Callable[[Any, Any, List[str]], List[str]]] = None):
    _indent, _stripped_lines = _strip_lines(src.__doc__)

    def _decorator(obj):
        _lines = (stripper or (lambda s, o, x: x))(src, obj, _stripped_lines)
        obj.__doc__ = _unstrip_lines(_indent, _lines)
        return obj

    return _decorator


_DIRECT_DOC = '__direct_doc__'


def direct_doc(obj):
    setattr(obj, _DIRECT_DOC, True)
    return obj
