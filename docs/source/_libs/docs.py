import io
from contextlib import contextmanager
from functools import partial
from typing import Optional, Tuple, List


def strip_docs(doc: Optional[str]) -> Tuple[str, List[str]]:
    _lines = (doc or '').splitlines()
    _non_empty_lines = sorted(filter(lambda t: t[1].strip(), enumerate(_lines)))
    if _non_empty_lines:
        _first_line_id, _ = _non_empty_lines[0]
        _lines = _lines[_first_line_id:]
    else:
        _lines = []
    _exist_lines = list(filter(str.strip, _lines))

    if not _exist_lines:
        _indent = ''
    else:
        l, r = 0, min(map(len, _exist_lines))
        while l < r:
            m = (l + r + 1) // 2
            _prefixes = list(map(lambda x: x[:m], _exist_lines))
            l, r = (m, r) if len(set(_prefixes)) <= 1 and not _prefixes[0].strip() else (l, m - 1)
        _indent = list(map(lambda x: x[:l], _exist_lines))[0]

    _stripped_lines = list(map(lambda x: x[len(_indent):] if x.strip() else '', _lines))
    return _indent, _stripped_lines


_DOC_FROM_TAG = '__doc_from__'


def get_origin(obj):
    return getattr(obj, _DOC_FROM_TAG, None)


def print_title(title: str, levelc='=', file=None):
    title = title.replace('_', '\\_')
    _print = partial(print, file=file)
    _print(title)
    _print(levelc * (len(title) + 5))
    _print()


def print_doc(doc: str, strip: bool = True, indent: str = '', file=None):
    _print = partial(print, indent, file=file, sep='')
    if strip:
        _, _lines = strip_docs(doc or '')
    else:
        _lines = (doc or '').splitlines()

    for _line in _lines:
        _print(_line)
    _print()


@contextmanager
def print_block(name: str, value: Optional[str] = None,
                params: Optional[dict] = None, file=None):
    _print = partial(print, file=file)
    _print(f'.. {name + "::" if name else ""} {str(value) if value is not None else ""}')
    for k, v in (params or {}).items():
        _print(f'    :{k}: {str(v) if v is not None else ""}')
    _print()

    with io.StringIO() as bf:
        try:
            yield bf
        finally:
            bf.flush()
            print_doc(bf.getvalue(), strip=True, indent='    ', file=file)


def current_module(module: str, file=None):
    with print_block('currentmodule', module, file=file):
        pass
