import builtins
from typing import List

import numpy as np
from treevalue import func_treelize as original_func_treelize

from .numpy import TreeNumpy
from ..common import ireduce, TreeObject
from ..utils import replaceable_partial, inherit_doc

__all__ = [
    'all', 'any',
    'equal', 'array_equal',
]


def _doc_stripper(src, _, lines: List[str]):
    _name, _version = src.__name__, np.__version__
    _short_version = '.'.join(_version.split('.')[:2])
    return [
        f'.. note::',
        f'',
        f'    This documentation is based on '
        f'    `numpy.{_name} <https://numpy.org/doc/{_short_version}/reference/generated/numpy.{_name}.html>`_ '
        f'    in `numpy v{_version} <https://numpy.org/doc/{_short_version}/>`_.',
        f'    **Its arguments\' arrangements depend on the version of numpy you installed**.',
        f'',
        *lines,
    ]


func_treelize = replaceable_partial(original_func_treelize, return_type=TreeNumpy)
docs = replaceable_partial(inherit_doc, stripper=_doc_stripper)


@docs(np.all)
@ireduce(builtins.all)
@func_treelize(return_type=TreeObject)
def all(a, *args, **kwargs):
    return np.all(a, *args, **kwargs)


@docs(np.any)
@ireduce(builtins.any)
@func_treelize()
def any(a, *args, **kwargs):
    return np.any(a, *args, **kwargs)


@docs(np.equal)
@func_treelize()
def equal(x1, x2, *args, **kwargs):
    return np.equal(x1, x2, *args, **kwargs)


@docs(np.array_equal)
@func_treelize()
def array_equal(a1, a2, *args, **kwargs):
    return np.array_equal(a1, a2, *args, **kwargs)
