import builtins

import numpy as np
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.utils import post_process

from .array import ndarray
from ..common import ireduce, TreeObject
from ..utils import replaceable_partial, doc_from, args_mapping

__all__ = [
    'all', 'any', 'array',
    'equal', 'array_equal',
]

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize, return_type=ndarray)
)


@doc_from(np.all)
@ireduce(builtins.all)
@func_treelize(return_type=TreeObject)
def all(a, *args, **kwargs):
    return np.all(a, *args, **kwargs)


@doc_from(np.any)
@ireduce(builtins.any)
@func_treelize()
def any(a, *args, **kwargs):
    return np.any(a, *args, **kwargs)


@doc_from(np.equal)
@func_treelize()
def equal(x1, x2, *args, **kwargs):
    return np.equal(x1, x2, *args, **kwargs)


@doc_from(np.array_equal)
@func_treelize()
def array_equal(a1, a2, *args, **kwargs):
    return np.array_equal(a1, a2, *args, **kwargs)


@doc_from(np.array)
@func_treelize()
def array(p_object, *args, **kwargs):
    """
    In ``treetensor``, you can create a tree of :class:`np.ndarray` with :func:`array`.

    Examples::

        >>> import numpy as np
        >>> import treetensor.numpy as tnp
        >>> tnp.array({
        >>>     'a': [1, 2, 3],
        >>>     'b': [[4, 5], [5, 6]],
        >>>     'c': True,
        >>> })
        tnp.ndarray({
            'a': np.array([1, 2, 3]),
            'b': np.array([[4, 5], [5, 6]]),
            'c': np.array(True),
        })
    """
    return np.array(p_object, *args, **kwargs)
