import builtins

import numpy as np
from hbutils.reflection import post_process
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import TreeStorage

from .array import ndarray
from ..common import ireduce, Object, module_func_loader
from ..utils import replaceable_partial, doc_from, args_mapping

__all__ = [
    'all', 'any', 'array',
    'equal', 'array_equal',
    'stack', 'concatenate', 'split',
    'zeros', 'ones',
]

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, TreeStorage, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize, return_type=ndarray)
)
get_func_from_numpy = module_func_loader(np, ndarray,
                                         [(np.ndarray, ndarray)])


@doc_from(np.all)
@ireduce(builtins.all)
@func_treelize(return_type=Object)
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


@doc_from(np.stack)
@func_treelize(subside=True)
def stack(arrays, *args, **kwargs):
    return np.stack(arrays, *args, **kwargs)


@doc_from(np.concatenate)
@func_treelize(subside=True)
def concatenate(arrays, *args, **kwargs):
    return np.concatenate(arrays, *args, **kwargs)


@doc_from(np.split)
@func_treelize(rise=True)
def split(ary, *args, **kwargs):
    return np.split(ary, *args, **kwargs)


@doc_from(np.zeros)
@func_treelize()
def zeros(shape, *args, **kwargs):
    return np.zeros(shape, *args, **kwargs)


@doc_from(np.ones)
@func_treelize()
def ones(shape, *args, **kwargs):
    return np.ones(shape, *args, **kwargs)
