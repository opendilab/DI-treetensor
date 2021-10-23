from functools import partial
from typing import Type

from hbutils.reflection import post_process
from treevalue import func_treelize as original_func_treelize
from treevalue import general_tree_value, TreeValue, typetrans
from treevalue.tree.common import TreeStorage

from ..utils import replaceable_partial, args_mapping

__all__ = [
    'BaseTreeStruct',
    'clsmeta', 'auto_tree',
]


class BaseTreeStruct(general_tree_value()):
    """
    Overview:
        Base structure of all the trees in ``treetensor``.
    """
    pass


def clsmeta(func, allow_dict: bool = False) -> Type[type]:
    """
    Overview:
        Create a metaclass based on generating function.

        Used in :py:class:`treetensor.common.Object`,
        :py:class:`treetensor.torch.Tensor` and :py:class:`treetensor.torch.Size`.
        Can do modify onto the constructor function of the classes.

    Arguments:
        - func: Generating function.
        - allow_dict (:obj:`bool`): Auto transform dictionary to :py:class:`treevalue.TreeValue` class, \
                                    default is ``False``.
    Returns:
        - metaclass (:obj:`Type[type]`): Metaclass for creating a new class.
    """

    class _TempTreeValue(TreeValue):
        pass

    def _mapping_func(_, x):
        if isinstance(x, TreeValue):
            return x
        elif isinstance(x, TreeStorage):
            return TreeValue(x)
        elif allow_dict and isinstance(x, dict):
            return TreeValue(x)
        else:
            return x

    func_treelize = post_process(post_process(args_mapping(_mapping_func)))(
        replaceable_partial(original_func_treelize, return_type=_TempTreeValue)
    )

    _wrapped_func = func_treelize()(func)

    class _MetaClass(type):
        def __call__(cls, data, *args, **kwargs):
            if isinstance(data, TreeStorage):
                return type.__call__(cls, data)
            elif isinstance(data, cls) and not args and not kwargs:
                return data

            _result = _wrapped_func(data, *args, **kwargs)
            if isinstance(_result, _TempTreeValue):
                return type.__call__(cls, _result)
            else:
                return _result

    return _MetaClass


def _auto_tree_func(t, cls):
    from .object import Object
    t = typetrans(t, return_type=Object)
    for key, value in cls:
        if isinstance(key, type):
            predict = lambda x: isinstance(x, key)
        elif callable(key):
            predict = lambda x: key(x)
        else:
            raise TypeError(f'Unknown type of prediction - {repr(key)}.')

        if t.map(predict).all():
            return typetrans(t, return_type=value)
    return t


# noinspection PyArgumentList
def auto_tree(v, cls):
    if isinstance(cls, type) and issubclass(cls, TreeValue):
        cls = partial(typetrans, return_type=cls)
    elif isinstance(cls, (list, tuple)):
        cls = partial(_auto_tree_func, cls=cls)
    elif callable(cls):
        pass
    else:
        raise TypeError(f'Unknown type of cls - {repr(cls)}.')

    if isinstance(v, TreeValue):
        return cls(v)
    elif isinstance(v, (tuple, list, set)):
        return type(v)((auto_tree(item, cls) for item in v))
    elif isinstance(v, dict):
        return type(v)({key: auto_tree(value, cls) for key, value in v.items()})
    else:
        return v
