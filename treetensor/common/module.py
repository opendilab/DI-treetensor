from functools import wraps
from typing import Type

from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import BaseTree
from treevalue.utils import post_process

from .trees import auto_tree
from .wrappers import return_self
from ..utils import doc_from_base as original_doc_from_base
from ..utils import replaceable_partial, args_mapping

__all__ = [
    'module_func_loader',
]


def module_func_loader(base, cls: Type[TreeValue], module_name: str):
    func_treelize = post_process(post_process(args_mapping(
        lambda i, x: TreeValue(x) if isinstance(x, (dict, BaseTree, TreeValue)) else x)))(
        replaceable_partial(original_func_treelize, return_type=cls)
    )
    doc_from_base = replaceable_partial(original_doc_from_base, base=base)
    auto_tree_cls = replaceable_partial(auto_tree, cls=cls)

    def _load_func(name):
        func = getattr(base, name)
        return_self_dec = return_self if func.__name__.endswith("_") else (lambda x: x)

        @doc_from_base()
        @return_self_dec
        @post_process(auto_tree_cls)
        @func_treelize(return_type=TreeValue, rise=True)
        @wraps(func, assigned=('__name__',), updated=())
        def _new_func(*args, **kwargs):
            return func(*args, **kwargs)

        _new_func.__qualname__ = _new_func.__name__
        _new_func.__module__ = module_name
        return _new_func

    return _load_func
