from functools import wraps

import torch
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import BaseTree
from treevalue.utils import post_process

from ..base import auto_torch
from ..tensor import Tensor
from ...common import return_self
from ...utils import doc_from_base as original_doc_from_base
from ...utils import replaceable_partial, args_mapping

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, BaseTree, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize, return_type=Tensor)
)
doc_from_base = replaceable_partial(original_doc_from_base, base=torch)
auto_tensor = replaceable_partial(auto_torch, cls=Tensor)

_funcs_module = '.'.join(__name__.split('.')[:-1])


def get_func_from_torch(name):
    func = getattr(torch, name)
    return_self_dec = return_self if func.__name__.endswith("_") else (lambda x: x)

    @doc_from_base()
    @return_self_dec
    @post_process(auto_tensor)
    @func_treelize(return_type=TreeValue, rise=True)
    @wraps(func, assigned=('__name__',), updated=())
    def _new_func(*args, **kwargs):
        return func(*args, **kwargs)

    _new_func.__qualname__ = _new_func.__name__
    _new_func.__module__ = _funcs_module
    return _new_func
