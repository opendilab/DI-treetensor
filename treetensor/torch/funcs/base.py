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


def get_func_from_torch(name):
    func = getattr(torch, name)

    @func_treelize()
    @wraps(func)
    def _new_func(*args, **kwargs):
        return func(*args, **kwargs)

    if func.__name__.endswith("_"):
        _new_func = return_self(_new_func)
    _new_func = doc_from_base()(_new_func)

    return _new_func
