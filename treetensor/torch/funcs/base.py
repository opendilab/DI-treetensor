from functools import wraps

import torch
from hbutils.testing import vpip
from treevalue import func_treelize as original_func_treelize

from ..tensor import Tensor
from ...common import auto_tree, module_func_loader
from ...utils import doc_from_base as original_doc_from_base
from ...utils import replaceable_partial

func_treelize = replaceable_partial(original_func_treelize, return_type=Tensor)
doc_from_base = replaceable_partial(original_doc_from_base, base=torch)
auto_tensor = replaceable_partial(auto_tree, cls=[(torch.is_tensor, Tensor)])
get_func_from_torch = module_func_loader(torch, Tensor,
                                         [(torch.is_tensor, Tensor)])

_is_torch_2 = vpip('torch') >= '2'


def wrap_for_treelize(*args, **kwargs):
    def _decorator(func):
        @wraps(func)
        def _new_func(*args_, **kwargs_):
            retval = func(*args_, **kwargs_)
            return func_treelize(*args, **kwargs)(retval)

        return _new_func

    return _decorator
