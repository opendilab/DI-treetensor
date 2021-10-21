import torch
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
