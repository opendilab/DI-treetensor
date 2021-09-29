import torch
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import BaseTree
from treevalue.utils import post_process

from ..tensor import Tensor
from ...common import auto_tree, module_func_loader
from ...utils import doc_from_base as original_doc_from_base
from ...utils import replaceable_partial, args_mapping

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, BaseTree, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize, return_type=Tensor)
)
doc_from_base = replaceable_partial(original_doc_from_base, base=torch)
auto_tensor = replaceable_partial(auto_tree, cls=Tensor)
get_func_from_torch = module_func_loader(torch, Tensor, '.'.join(__name__.split('.')[:-1]))
