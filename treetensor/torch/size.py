import torch
from treevalue import func_treelize as original_func_treelize

from .base import TreeTorch
from ..common import TreeObject
from ..utils import replaceable_partial, doc_from, current_names

func_treelize = replaceable_partial(original_func_treelize)

__all__ = [
    'Size'
]


# noinspection PyTypeChecker
@current_names()
class Size(TreeTorch):
    @doc_from(torch.Size.numel)
    @func_treelize(return_type=TreeObject)
    def numel(self: torch.Size) -> TreeObject:
        return self.numel()

    @doc_from(torch.Size.index)
    @func_treelize(return_type=TreeObject)
    def index(self: torch.Size, *args, **kwargs) -> TreeObject:
        return self.index(*args, **kwargs)

    @doc_from(torch.Size.count)
    @func_treelize(return_type=TreeObject)
    def count(self: torch.Size, *args, **kwargs) -> TreeObject:
        return self.count(*args, **kwargs)
