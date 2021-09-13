import torch
from treevalue import func_treelize as original_func_treelize

from ..common import TreeObject
from ..utils import replaceable_partial

func_treelize = replaceable_partial(original_func_treelize)

__all__ = [
    'Size'
]


# noinspection PyTypeChecker
class Size(TreeObject):
    @func_treelize(return_type=TreeObject)
    def numel(self: torch.Size) -> TreeObject:
        return self.numel()

    @func_treelize(return_type=TreeObject)
    def index(self: torch.Size, *args, **kwargs) -> TreeObject:
        return self.index(*args, **kwargs)

    @func_treelize(return_type=TreeObject)
    def count(self: torch.Size, *args, **kwargs) -> TreeObject:
        return self.count(*args, **kwargs)
