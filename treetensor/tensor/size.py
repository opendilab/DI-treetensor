import torch
from treevalue import func_treelize

from ..common import BaseTreeStruct, TreeObject


# noinspection PyTypeChecker
class TreeSize(BaseTreeStruct):
    @func_treelize(return_type=TreeObject)
    def numel(self: torch.Size) -> TreeObject:
        return self.numel()

    @func_treelize(return_type=TreeObject)
    def index(self: torch.Size, *args, **kwargs) -> TreeObject:
        return self.index(*args, **kwargs)

    @func_treelize(return_type=TreeObject)
    def count(self: torch.Size, *args, **kwargs) -> TreeObject:
        return self.count(*args, **kwargs)
