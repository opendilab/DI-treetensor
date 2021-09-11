import numpy as np
import torch
from treevalue import method_treelize
from treevalue.utils import pre_process

from .size import TreeSize
from ..common import TreeObject, TreeData, ireduce
from ..numpy import TreeNumpy

__all__ = [
    'TreeTensor'
]

_reduce_tensor_wrap = pre_process(lambda it: ((torch.tensor([*it]),), {}))
tireduce = pre_process(lambda rfunc: ((_reduce_tensor_wrap(rfunc),), {}))(ireduce)


# noinspection PyTypeChecker,PyShadowingBuiltins,PyArgumentList
class TreeTensor(TreeData):
    @method_treelize(return_type=TreeNumpy)
    def numpy(self: torch.Tensor) -> np.ndarray:
        return self.numpy()

    @method_treelize(return_type=TreeObject)
    def tolist(self: torch.Tensor):
        return self.tolist()

    @method_treelize()
    def cpu(self: torch.Tensor, *args, **kwargs):
        return self.cpu(*args, **kwargs)

    @method_treelize()
    def cuda(self: torch.Tensor, *args, **kwargs):
        return self.cuda(*args, **kwargs)

    @method_treelize()
    def to(self: torch.Tensor, *args, **kwargs):
        return self.to(*args, **kwargs)

    @ireduce(sum)
    @method_treelize(return_type=TreeObject)
    def numel(self: torch.Tensor):
        return self.numel()

    @property
    @method_treelize(return_type=TreeSize)
    def shape(self: torch.Tensor):
        return self.shape

    @tireduce(torch.all)
    @method_treelize(return_type=TreeObject)
    def all(self: torch.Tensor, *args, **kwargs) -> bool:
        return self.all(*args, **kwargs)

    @tireduce(torch.any)
    @method_treelize(return_type=TreeObject)
    def any(self: torch.Tensor, *args, **kwargs) -> bool:
        return self.any(*args, **kwargs)

    @tireduce(torch.max)
    @method_treelize(return_type=TreeObject)
    def max(self: torch.Tensor, *args, **kwargs):
        return self.max(*args, **kwargs)

    @tireduce(torch.min)
    @method_treelize(return_type=TreeObject)
    def min(self: torch.Tensor, *args, **kwargs):
        return self.min(*args, **kwargs)

    @tireduce(torch.sum)
    @method_treelize(return_type=TreeObject)
    def sum(self: torch.Tensor, *args, **kwargs):
        return self.sum(*args, **kwargs)
