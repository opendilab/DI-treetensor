import numpy as np
import torch
from treevalue import method_treelize
from treevalue.utils import pre_process

from .size import TreeSize
from ..common import TreeObject, TreeData, ireduce
from ..numpy import TreeNumpy

_reduce_tensor_wrap = pre_process(lambda it: ((torch.tensor([*it]),), {}))


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

    @ireduce(_reduce_tensor_wrap(torch.all))
    @method_treelize(return_type=TreeObject)
    def all(self: torch.Tensor, *args, **kwargs) -> bool:
        return self.all(*args, **kwargs)

    @ireduce(_reduce_tensor_wrap(torch.any))
    @method_treelize(return_type=TreeObject)
    def any(self: torch.Tensor, *args, **kwargs) -> bool:
        return self.any(*args, **kwargs)

    @ireduce(_reduce_tensor_wrap(torch.max))
    @method_treelize(return_type=TreeObject)
    def max(self: torch.Tensor, *args, **kwargs):
        return self.max(*args, **kwargs)

    @ireduce(_reduce_tensor_wrap(torch.min))
    @method_treelize(return_type=TreeObject)
    def min(self: torch.Tensor, *args, **kwargs):
        return self.min(*args, **kwargs)

    @ireduce(_reduce_tensor_wrap(torch.sum))
    @method_treelize(return_type=TreeObject)
    def sum(self: torch.Tensor, *args, **kwargs):
        return self.sum(*args, **kwargs)
