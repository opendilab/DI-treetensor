import numpy as np
import torch
from treevalue import method_treelize
from treevalue.utils import pre_process

from .size import Size
from ..common import TreeObject, TreeData, ireduce
from ..numpy import TreeNumpy
from ..utils import inherit_names, current_names, doc_from

__all__ = [
    'Tensor'
]

_reduce_tensor_wrap = pre_process(lambda it: ((torch.tensor([*it]),), {}))
tireduce = pre_process(lambda rfunc: ((_reduce_tensor_wrap(rfunc),), {}))(ireduce)


# noinspection PyTypeChecker,PyShadowingBuiltins,PyArgumentList
@current_names()
@inherit_names(TreeData)
class Tensor(TreeData):
    @doc_from(torch.Tensor.numpy)
    @method_treelize(return_type=TreeNumpy)
    def numpy(self: torch.Tensor) -> np.ndarray:
        return self.numpy()

    @doc_from(torch.Tensor.tolist)
    @method_treelize(return_type=TreeObject)
    def tolist(self: torch.Tensor):
        return self.tolist()

    @doc_from(torch.Tensor.cpu)
    @method_treelize()
    def cpu(self: torch.Tensor, *args, **kwargs):
        return self.cpu(*args, **kwargs)

    @doc_from(torch.Tensor.cuda)
    @method_treelize()
    def cuda(self: torch.Tensor, *args, **kwargs):
        return self.cuda(*args, **kwargs)

    @doc_from(torch.Tensor.to)
    @method_treelize()
    def to(self: torch.Tensor, *args, **kwargs):
        return self.to(*args, **kwargs)

    @doc_from(torch.Tensor.numel)
    @ireduce(sum)
    @method_treelize(return_type=TreeObject)
    def numel(self: torch.Tensor):
        return self.numel()

    @property
    @doc_from(torch.Tensor.shape)
    @method_treelize(return_type=Size)
    def shape(self: torch.Tensor):
        return self.shape

    @doc_from(torch.Tensor.all)
    @tireduce(torch.all)
    @method_treelize(return_type=TreeObject)
    def all(self: torch.Tensor, *args, **kwargs) -> bool:
        return self.all(*args, **kwargs)

    @doc_from(torch.Tensor.any)
    @tireduce(torch.any)
    @method_treelize(return_type=TreeObject)
    def any(self: torch.Tensor, *args, **kwargs) -> bool:
        return self.any(*args, **kwargs)

    @doc_from(torch.Tensor.max)
    @tireduce(torch.max)
    @method_treelize(return_type=TreeObject)
    def max(self: torch.Tensor, *args, **kwargs):
        return self.max(*args, **kwargs)

    @doc_from(torch.Tensor.min)
    @tireduce(torch.min)
    @method_treelize(return_type=TreeObject)
    def min(self: torch.Tensor, *args, **kwargs):
        return self.min(*args, **kwargs)

    @doc_from(torch.Tensor.sum)
    @tireduce(torch.sum)
    @method_treelize(return_type=TreeObject)
    def sum(self: torch.Tensor, *args, **kwargs):
        return self.sum(*args, **kwargs)
