import numpy as np
import torch
from treevalue import method_treelize, TreeValue

from .size import TreeSize
from ..common import TreeObject, TreeData
from ..numpy import TreeNumpy


def _same_merge(eq, hash_, **kwargs):
    kws = {
        key: value for key, value in kwargs.items()
        if not (isinstance(value, TreeValue) and not value)
    }

    class _Wrapper:
        def __init__(self, v):
            self.v = v

        def __hash__(self):
            return hash_(self.v)

        def __eq__(self, other):
            return eq(self.v, other.v)

    if len(set(_Wrapper(v) for v in kws.values())) == 1:
        return list(kws.values())[0]
    else:
        return TreeTensor(kws)


# noinspection PyTypeChecker,PyShadowingBuiltins
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

    @method_treelize(return_type=TreeObject)
    def numel(self: torch.Tensor):
        return self.numel()

    @property
    @method_treelize(return_type=TreeSize)
    def shape(self: torch.Tensor):
        return self.shape
