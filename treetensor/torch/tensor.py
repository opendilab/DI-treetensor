"""
Overview:
    ``Tensor`` class, based on ``torch`` module.
"""

import numpy as np
import torch
from treevalue import method_treelize
from treevalue.utils import pre_process

from .base import TreeTorch
from .size import Size
from ..common import TreeObject, ireduce
from ..numpy import ndarray
from ..utils import current_names, doc_from

__all__ = [
    'Tensor'
]

_reduce_tensor_wrap = pre_process(lambda it: ((torch.tensor([*it]),), {}))
tireduce = pre_process(lambda rfunc: ((_reduce_tensor_wrap(rfunc),), {}))(ireduce)


# noinspection PyTypeChecker,PyShadowingBuiltins,PyArgumentList
@current_names()
class Tensor(TreeTorch):
    @doc_from(torch.Tensor.numpy)
    @method_treelize(return_type=ndarray)
    def numpy(self: torch.Tensor) -> np.ndarray:
        """
        Returns ``self`` tree tensor as a NumPy ``ndarray``.
        This tensor and the returned :class:`treetensor.numpy.ndarray` share the same underlying storage.
        Changes to self tensor will be reflected in the ``ndarray`` and vice versa.
        """
        return self.numpy()

    @doc_from(torch.Tensor.tolist)
    @method_treelize(return_type=TreeObject)
    def tolist(self: torch.Tensor):
        """
        Get the dump result of tree tensor.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.tensor({
            >>>     'a': [[1, 2], [3, 4]],
            >>>     'b': [1, 2, 3],
            >>>     'c': True,
            >>> }).tolist()
            TreeObject({
                'a': [[1, 2], [3, 4]],
                'b': [1, 2, 3],
                'c': True,
            })
        """
        return self.tolist()

    @doc_from(torch.Tensor.cpu)
    @method_treelize()
    def cpu(self: torch.Tensor, *args, **kwargs):
        """
        Returns a copy of this tree tensor in CPU memory.

        If this tree tensor is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return self.cpu(*args, **kwargs)

    @doc_from(torch.Tensor.cuda)
    @method_treelize()
    def cuda(self: torch.Tensor, *args, **kwargs):
        """
        Returns a copy of this tree tensor in CUDA memory.

        If this tree tensor is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return self.cuda(*args, **kwargs)

    @doc_from(torch.Tensor.to)
    @method_treelize()
    def to(self: torch.Tensor, *args, **kwargs):
        """
        Turn the original tree tensor to another format.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.tensor({
            ...     'a': [[1, 11], [2, 22], [3, 33]],
            ...     'b': {'x': [[4, 5], [6, 7]]},
            ... }).to(torch.float64)
            <Tensor 0x7ff363bb6518>
            ├── a --> tensor([[ 1., 11.],
            │                 [ 2., 22.],
            │                 [ 3., 33.]], dtype=torch.float64)
            └── b --> <Tensor 0x7ff363bb6ef0>
                └── x --> tensor([[4., 5.],
                                  [6., 7.]], dtype=torch.float64)
        """
        return self.to(*args, **kwargs)

    @doc_from(torch.Tensor.numel)
    @ireduce(sum)
    @method_treelize(return_type=TreeObject)
    def numel(self: torch.Tensor):
        """
        See :func:`treetensor.torch.numel`
        """
        return self.numel()

    @property
    @doc_from(torch.Tensor.shape)
    @method_treelize(return_type=Size)
    def shape(self: torch.Tensor):
        """
        Get the size of the tensors in the tree.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.tensor({
            ...     'a': [[1, 11], [2, 22], [3, 33]],
            ...     'b': {'x': [[4, 5], [6, 7]]},
            ... }).shape
            <Size 0x7ff363bbbd68>
            ├── a --> torch.Size([3, 2])
            └── b --> <Size 0x7ff363bbbcf8>
                └── x --> torch.Size([2, 2])
        """
        return self.shape

    @doc_from(torch.Tensor.all)
    @tireduce(torch.all)
    @method_treelize(return_type=TreeObject)
    def all(self: torch.Tensor, *args, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.all`
        """
        return self.all(*args, **kwargs)

    @doc_from(torch.Tensor.any)
    @tireduce(torch.any)
    @method_treelize(return_type=TreeObject)
    def any(self: torch.Tensor, *args, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.any`
        """
        return self.any(*args, **kwargs)

    @doc_from(torch.Tensor.max)
    @tireduce(torch.max)
    @method_treelize(return_type=TreeObject)
    def max(self: torch.Tensor, *args, **kwargs):
        """
        See :func:`treetensor.torch.max`
        """
        return self.max(*args, **kwargs)

    @doc_from(torch.Tensor.min)
    @tireduce(torch.min)
    @method_treelize(return_type=TreeObject)
    def min(self: torch.Tensor, *args, **kwargs):
        """
        See :func:`treetensor.torch.min`
        """
        return self.min(*args, **kwargs)

    @doc_from(torch.Tensor.sum)
    @tireduce(torch.sum)
    @method_treelize(return_type=TreeObject)
    def sum(self: torch.Tensor, *args, **kwargs):
        """
        See :func:`treetensor.torch.sum`
        """
        return self.sum(*args, **kwargs)

    @method_treelize()
    def __eq__(self, other):
        """
        See :func:`treetensor.torch.eq`.
        """
        return self == other

    @method_treelize()
    def __ne__(self, other):
        """
        See :func:`treetensor.torch.ne`.
        """
        return self != other

    @method_treelize()
    def __lt__(self, other):
        """
        See :func:`treetensor.torch.lt`.
        """
        return self < other

    @method_treelize()
    def __gt__(self, other):
        """
        See :func:`treetensor.torch.gt`.
        """
        return self > other

    @method_treelize()
    def __le__(self, other):
        """
        See :func:`treetensor.torch.le`.
        """
        return self <= other

    @method_treelize()
    def __ge__(self, other):
        """
        See :func:`treetensor.torch.ge`.
        """
        return self >= other
