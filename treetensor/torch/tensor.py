import numpy as np
import torch
from treevalue import method_treelize
from treevalue.utils import pre_process

from .base import Torch
from .size import Size
from ..common import Object, ireduce, clsmeta, return_self
from ..numpy import ndarray
from ..utils import current_names, class_autoremove, replaceable_partial
from ..utils import doc_from_base as original_doc_from_base

__all__ = [
    'Tensor'
]

_reduce_tensor_wrap = pre_process(lambda it: ((torch.tensor([*it]),), {}))
tireduce = pre_process(lambda rfunc: ((_reduce_tensor_wrap(rfunc),), {}))(ireduce)
doc_from_base = replaceable_partial(original_doc_from_base, base=torch.Tensor)


def _to_tensor(*args, **kwargs):
    if (len(args) == 1 and not kwargs) or \
            (not args and set(kwargs.keys()) == {'data'}):
        data = args[0] if len(args) == 1 else kwargs['data']
        if isinstance(data, torch.Tensor):
            return data

    return torch.tensor(*args, **kwargs)


# noinspection PyTypeChecker
@current_names()
@class_autoremove
class Tensor(Torch, metaclass=clsmeta(_to_tensor, allow_dict=True)):
    # noinspection PyUnusedLocal
    def __init__(self, data, *args, **kwargs):
        """
        In :class:`treetensor.torch.Tensor`, it's similar but a little bit different with the
        original :class:`torch.Tensor`.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> torch.Tensor([1, 2, 3])  # in torch.Tensor, default type is float32
            tensor([1., 2., 3.])

            >>> ttorch.Tensor([1, 2, 3])  # a native Tensor object, its type is auto detected with torch.tensor
            tensor([1, 2, 3])

            >>> ttorch.Tensor([1, 2, 3], dtype=torch.float32)  # with float32 type
            tensor([1., 2., 3.])

            >>> ttorch.Tensor({
            ...     'a': [1, 2, 3],
            ...     'b': {'x': [4.0, 5, 6]},
            ...     'c': [[True, ], [False, ]],
            ... })  # a tree-based Tensor object
            <Tensor 0x7f537bb9a880>
            ├── a --> tensor([1, 2, 3])
            ├── b --> <Tensor 0x7f537bb9a0d0>
            │   └── x --> tensor([4., 5., 6.])
            └── c --> tensor([[ True],
                              [False]])
        """
        super(Torch, self).__init__(data)

    @doc_from_base()
    @method_treelize(return_type=ndarray)
    def numpy(self: torch.Tensor) -> np.ndarray:
        """
        Returns ``self`` tree tensor as a NumPy ``ndarray``.
        This tensor and the returned :class:`treetensor.numpy.ndarray` share the same underlying storage.
        Changes to self tensor will be reflected in the ``ndarray`` and vice versa.
        """
        return self.numpy()

    @doc_from_base()
    @method_treelize(return_type=Object)
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
            Object({
                'a': [[1, 2], [3, 4]],
                'b': [1, 2, 3],
                'c': True,
            })
        """
        return self.tolist()

    @doc_from_base()
    @method_treelize()
    def cpu(self: torch.Tensor, *args, **kwargs):
        """
        Returns a copy of this tree tensor in CPU memory.

        If this tree tensor is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return self.cpu(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def cuda(self: torch.Tensor, *args, **kwargs):
        """
        Returns a copy of this tree tensor in CUDA memory.

        If this tree tensor is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return self.cuda(*args, **kwargs)

    @doc_from_base()
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

    @doc_from_base()
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def numel(self: torch.Tensor):
        """
        See :func:`treetensor.torch.numel`
        """
        return self.numel()

    @property
    @doc_from_base()
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

    # noinspection PyArgumentList
    @doc_from_base()
    @tireduce(torch.all)
    @method_treelize(return_type=Object)
    def all(self: torch.Tensor, *args, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.all`
        """
        return self.all(*args, **kwargs)

    # noinspection PyArgumentList
    @doc_from_base()
    @tireduce(torch.any)
    @method_treelize(return_type=Object)
    def any(self: torch.Tensor, *args, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.any`
        """
        return self.any(*args, **kwargs)

    @doc_from_base()
    @tireduce(torch.max)
    @method_treelize(return_type=Object)
    def max(self: torch.Tensor, *args, **kwargs):
        """
        See :func:`treetensor.torch.max`
        """
        return self.max(*args, **kwargs)

    @doc_from_base()
    @tireduce(torch.min)
    @method_treelize(return_type=Object)
    def min(self: torch.Tensor, *args, **kwargs):
        """
        See :func:`treetensor.torch.min`
        """
        return self.min(*args, **kwargs)

    @doc_from_base()
    @tireduce(torch.sum)
    @method_treelize(return_type=Object)
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

    @doc_from_base()
    @method_treelize()
    def clone(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.clone`.
        """
        return self.clone(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def dot(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.dot`.
        """
        return self.dot(other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def mm(self, mat2, *args, **kwargs):
        """
        See :func:`treetensor.torch.mm`.
        """
        return self.mm(mat2, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def matmul(self, tensor2, *args, **kwargs):
        """
        See :func:`treetensor.torch.matmul`.
        """
        return self.matmul(tensor2, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def isfinite(self):
        """
        See :func:`treetensor.torch.isfinite`.
        """
        return self.isfinite()

    @doc_from_base()
    @method_treelize()
    def isinf(self):
        """
        See :func:`treetensor.torch.isinf`.
        """
        return self.isinf()

    @doc_from_base()
    @method_treelize()
    def isnan(self):
        """
        See :func:`treetensor.torch.isnan`.
        """
        return self.isnan()

    @doc_from_base()
    @method_treelize()
    def abs(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.abs`.
        """
        return self.abs(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def abs_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.abs_`.
        """
        return self.abs_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def clamp(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.clamp`.
        """
        return self.clamp(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def clamp_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.clamp_`.
        """
        return self.clamp_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sign(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sign`.
        """
        return self.sign(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sigmoid(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sigmoid`.
        """
        return self.sigmoid(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sigmoid_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sigmoid_`.
        """
        return self.sigmoid_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def floor(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.floor`.
        """
        return self.floor(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def floor_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.floor_`.
        """
        return self.floor_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def ceil(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.ceil`.
        """
        return self.ceil(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def ceil_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.ceil_`.
        """
        return self.ceil_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def round(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.round`.
        """
        return self.round(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def round_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.round_`.
        """
        return self.round_(*args, **kwargs)
