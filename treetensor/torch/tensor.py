import numpy as np
import torch
from treevalue import method_treelize, TreeValue
from treevalue.utils import post_process

from .base import Torch, auto_torch, rmreduce, post_reduce, auto_reduce
from .size import Size
from ..common import Object, ireduce, clsmeta, return_self
from ..numpy import ndarray
from ..utils import current_names, class_autoremove, replaceable_partial
from ..utils import doc_from_base as original_doc_from_base

__all__ = [
    'Tensor'
]

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

    @doc_from_base()
    @return_self
    @method_treelize()
    def requires_grad_(self, requires_grad=True):
        """
        Change if autograd should record operations on this tensor:
        sets this tensor’s ``requires_grad`` attribute in-place. Returns this tensor.
        """
        return self.requires_grad_(requires_grad)

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(torch.all)
    @method_treelize(return_type=Object)
    def __all_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @method_treelize()
    def __all_nr(self, *args, **kwargs):
        return torch.all(self, *args, **kwargs)

    # noinspection PyArgumentList
    @doc_from_base()
    @auto_reduce(__all_r, __all_nr)
    def all(self: torch.Tensor, *args, reduce=None, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.all`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(torch.any)
    @method_treelize(return_type=Object)
    def __any_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @method_treelize()
    def __any_nr(self, *args, **kwargs):
        return torch.any(self, *args, **kwargs)

    # noinspection PyArgumentList
    @doc_from_base()
    @auto_reduce(__any_r, __any_nr)
    def any(self: torch.Tensor, *args, reduce=None, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.any`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(torch.max)
    @method_treelize(return_type=Object)
    def __max_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @post_process(lambda r: replaceable_partial(auto_torch, cls=Tensor)(r))
    @method_treelize(return_type=TreeValue, rise=True)
    def __max_nr(self, *args, **kwargs):
        return torch.max(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__max_r, __max_nr)
    def max(self: torch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.max`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(torch.min)
    @method_treelize(return_type=Object)
    def __min_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @post_process(lambda r: replaceable_partial(auto_torch, cls=Tensor)(r))
    @method_treelize(return_type=TreeValue, rise=True)
    def __min_nr(self, *args, **kwargs):
        return torch.min(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__min_r, __min_nr)
    def min(self: torch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.min`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(torch.sum)
    @method_treelize(return_type=Object)
    def __sum_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @post_process(lambda r: replaceable_partial(auto_torch, cls=Tensor)(r))
    @method_treelize(return_type=TreeValue, rise=True)
    def __sum_nr(self, *args, **kwargs):
        return torch.sum(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__sum_r, __sum_nr)
    def sum(self: torch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.sum`
        """
        pass  # pragma: no cover

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
    def isclose(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.isclose`.
        """
        return self.isclose(other, *args, **kwargs)

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
    @return_self
    @method_treelize()
    def sign_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.sign`.
        """
        return self.sign_(*args, **kwargs)

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

    @doc_from_base()
    @method_treelize()
    def add(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.add`.
        """
        return self.add(other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def add_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.add`.
        """
        return self.add_(other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sub(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.sub`.
        """
        return self.sub(other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sub_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.sub`.
        """
        return self.sub_(other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def mul(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.mul`.
        """
        return self.mul(other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def mul_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.mul`.
        """
        return self.mul_(other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def div(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.div`.
        """
        return self.div(other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def div_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.div`.
        """
        return self.div_(other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def pow(self, exponent, *args, **kwargs):
        """
        See :func:`treetensor.torch.pow`.
        """
        return self.pow(exponent, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def pow_(self, exponent, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.pow`.
        """
        return self.pow_(exponent, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def neg(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.neg`.
        """
        return self.neg(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def neg_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.neg`.
        """
        return self.neg_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def exp(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.exp`.
        """
        return self.exp(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def exp_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.exp`.
        """
        return self.exp_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def exp2(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.exp2`.
        """
        return self.exp2(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def exp2_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.exp2`.
        """
        return self.exp2_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sqrt(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sqrt`.
        """
        return self.sqrt(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sqrt_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.sqrt`.
        """
        return self.sqrt_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def log(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.log`.
        """
        return self.log(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def log_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.log`.
        """
        return self.log_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def log2(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.log2`.
        """
        return self.log2(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def log2_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.log2`.
        """
        return self.log2_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def log10(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.log10`.
        """
        return self.log10(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def log10_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.log10`.
        """
        return self.log10_(*args, **kwargs)

    @doc_from_base()
    @post_process(lambda r: replaceable_partial(auto_torch, cls=Tensor)(r))
    @method_treelize(return_type=TreeValue, rise=dict(template=[None]))
    @post_process(lambda r: list(r))
    def split(self, split_size, *args, **kwargs):
        """
        See :func:`treetensor.torch.split`.
        """
        return self.split(split_size, *args, **kwargs)

    @doc_from_base()
    @post_process(lambda r: replaceable_partial(auto_torch, cls=Tensor)(r))
    @method_treelize(return_type=TreeValue, rise=dict(template=[None]))
    @post_process(lambda r: list(r))
    def chunk(self, chunks, *args, **kwargs):
        """
        See :func:`treetensor.torch.chunk`.
        """
        return self.chunk(chunks, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def reshape(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.reshape`.
        """
        return self.reshape(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def squeeze(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.squeeze`.
        """
        return self.squeeze(*args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def squeeze_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.squeeze'.
        """
        return self.squeeze_(*args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def unsqueeze(self, dim):
        """
        See :func:`treetensor.torch.unsqueeze`.
        """
        return self.unsqueeze(dim)

    @doc_from_base()
    @return_self
    @method_treelize()
    def unsqueeze_(self, dim):
        """
        In-place version of :meth:`Tensor.unsqueeze'.
        """
        return self.unsqueeze_(dim)

    @doc_from_base()
    @method_treelize()
    def where(self, condition, y, *args, **kwargs):
        """
        ``self.where(condition, y)`` is equivalent to
        ``treetensor.torch.where(condition, self, y)``.
        See :func:`treetensor.torch.where`.
        """
        return self.where(condition, y, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def index_select(self, dim, index):
        """
        See :func:`treetensor.torch.index_select`.
        """
        return self.index_select(dim, index)

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @rmreduce()
    @method_treelize(return_type=Object)
    def __masked_select_r(self, mask, *args, **kwargs):
        return torch.masked_select(self, mask, *args, **kwargs)

    # noinspection PyShadowingBuiltins
    @method_treelize()
    def __masked_select_nr(self, mask, *args, **kwargs):
        return torch.masked_select(self, mask, *args, **kwargs)

    # noinspection PyUnusedLocal,PyMethodParameters,PyMethodMayBeStatic
    def __ms_determine(mask, *args, out=None, **kwargs):
        return False if args or kwargs else None

    # noinspection PyUnusedLocal,PyMethodParameters,PyMethodMayBeStatic
    def __ms_condition(mask, *args, out=None, **kwargs):
        return not args and not kwargs

    @doc_from_base()
    @auto_reduce(__masked_select_r, __masked_select_nr,
                 __ms_determine, __ms_condition)
    def masked_select(self, mask):
        """
        See :func:`treetensor.torch.masked_select`.
        """
        pass  # pragma: no cover

    # noinspection PyUnusedLocal
    @post_reduce(torch.std)
    @method_treelize(return_type=Object)
    def __std_r(self, *args, **kwargs):
        return self

    @method_treelize()
    def __std_nr(self, *args, **kwargs):
        return torch.std(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__std_r, __std_nr)
    @method_treelize()
    def std(self, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.std`.
        """
        pass  # pragma: no cover

    # noinspection PyUnusedLocal
    @post_reduce(torch.mean)
    @method_treelize(return_type=Object)
    def __mean_r(self, *args, **kwargs):
        return self

    @method_treelize()
    def __mean_nr(self, *args, **kwargs):
        return torch.mean(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__mean_r, __mean_nr)
    @method_treelize()
    def mean(self, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.mean`.
        """
        pass  # pragma: no cover

    @doc_from_base()
    @method_treelize()
    def dist(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.dist`.
        """
        return self.dist(other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def norm(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.norm`.
        """
        return self.norm(*args, **kwargs)
