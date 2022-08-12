import numpy as np
import torch as pytorch
from hbutils.reflection import post_process
from treevalue import method_treelize, TreeValue, typetrans

from .base import Torch, rmreduce, post_reduce, auto_reduce
from .size import Size
from .stream import stream_call
from ..common import Object, ireduce, clsmeta, return_self, auto_tree, get_tree_proxy
from ..numpy import ndarray
from ..utils import current_names, class_autoremove, replaceable_partial
from ..utils import doc_from_base as original_doc_from_base

__all__ = [
    'Tensor'
]

doc_from_base = replaceable_partial(original_doc_from_base, base=pytorch.Tensor)


def _auto_tensor(t):
    if isinstance(t, TreeValue):
        t = typetrans(t, Object)
        if t.map(lambda x, _: pytorch.is_tensor(x)).all():
            return typetrans(t, Tensor)

    return t


_TorchProxy, _InstanceTorchProxy = get_tree_proxy(pytorch.Tensor, _auto_tensor)


def _to_tensor(data, *args, **kwargs):
    if not args and not kwargs and pytorch.is_tensor(data):
        return data
    else:
        return pytorch.as_tensor(data, *args, **kwargs)


class _BaseTensorMeta(clsmeta(_to_tensor, allow_dict=True)):
    pass


# noinspection PyMethodParameters
class _TensorMeta(_BaseTensorMeta):
    def __init__(cls, *args, **kwargs):
        _BaseTensorMeta.__init__(cls, *args, **kwargs)
        cls.__proxy = None

    @property
    def torch(cls):
        if not cls.__proxy:
            cls.__proxy = _TorchProxy(cls)
        return cls.__proxy

    def __getattr__(cls, name):
        try:
            return cls.torch.__getattr__(name)
        except AttributeError:
            raise AttributeError(f"type object {repr(cls.__name__)} has no attribute {repr(name)}")


# noinspection PyTypeChecker
@current_names()
@class_autoremove
class Tensor(Torch, metaclass=_TensorMeta):
    __auto_tensor = lambda x: replaceable_partial(
        auto_tree,
        cls=[(pytorch.is_tensor, Tensor)]
    )(x)

    # noinspection PyUnusedLocal
    def __init__(self, data, *args, **kwargs):
        """
        In :class:`treetensor.torch.Tensor`, it's similar but a little bit different with the
        original :class:`torch.Tensor`.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> pytorch.Tensor([1, 2, 3])  # in torch.Tensor, default type is float32
            tensor([1., 2., 3.])

            >>> ttorch.Tensor([1, 2, 3])  # a native Tensor object, its type is auto detected with torch.tensor
            tensor([1, 2, 3])

            >>> ttorch.Tensor([1, 2, 3], dtype=pytorch.float32)  # with float32 type
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

    @method_treelize(return_type=Object)
    def __get_attr(self, key):
        return getattr(self, key)

    def _attr_extern(self, name):
        try:
            return getattr(self.torch, name)
        except AttributeError:
            tree = self.__get_attr(name)
            if tree.map(lambda x: pytorch.is_tensor(x)).all():
                return tree.type(Tensor)
            else:
                return tree

    @property
    def torch(self):
        """
        Returns a proxy to get the auto generated function from ``torch``.

        .. note::

            This is useful when some of the method of ``TreeValue`` and ``Tensor``
            have the same name, such as ``view``, you can use :meth:`torch.Tensor.view`
            in this way.

                >>> import treetensor.torch as ttorch
                >>> t = ttorch.randn({'a': (2, 3), 'b': {'x': (3, 4)}})
                <Tensor 0x7ff1b4720340>
                ├── a --> tensor([[-0.8067, -0.7860,  2.1065],
                │                 [ 1.8428, -0.0960,  0.9911]])
                └── b --> <Tensor 0x7ff1b4720400>
                    └── x --> tensor([[ 1.5568, -0.8541, -0.1199,  1.1190],
                                      [-0.7324,  2.7439,  1.0143, -0.0680],
                                      [ 0.0344,  1.2085, -1.3644,  2.9778]])
                >>> t.torch.view((-1, ))  # t.view is a method of treevalue
                <Tensor 0x7ff1b4725610>
                ├── a --> tensor([-0.8067, -0.7860,  2.1065,  1.8428, -0.0960,  0.9911])
                └── b --> <Tensor 0x7ff1b4725400>
                    └── x --> tensor([ 1.5568, -0.8541, -0.1199,  1.1190, -0.7324,  2.7439,  1.0143, -0.0680,
                                       0.0344,  1.2085, -1.3644,  2.9778])
        """
        return _InstanceTorchProxy(self.__class__.torch, self)

    @doc_from_base()
    @method_treelize(return_type=ndarray)
    def numpy(self: pytorch.Tensor) -> np.ndarray:
        """
        Returns ``self`` tree tensor as a NumPy ``ndarray``.
        This tensor and the returned :class:`treetensor.numpy.ndarray` share the same underlying storage.
        Changes to self tensor will be reflected in the ``ndarray`` and vice versa.
        """
        return stream_call(self.numpy, )

    @doc_from_base()
    @method_treelize(return_type=Object)
    def tolist(self: pytorch.Tensor):
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
        return stream_call(self.tolist, )

    @doc_from_base()
    @method_treelize()
    def cpu(self: pytorch.Tensor, *args, **kwargs):
        """
        Returns a copy of this tree tensor in CPU memory.

        If this tree tensor is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return stream_call(self.cpu, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def cuda(self: pytorch.Tensor, *args, **kwargs):
        """
        Returns a copy of this tree tensor in CUDA memory.

        If this tree tensor is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return stream_call(self.cuda, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def to(self: pytorch.Tensor, *args, **kwargs):
        """
        Turn the original tree tensor to another format.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.tensor({
            ...     'a': [[1, 11], [2, 22], [3, 33]],
            ...     'b': {'x': [[4, 5], [6, 7]]},
            ... }).to(pytorch.float64)
            <Tensor 0x7ff363bb6518>
            ├── a --> tensor([[ 1., 11.],
            │                 [ 2., 22.],
            │                 [ 3., 33.]], dtype=torch.float64)
            └── b --> <Tensor 0x7ff363bb6ef0>
                └── x --> tensor([[4., 5.],
                                  [6., 7.]], dtype=torch.float64)
        """
        return stream_call(self.to, *args, **kwargs)

    @doc_from_base()
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def numel(self: pytorch.Tensor):
        """
        See :func:`treetensor.torch.numel`
        """
        return stream_call(self.numel, )

    @property
    @doc_from_base()
    @method_treelize(return_type=Size)
    def shape(self: pytorch.Tensor):
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

    @property
    @method_treelize()
    def grad(self):
        """
        Return the grad data of the whole tree.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> tt = ttorch.randn({
            ...     'a': (2, 3),
            ...     'b': {'x': (3, 4)},
            ... })
            >>> tt.requires_grad_(True)
            >>> tt
            <Tensor 0x7feec3bcce80>
            ├── a --> tensor([[-1.4375,  0.0988,  1.2198],
            │                 [-0.7627, -0.8797, -0.9299]], requires_grad=True)
            └── b --> <Tensor 0x7feec3bccdd8>
                └── x --> tensor([[ 0.2149, -0.5839, -0.6049, -0.9151],
                                  [ 1.5381, -1.4386,  0.1831,  0.2018],
                                  [-0.0725, -0.9062, -2.6212,  0.5929]], requires_grad=True)
            >>> mq = tt.mean() ** 2
            >>> mq.backward()
            >>> tt.grad
            <Tensor 0x7feec3c0fa90>
            ├── a --> tensor([[-0.0438, -0.0438, -0.0438],
            │                 [-0.0438, -0.0438, -0.0438]])
            └── b --> <Tensor 0x7feec3c0f9e8>
                └── x --> tensor([[-0.0438, -0.0438, -0.0438, -0.0438],
                                  [-0.0438, -0.0438, -0.0438, -0.0438],
                                  [-0.0438, -0.0438, -0.0438, -0.0438]])
        """
        return self.grad

    @property
    @method_treelize(return_type=Object)
    def requires_grad(self):
        """
        Return the grad situation of current tree.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> tt = ttorch.randn({
            ...     'a': (2, 3),
            ...     'b': {'x': (3, 4)},
            ... })
            >>> tt.requires_grad_(True)
            >>> tt.requires_grad
            <Object 0x7feec3c229e8>
            ├── a --> True
            └── b --> <Object 0x7feec3c22940>
                └── x --> True

            >>> tt.a.requires_grad_(False)
            >>> tt.requires_grad
            <Object 0x7feec3c0fa58>
            ├── a --> False
            └── b --> <Object 0x7feec3c0f5f8>
                └── x --> True
        """
        return self.requires_grad

    @doc_from_base()
    @return_self
    @method_treelize()
    def requires_grad_(self, requires_grad=True):
        """
        Change if autograd should record operations on this tensor:
        sets this tensor’s ``requires_grad`` attribute in-place. Returns this tensor.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> tt = ttorch.randn({
            ...     'a': (2, 3),
            ...     'b': {'x': (3, 4)},
            ... })
            >>> tt.requires_grad_(True)
            >>> tt
            <Tensor 0x7feec3c22240>
            ├── a --> tensor([[ 1.4754,  1.1167,  1.5431],
            │                 [-0.5816,  0.4746,  0.8392]], requires_grad=True)
            └── b --> <Tensor 0x7feec3c22128>
                └── x --> tensor([[ 0.3361,  0.8194,  0.1297, -0.5547],
                                  [ 0.2531, -0.0637,  0.9822,  2.1618],
                                  [ 2.0140, -0.0929,  0.9304,  1.5430]], requires_grad=True)
        """
        return stream_call(self.requires_grad_, requires_grad)

    @doc_from_base()
    @method_treelize()
    def detach(self):
        """
        See :func:`treetensor.torch.detach`.
        """
        return stream_call(self.detach, )

    @doc_from_base()
    @return_self
    @method_treelize()
    def detach_(self):
        """
        In-place version of :meth:`Tensor.detach`.
        """
        return stream_call(self.detach_, )

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(pytorch.all)
    @method_treelize(return_type=Object)
    def __all_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @method_treelize()
    def __all_nr(self, *args, **kwargs):
        return pytorch.all(self, *args, **kwargs)

    # noinspection PyArgumentList
    @doc_from_base()
    @auto_reduce(__all_r, __all_nr)
    def all(self: pytorch.Tensor, *args, reduce=None, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.all`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(pytorch.any)
    @method_treelize(return_type=Object)
    def __any_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @method_treelize()
    def __any_nr(self, *args, **kwargs):
        return pytorch.any(self, *args, **kwargs)

    # noinspection PyArgumentList
    @doc_from_base()
    @auto_reduce(__any_r, __any_nr)
    def any(self: pytorch.Tensor, *args, reduce=None, **kwargs) -> bool:
        """
        See :func:`treetensor.torch.any`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(pytorch.max)
    @method_treelize(return_type=Object)
    def __max_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __max_nr(self, *args, **kwargs):
        return pytorch.max(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__max_r, __max_nr)
    def max(self: pytorch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.max`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(pytorch.min)
    @method_treelize(return_type=Object)
    def __min_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __min_nr(self, *args, **kwargs):
        return pytorch.min(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__min_r, __min_nr)
    def min(self: pytorch.Tensor, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.min`
        """
        pass  # pragma: no cover

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @post_reduce(pytorch.sum)
    @method_treelize(return_type=Object)
    def __sum_r(self, *args, **kwargs):
        return self

    # noinspection PyShadowingBuiltins
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __sum_nr(self, *args, **kwargs):
        return pytorch.sum(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__sum_r, __sum_nr)
    def sum(self: pytorch.Tensor, *args, reduce=None, **kwargs):
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
        return stream_call(self.clone, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def dot(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.dot`.
        """
        return stream_call(self.dot, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def mm(self, mat2, *args, **kwargs):
        """
        See :func:`treetensor.torch.mm`.
        """
        return stream_call(self.mm, mat2, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def matmul(self, tensor2, *args, **kwargs):
        """
        See :func:`treetensor.torch.matmul`.
        """
        return stream_call(self.matmul, tensor2, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def isfinite(self):
        """
        See :func:`treetensor.torch.isfinite`.
        """
        return stream_call(self.isfinite, )

    @doc_from_base()
    @method_treelize()
    def isinf(self):
        """
        See :func:`treetensor.torch.isinf`.
        """
        return stream_call(self.isinf, )

    @doc_from_base()
    @method_treelize()
    def isnan(self):
        """
        See :func:`treetensor.torch.isnan`.
        """
        return stream_call(self.isnan, )

    @doc_from_base()
    @method_treelize()
    def isclose(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.isclose`.
        """
        return stream_call(self.isclose, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def abs(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.abs`.
        """
        return stream_call(self.abs, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def abs_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.abs_`.
        """
        return stream_call(self.abs_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def clamp(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.clamp`.
        """
        return stream_call(self.clamp, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def clamp_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.clamp_`.
        """
        return stream_call(self.clamp_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sign(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sign`.
        """
        return stream_call(self.sign, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sign_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.sign`.
        """
        return stream_call(self.sign_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sigmoid(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sigmoid`.
        """
        return stream_call(self.sigmoid, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sigmoid_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sigmoid_`.
        """
        return stream_call(self.sigmoid_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def floor(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.floor`.
        """
        return stream_call(self.floor, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def floor_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.floor_`.
        """
        return stream_call(self.floor_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def ceil(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.ceil`.
        """
        return stream_call(self.ceil, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def ceil_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.ceil_`.
        """
        return stream_call(self.ceil_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def round(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.round`.
        """
        return stream_call(self.round, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def round_(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.round_`.
        """
        return stream_call(self.round_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def add(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.add`.
        """
        return stream_call(self.add, other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def add_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.add`.
        """
        return stream_call(self.add_, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sub(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.sub`.
        """
        return stream_call(self.sub, other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sub_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.sub`.
        """
        return stream_call(self.sub_, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def mul(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.mul`.
        """
        return stream_call(self.mul, other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def mul_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.mul`.
        """
        return stream_call(self.mul_, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def div(self, other, *args, **kwargs):
        """
        See :func:`treetensor.torch.div`.
        """
        return stream_call(self.div, other, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def div_(self, other, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.div`.
        """
        return stream_call(self.div_, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def pow(self, exponent, *args, **kwargs):
        """
        See :func:`treetensor.torch.pow`.
        """
        return stream_call(self.pow, exponent, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def pow_(self, exponent, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.pow`.
        """
        return stream_call(self.pow_, exponent, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def neg(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.neg`.
        """
        return stream_call(self.neg, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def neg_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.neg`.
        """
        return stream_call(self.neg_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def exp(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.exp`.
        """
        return stream_call(self.exp, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def exp_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.exp`.
        """
        return stream_call(self.exp_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def exp2(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.exp2`.
        """
        return stream_call(self.exp2, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def exp2_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.exp2`.
        """
        return stream_call(self.exp2_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def sqrt(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.sqrt`.
        """
        return stream_call(self.sqrt, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def sqrt_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.sqrt`.
        """
        return stream_call(self.sqrt_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def log(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.log`.
        """
        return stream_call(self.log, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def log_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.log`.
        """
        return stream_call(self.log_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def log2(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.log2`.
        """
        return stream_call(self.log2, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def log2_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.log2`.
        """
        return stream_call(self.log2_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def log10(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.log10`.
        """
        return stream_call(self.log10, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def log10_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.log10`.
        """
        return stream_call(self.log10_, *args, **kwargs)

    @doc_from_base()
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def split(self, split_size, *args, **kwargs):
        """
        See :func:`treetensor.torch.split`.
        """
        return stream_call(self.split, split_size, *args, **kwargs)

    @doc_from_base()
    @post_process(__auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def chunk(self, chunks, *args, **kwargs):
        """
        See :func:`treetensor.torch.chunk`.
        """
        return stream_call(self.chunk, chunks, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def reshape(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.reshape`.
        """
        return stream_call(self.reshape, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def squeeze(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.squeeze`.
        """
        return stream_call(self.squeeze, *args, **kwargs)

    @doc_from_base()
    @return_self
    @method_treelize()
    def squeeze_(self, *args, **kwargs):
        """
        In-place version of :meth:`Tensor.squeeze'.
        """
        return stream_call(self.squeeze_, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def unsqueeze(self, dim):
        """
        See :func:`treetensor.torch.unsqueeze`.
        """
        return stream_call(self.unsqueeze, dim)

    @doc_from_base()
    @return_self
    @method_treelize()
    def unsqueeze_(self, dim):
        """
        In-place version of :meth:`Tensor.unsqueeze'.
        """
        return stream_call(self.unsqueeze_, dim)

    @doc_from_base()
    @method_treelize()
    def where(self, condition, y, *args, **kwargs):
        """
        ``self.where(condition, y)`` is equivalent to
        ``treetensor.torch.where(condition, self, y)``.
        See :func:`treetensor.torch.where`.
        """
        return stream_call(self.where, condition, y, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def index_select(self, dim, index):
        """
        See :func:`treetensor.torch.index_select`.
        """
        return stream_call(self.index_select, dim, index)

    # noinspection PyShadowingBuiltins,PyUnusedLocal
    @rmreduce()
    @method_treelize(return_type=Object)
    def __masked_select_r(self, mask, *args, **kwargs):
        return pytorch.masked_select(self, mask, *args, **kwargs)

    # noinspection PyShadowingBuiltins
    @method_treelize()
    def __masked_select_nr(self, mask, *args, **kwargs):
        return pytorch.masked_select(self, mask, *args, **kwargs)

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
    @post_reduce(pytorch.std)
    @method_treelize(return_type=Object)
    def __std_r(self, *args, **kwargs):
        return self

    @method_treelize()
    def __std_nr(self, *args, **kwargs):
        return pytorch.std(self, *args, **kwargs)

    @doc_from_base()
    @auto_reduce(__std_r, __std_nr)
    @method_treelize()
    def std(self, *args, reduce=None, **kwargs):
        """
        See :func:`treetensor.torch.std`.
        """
        pass  # pragma: no cover

    # noinspection PyUnusedLocal
    @post_reduce(pytorch.mean)
    @method_treelize(return_type=Object)
    def __mean_r(self, *args, **kwargs):
        return self

    @method_treelize()
    def __mean_nr(self, *args, **kwargs):
        return pytorch.mean(self, *args, **kwargs)

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
        return stream_call(self.dist, other, *args, **kwargs)

    @doc_from_base()
    @method_treelize()
    def norm(self, *args, **kwargs):
        """
        See :func:`treetensor.torch.norm`.
        """
        return stream_call(self.norm, *args, **kwargs)
