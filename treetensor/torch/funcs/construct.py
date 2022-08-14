import torch
from treevalue import TreeValue
from treevalue.tree.common import TreeStorage

from .base import doc_from_base, func_treelize
from ..stream import stream_call
from ...utils import args_mapping

__all__ = [
    'tensor', 'as_tensor', 'clone',
    'zeros', 'zeros_like',
    'randn', 'randn_like',
    'randint', 'randint_like',
    'ones', 'ones_like',
    'full', 'full_like',
    'empty', 'empty_like',
]

args_treelize = args_mapping(lambda i, x: TreeValue(x) if isinstance(x, (dict, TreeStorage, TreeValue)) else x)


@doc_from_base()
@args_treelize
@func_treelize()
def tensor(data, *args, **kwargs):
    """
    In ``treetensor``, you can create a tree tensor with simple data structure.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.tensor(True)  # the same as torch.tensor(True)
        tensor(True)

        >>> ttorch.tensor([1, 2, 3])  # the same as torch.tensor([1, 2, 3])
        tensor([1, 2, 3])

        >>> ttorch.tensor({'a': 1, 'b': [1, 2, 3], 'c': [[True, False], [False, True]]})
        <Tensor 0x7ff363bbcc50>
        ├── a --> tensor(1)
        ├── b --> tensor([1, 2, 3])
        └── c --> tensor([[ True, False],
                          [False,  True]])
    """
    return stream_call(torch.tensor, data, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def as_tensor(data, *args, **kwargs):
    """
    Convert the data into a :class:`treetensor.torch.Tensor` or :class:`torch.Tensor`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.as_tensor(True)
        tensor(True)

        >>> ttorch.as_tensor([1, 2, 3], dtype=torch.float32)
        tensor([1., 2., 3.])

        >>> ttorch.as_tensor({
        ...     'a': torch.tensor([1, 2, 3]),
        ...     'b': {'x': [[4, 5], [6, 7]]}
        ... }, dtype=torch.float32)
        <Tensor 0x7fc2b80c25c0>
        ├── a --> tensor([1., 2., 3.])
        └── b --> <Tensor 0x7fc2b80c24e0>
            └── x --> tensor([[4., 5.],
                              [6., 7.]])
    """
    return stream_call(torch.as_tensor, data, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def clone(input, *args, **kwargs):
    """
    In ``treetensor``, you can create a clone of the original tree with :func:`treetensor.torch.clone`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.clone(torch.tensor([[1, 2], [3, 4]]))
        tensor([[1, 2],
                [3, 4]])

        >>> ttorch.clone(ttorch.tensor({
        ...     'a': [[1, 2], [3, 4]],
        ...     'b': {'x': [[5], [6], [7]]},
        ... }))
        <Tensor 0x7f2a820ba5e0>
        ├── a --> tensor([[1, 2],
        │                 [3, 4]])
        └── b --> <Tensor 0x7f2a820aaf70>
            └── x --> tensor([[5],
                              [6],
                              [7]])
    """
    return stream_call(torch.clone, input, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def zeros(*args, **kwargs):
    """
    In ``treetensor``, you can use ``zeros`` to create a tree of tensors with all zeros.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.zeros(2, 3)  # the same as torch.zeros(2, 3)
        tensor([[0., 0., 0.],
                [0., 0., 0.]])

        >>> ttorch.zeros({'a': (2, 3), 'b': {'x': (4, )}})
        <Tensor 0x7f5f6ccf1ef0>
        ├── a --> tensor([[0., 0., 0.],
        │                 [0., 0., 0.]])
        └── b --> <Tensor 0x7f5fe0107208>
            └── x --> tensor([0., 0., 0., 0.])
    """
    return stream_call(torch.zeros, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@args_treelize
@func_treelize()
def zeros_like(input, *args, **kwargs):
    """
    In ``treetensor``, you can use ``zeros_like`` to create a tree of tensors with all zeros like another tree.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.zeros_like(torch.randn(2, 3))  # the same as torch.zeros_like(torch.randn(2, 3))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])

        >>> ttorch.zeros_like({
        ...    'a': torch.randn(2, 3),
        ...    'b': {'x': torch.randn(4, )},
        ... })
        <Tensor 0x7ff363bb6128>
        ├── a --> tensor([[0., 0., 0.],
        │                 [0., 0., 0.]])
        └── b --> <Tensor 0x7ff363bb6080>
            └── x --> tensor([0., 0., 0., 0.])
    """
    return stream_call(torch.zeros_like, input, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def randn(*args, **kwargs):
    """
    In ``treetensor``, you can use ``randn`` to create a tree of tensors with numbers
    obey standard normal distribution.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.randn(2, 3)  # the same as torch.randn(2, 3)
        tensor([[-0.8534, -0.5754, -0.2507],
                [ 0.0826, -1.4110,  0.9748]])

        >>> ttorch.randn({'a': (2, 3), 'b': {'x': (4, )}})
        <Tensor 0x7ff363bb6518>
        ├── a --> tensor([[ 0.5398,  0.7529, -2.0339],
        │                 [-0.5722, -1.1900,  0.7945]])
        └── b --> <Tensor 0x7ff363bb6438>
            └── x --> tensor([-0.7181,  0.1670, -1.3587, -1.5129])
    """
    return stream_call(torch.randn, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@args_treelize
@func_treelize()
def randn_like(input, *args, **kwargs):
    """
    In ``treetensor``, you can use ``randn_like`` to create a tree of tensors with numbers
    obey standard normal distribution like another tree.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.randn_like(torch.ones(2, 3))  # the same as torch.randn_like(torch.ones(2, 3))
        tensor([[ 1.8436,  0.2601,  0.9687],
                [ 1.6430, -0.1765, -1.1732]])

        >>> ttorch.randn_like({
        ...     'a': torch.ones(2, 3),
        ...     'b': {'x': torch.ones(4, )},
        ... })
        <Tensor 0x7ff3d6f3cb38>
        ├── a --> tensor([[-0.1532,  1.3965, -1.2956],
        │                 [-0.0750,  0.6475,  1.1421]])
        └── b --> <Tensor 0x7ff3d6f420b8>
            └── x --> tensor([ 0.1730,  1.6085,  0.6487, -1.1022])
    """
    return stream_call(torch.randn_like, input, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def randint(*args, **kwargs):
    """
    In ``treetensor``, you can use ``randint`` to create a tree of tensors with numbers
    in an integer range.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.randint(10, (2, 3))  # the same as torch.randint(10, (2, 3))
        tensor([[3, 4, 5],
                [4, 5, 5]])

        >>> ttorch.randint(10, {'a': (2, 3), 'b': {'x': (4, )}})
        <Tensor 0x7ff363bb6438>
        ├── a --> tensor([[5, 3, 7],
        │                 [8, 1, 8]])
        └── b --> <Tensor 0x7ff363bb6240>
            └── x --> tensor([8, 8, 2, 4])
    """
    return stream_call(torch.randint, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@args_treelize
@func_treelize()
def randint_like(input, *args, **kwargs):
    """
    In ``treetensor``, you can use ``randint_like`` to create a tree of tensors with numbers
    in an integer range.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.randint_like(torch.ones(2, 3), 10)  # the same as torch.randint_like(torch.ones(2, 3), 10)
        tensor([[0., 5., 0.],
                [2., 0., 9.]])

        >>> ttorch.randint_like({
        ...     'a': torch.ones(2, 3),
        ...     'b': {'x': torch.ones(4, )},
        ... }, 10)
        <Tensor 0x7ff363bb6748>
        ├── a --> tensor([[3., 6., 1.],
        │                 [8., 9., 5.]])
        └── b --> <Tensor 0x7ff363bb6898>
            └── x --> tensor([4., 4., 7., 1.])
    """
    return stream_call(torch.randint_like, input, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def ones(*args, **kwargs):
    """
    In ``treetensor``, you can use ``ones`` to create a tree of tensors with all ones.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.ones(2, 3)  # the same as torch.ones(2, 3)
        tensor([[1., 1., 1.],
                [1., 1., 1.]])

        >>> ttorch.ones({'a': (2, 3), 'b': {'x': (4, )}})
        <Tensor 0x7ff363bb6eb8>
        ├── a --> tensor([[1., 1., 1.],
        │                 [1., 1., 1.]])
        └── b --> <Tensor 0x7ff363bb6dd8>
            └── x --> tensor([1., 1., 1., 1.])
    """
    return stream_call(torch.ones, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@args_treelize
@func_treelize()
def ones_like(input, *args, **kwargs):
    """
    In ``treetensor``, you can use ``ones_like`` to create a tree of tensors with all ones like another tree.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.ones_like(torch.randn(2, 3))  # the same as torch.ones_like(torch.randn(2, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.]])

        >>> ttorch.ones_like({
        ...     'a': torch.randn(2, 3),
        ...     'b': {'x': torch.randn(4, )},
        ... })
        <Tensor 0x7ff363bbc320>
        ├── a --> tensor([[1., 1., 1.],
        │                 [1., 1., 1.]])
        └── b --> <Tensor 0x7ff363bbc240>
            └── x --> tensor([1., 1., 1., 1.])
    """
    return stream_call(torch.ones_like, input, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def full(*args, **kwargs):
    """
    In ``treetensor``, you can use ``ones`` to create a tree of tensors with the same value.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.full((2, 3), 2.3)  # the same as torch.full((2, 3), 2.3)
        tensor([[2.3000, 2.3000, 2.3000],
                [2.3000, 2.3000, 2.3000]])

        >>> ttorch.full({'a': (2, 3), 'b': {'x': (4, )}}, 2.3)
        <Tensor 0x7ff363bbc7f0>
        ├── a --> tensor([[2.3000, 2.3000, 2.3000],
        │                 [2.3000, 2.3000, 2.3000]])
        └── b --> <Tensor 0x7ff363bbc8d0>
            └── x --> tensor([2.3000, 2.3000, 2.3000, 2.3000])
    """
    return stream_call(torch.full, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@args_treelize
@func_treelize()
def full_like(input, *args, **kwargs):
    """
    In ``treetensor``, you can use ``ones_like`` to create a tree of tensors with
    all the same value of like another tree.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.full_like(torch.randn(2, 3), 2.3)  # the same as torch.full_like(torch.randn(2, 3), 2.3)
        tensor([[2.3000, 2.3000, 2.3000],
                [2.3000, 2.3000, 2.3000]])

        >>> ttorch.full_like({
        ...     'a': torch.randn(2, 3),
        ...     'b': {'x': torch.randn(4, )},
        ... }, 2.3)
        <Tensor 0x7ff363bb6cf8>
        ├── a --> tensor([[2.3000, 2.3000, 2.3000],
        │                 [2.3000, 2.3000, 2.3000]])
        └── b --> <Tensor 0x7ff363bb69e8>
            └── x --> tensor([2.3000, 2.3000, 2.3000, 2.3000])
    """
    return stream_call(torch.full_like, input, *args, **kwargs)


@doc_from_base()
@args_treelize
@func_treelize()
def empty(*args, **kwargs):
    """
    In ``treetensor``, you can use ``ones`` to create a tree of tensors with
    the uninitialized values.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.empty(2, 3)  # the same as torch.empty(2, 3)
        tensor([[-1.3267e-36,  3.0802e-41,  2.3000e+00],
                [ 2.3000e+00,  2.3000e+00,  2.3000e+00]])

        >>> ttorch.empty({'a': (2, 3), 'b': {'x': (4, )}})
        <Tensor 0x7ff363bb6080>
        ├── a --> tensor([[-3.6515e+14,  4.5900e-41, -1.3253e-36],
        │                 [ 3.0802e-41,  2.3000e+00,  2.3000e+00]])
        └── b --> <Tensor 0x7ff363bb66d8>
            └── x --> tensor([-3.6515e+14,  4.5900e-41, -3.8091e-38,  3.0802e-41])
    """
    return stream_call(torch.empty, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@args_treelize
@func_treelize()
def empty_like(input, *args, **kwargs):
    """
    In ``treetensor``, you can use ``ones_like`` to create a tree of tensors with
    all the uninitialized values of like another tree.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.empty_like(torch.randn(2, 3))  # the same as torch.empty_like(torch.randn(2, 3), 2.3)
        tensor([[-3.6515e+14,  4.5900e-41, -1.3266e-36],
                [ 3.0802e-41,  4.4842e-44,  0.0000e+00]])

        >>> ttorch.empty_like({
        ...     'a': torch.randn(2, 3),
        ...     'b': {'x': torch.randn(4, )},
        ... })
        <Tensor 0x7ff363bbc780>
        ├── a --> tensor([[-3.6515e+14,  4.5900e-41, -3.6515e+14],
        │                 [ 4.5900e-41,  1.1592e-41,  0.0000e+00]])
        └── b --> <Tensor 0x7ff3d6f3cb38>
            └── x --> tensor([-1.3267e-36,  3.0802e-41, -3.8049e-38,  3.0802e-41])
    """
    return stream_call(torch.empty_like, input, *args, **kwargs)
