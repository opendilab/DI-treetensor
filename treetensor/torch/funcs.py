import builtins

import torch
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import BaseTree
from treevalue.utils import post_process

from .tensor import Tensor, tireduce
from ..common import Object, ireduce, return_self
from ..utils import replaceable_partial, doc_from, args_mapping

__all__ = [
    'zeros', 'zeros_like',
    'randn', 'randn_like',
    'randint', 'randint_like',
    'ones', 'ones_like',
    'full', 'full_like',
    'empty', 'empty_like',
    'all', 'any',
    'min', 'max', 'sum',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'equal', 'tensor', 'clone',
    'dot', 'matmul', 'mm',
    'isfinite', 'isinf', 'isnan',
    'abs', 'abs_', 'clamp', 'clamp_', 'sign', 'sigmoid', 'sigmoid_',
    'round', 'round_', 'floor', 'floor_', 'ceil', 'ceil_',
]

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, BaseTree, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize, return_type=Tensor)
)


@doc_from(torch.zeros)
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
    return torch.zeros(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.zeros_like)
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
    return torch.zeros_like(input, *args, **kwargs)


@doc_from(torch.randn)
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
    return torch.randn(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.randn_like)
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
    return torch.randn_like(input, *args, **kwargs)


@doc_from(torch.randint)
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
    return torch.randint(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.randint_like)
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
    return torch.randint_like(input, *args, **kwargs)


@doc_from(torch.ones)
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
    return torch.ones(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.ones_like)
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
    return torch.ones_like(input, *args, **kwargs)


@doc_from(torch.full)
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
    return torch.full(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.full_like)
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
    return torch.full_like(input, *args, **kwargs)


@doc_from(torch.empty)
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
    return torch.empty(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.empty_like)
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
    return torch.empty_like(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.all)
@tireduce(torch.all)
@func_treelize(return_type=Object)
def all(input, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``all`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.all(torch.tensor([True, True]))  # the same as torch.all
        tensor(True)

        >>> ttorch.all(ttorch.tensor({'a': [True, True], 'b': {'x': [True, True]}}))
        tensor(True)

        >>> ttorch.all(ttorch.tensor({'a': [True, True], 'b': {'x': [True, False]}}))
        tensor(False)

    .. note::

        In this ``all`` function, the return value should be a tensor with single boolean value.

        If what you need is a tree of boolean tensors, you should do like this

            >>> ttorch.tensor({
            ...     'a': [True, True],
            ...     'b': {'x': [True, False]},
            ... }).map(lambda x: torch.all(x))
            <Tensor 0x7ff363bbc588>
            ├── a --> tensor(True)
            └── b --> <Tensor 0x7ff363bb6438>
                └── x --> tensor(False)
    """
    return torch.all(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.any)
@tireduce(torch.any)
@func_treelize(return_type=Object)
def any(input, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``any`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.any(torch.tensor([False, False]))  # the same as torch.any
        tensor(False)

        >>> ttorch.any(ttorch.tensor({'a': [True, False], 'b': {'x': [False, False]}}))
        tensor(True)

        >>> ttorch.any(ttorch.tensor({'a': [False, False], 'b': {'x': [False, False]}}))
        tensor(False)

    .. note::

        In this ``any`` function, the return value should be a tensor with single boolean value.

        If what you need is a tree of boolean tensors, you should do like this

            >>> ttorch.tensor({
            >>>     'a': [True, False],
            >>>     'b': {'x': [False, False]},
            >>> }).map(lambda x: torch.any(x))
            <Tensor 0x7ff363bc6898>
            ├── a --> tensor(True)
            └── b --> <Tensor 0x7ff363bc67f0>
                └── x --> tensor(False)
    """
    return torch.any(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.min)
@tireduce(torch.min)
@func_treelize(return_type=Object)
def min(input, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``min`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.min(torch.tensor([1.0, 2.0, 1.5]))  # the same as torch.min
        tensor(1.)

        >>> ttorch.min(ttorch.tensor({
        ...     'a': [1.0, 2.0, 1.5],
        ...     'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        ... }))
        tensor(0.9000)

    .. note::

        In this ``min`` function, the return value should be a tensor with single value.

        If what you need is a tree of tensors, you should do like this

            >>> ttorch.tensor({
            ...     'a': [1.0, 2.0, 1.5],
            ...     'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
            ... }).map(lambda x: torch.min(x))
            <Tensor 0x7ff363bbb2b0>
            ├── a --> tensor(1.)
            └── b --> <Tensor 0x7ff363bbb0b8>
                └── x --> tensor(0.9000)
    """
    return torch.min(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.max)
@tireduce(torch.max)
@func_treelize(return_type=Object)
def max(input, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``max`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.max(torch.tensor([1.0, 2.0, 1.5]))  # the same as torch.max
        tensor(2.)

        >>> ttorch.max(ttorch.tensor({
        ...     'a': [1.0, 2.0, 1.5],
        ...     'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        ... }))
        tensor(2.5000)

    .. note::

        In this ``max`` function, the return value should be a tensor with single value.

        If what you need is a tree of tensors, you should do like this

            >>> ttorch.tensor({
            ...     'a': [1.0, 2.0, 1.5],
            ...     'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
            ... }).map(lambda x: torch.max(x))
            <Tensor 0x7ff363bc6b00>
            ├── a --> tensor(2.)
            └── b --> <Tensor 0x7ff363bc6c18>
                └── x --> tensor(2.5000)
    """
    return torch.max(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.sum)
@tireduce(torch.sum)
@func_treelize(return_type=Object)
def sum(input, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``sum`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.sum(torch.tensor([1.0, 2.0, 1.5]))  # the same as torch.sum
        tensor(4.5000)

        >>> ttorch.sum(ttorch.tensor({
        ...     'a': [1.0, 2.0, 1.5],
        ...     'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        ... }))
        tensor(11.)

    .. note::

        In this ``sum`` function, the return value should be a tensor with single value.

        If what you need is a tree of tensors, you should do like this

            >>> ttorch.tensor({
            ...     'a': [1.0, 2.0, 1.5],
            ...     'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
            ... }).map(lambda x: torch.sum(x))
            <Tensor 0x7ff363bbbda0>
            ├── a --> tensor(4.5000)
            └── b --> <Tensor 0x7ff363bbbcf8>
                └── x --> tensor(6.5000)
    """
    return torch.sum(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.eq)
@func_treelize()
def eq(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get the equality of the two tree tensors with :func:`eq`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.eq(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[1, 1], [4, 4]]),
        ... )
        tensor([[ True, False],
                [False,  True]])

        >>> ttorch.eq(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': [1.0, 1.5, 2.0],
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[1, 1], [4, 4]],
        ...         'b': [1.3, 1.2, 2.0],
        ...     }),
        ... )
        <Tensor 0x7ff363bbce10>
        ├── a --> tensor([[ True, False],
        │                 [False,  True]])
        └── b --> tensor([False, False,  True])
    """
    return torch.eq(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.ne)
@func_treelize()
def ne(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get the non-equality of the two tree tensors with :func:`ne`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.ne(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[1, 1], [4, 4]]),
        ... )
        tensor([[False,  True],
                [ True, False]])

        >>> ttorch.ne(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': [1.0, 1.5, 2.0],
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[1, 1], [4, 4]],
        ...         'b': [1.3, 1.2, 2.0],
        ...     }),
        ... )
        <Tensor 0x7ff363bb6cf8>
        ├── a --> tensor([[False,  True],
        │                 [ True, False]])
        └── b --> tensor([ True,  True, False])
    """
    return torch.ne(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.lt)
@func_treelize()
def lt(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get less-than situation of the two tree tensors with :func:`lt`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.lt(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[1, 1], [4, 4]]),
        ... )
        tensor([[False, False],
                [ True, False]])

        >>> ttorch.lt(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': [1.0, 1.5, 2.0],
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[1, 1], [4, 4]],
        ...         'b': [1.3, 1.2, 2.0],
        ...     }),
        ... )
        <Tensor 0x7ff363bc67f0>
        ├── a --> tensor([[False, False],
        │                 [ True, False]])
        └── b --> tensor([ True, False, False])
    """
    return torch.lt(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.le)
@func_treelize()
def le(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get less-than-or-equal situation of the two tree tensors with :func:`le`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.le(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[1, 1], [4, 4]]),
        ... )
        tensor([[ True, False],
                [ True,  True]])

        >>> ttorch.le(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': [1.0, 1.5, 2.0],
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[1, 1], [4, 4]],
        ...         'b': [1.3, 1.2, 2.0],
        ...     }),
        ... )
        <Tensor 0x7ff363bc6198>
        ├── a --> tensor([[ True, False],
        │                 [ True,  True]])
        └── b --> tensor([ True, False,  True])
    """
    return torch.le(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.gt)
@func_treelize()
def gt(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get greater-than situation of the two tree tensors with :func:`gt`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.gt(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[1, 1], [4, 4]]),
        ... )
        tensor([[False,  True],
                [False, False]])

        >>> ttorch.gt(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': [1.0, 1.5, 2.0],
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[1, 1], [4, 4]],
        ...         'b': [1.3, 1.2, 2.0],
        ...     }),
        ... )
        <Tensor 0x7ff363bc6f28>
        ├── a --> tensor([[False,  True],
        │                 [False, False]])
        └── b --> tensor([False,  True, False])
    """
    return torch.gt(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.ge)
@func_treelize()
def ge(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get greater-than-or-equal situation of the two tree tensors with :func:`ge`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.ge(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[1, 1], [4, 4]]),
        ... )
        tensor([[ True,  True],
                [False,  True]])

        >>> ttorch.ge(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': [1.0, 1.5, 2.0],
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[1, 1], [4, 4]],
        ...         'b': [1.3, 1.2, 2.0],
        ...     }),
        ... )
        <Tensor 0x7ff363bc6f28>
        ├── a --> tensor([[ True,  True],
        │                 [False,  True]])
        └── b --> tensor([False,  True,  True])
    """
    return torch.ge(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyArgumentList
@doc_from(torch.equal)
@ireduce(builtins.all)
@func_treelize()
def equal(input, other):
    """
    In ``treetensor``, you can get the equality of the two tree tensors.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.equal(
        ...     torch.tensor([1, 2, 3]),
        ...     torch.tensor([1, 2, 3]),
        ... )  # the same as torch.equal
        True

        >>> ttorch.equal(
        ...     ttorch.tensor({
        ...         'a': torch.tensor([1, 2, 3]),
        ...         'b': torch.tensor([[4, 5], [6, 7]]),
        ...     }),
        ...     ttorch.tensor({
        ...         'a': torch.tensor([1, 2, 3]),
        ...         'b': torch.tensor([[4, 5], [6, 7]]),
        ...     }),
        ... )
        True
    """
    return torch.equal(input, other)


@doc_from(torch.tensor)
@func_treelize()
def tensor(*args, **kwargs):
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
    return torch.tensor(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.clone)
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
    return torch.clone(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.dot)
@func_treelize()
def dot(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can get the dot product of 2 tree tensors with :func:`treetensor.torch.dot`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.dot(torch.tensor([1, 2]), torch.tensor([2, 3]))
        tensor(8)

        >>> ttorch.dot(
        ...     ttorch.tensor({
        ...         'a': [1, 2, 3],
        ...         'b': {'x': [3, 4]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [5, 6, 7],
        ...         'b': {'x': [1, 2]},
        ...     })
        ... )
        <Tensor 0x7feac55bde50>
        ├── a --> tensor(38)
        └── b --> <Tensor 0x7feac55c9250>
            └── x --> tensor(11)
    """
    return torch.dot(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.matmul)
@func_treelize()
def matmul(input, other, *args, **kwargs):
    """
    In ``treetensor``, you can create a matrix product with :func:`treetensor.torch.matmul`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.matmul(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[5, 6], [7, 2]]),
        ... )
        tensor([[19, 10],
                [43, 26]])

        >>> ttorch.matmul(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': {'x': [3, 4, 5, 6]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[5, 6], [7, 2]],
        ...         'b': {'x': [4, 3, 2, 1]},
        ...     }),
        ... )
        <Tensor 0x7f2e74883f40>
        ├── a --> tensor([[19, 10],
        │                 [43, 26]])
        └── b --> <Tensor 0x7f2e74886430>
            └── x --> tensor(40)
    """
    return torch.matmul(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.mm)
@func_treelize()
def mm(input, mat2, *args, **kwargs):
    """
    In ``treetensor``, you can create a matrix multiplication with :func:`treetensor.torch.mm`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.mm(
        ...     torch.tensor([[1, 2], [3, 4]]),
        ...     torch.tensor([[5, 6], [7, 2]]),
        ... )
        tensor([[19, 10],
                [43, 26]])

        >>> ttorch.mm(
        ...     ttorch.tensor({
        ...         'a': [[1, 2], [3, 4]],
        ...         'b': {'x': [[3, 4, 5], [6, 7, 8]]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [[5, 6], [7, 2]],
        ...         'b': {'x': [[6, 5], [4, 3], [2, 1]]},
        ...     }),
        ... )
        <Tensor 0x7f2e7489f340>
        ├── a --> tensor([[19, 10],
        │                 [43, 26]])
        └── b --> <Tensor 0x7f2e74896e50>
            └── x --> tensor([[44, 32],
                              [80, 59]])
    """
    return torch.mm(input, mat2, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.isfinite)
@func_treelize()
def isfinite(input):
    """
    In ``treetensor``, you can get a tree of new tensors with boolean elements
    representing if each element is `finite` or not.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([ True, False,  True, False, False])

        >>> ttorch.isfinite(ttorch.tensor({
        ...     'a': [1, float('inf'), 2, float('-inf'), float('nan')],
        ...     'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        ... }))
        <Tensor 0x7fb782a15970>
        ├── a --> tensor([ True, False,  True, False, False])
        └── b --> <Tensor 0x7fb782a1e040>
            └── x --> tensor([[ True, False,  True],
                              [False,  True, False]])
    """
    return torch.isfinite(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.isinf)
@func_treelize()
def isinf(input):
    """
    In ``treetensor``, you can test if each element of ``input``
    is infinite (positive or negative infinity) or not.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([False,  True, False,  True, False])

        >>> ttorch.isinf(ttorch.tensor({
        ...     'a': [1, float('inf'), 2, float('-inf'), float('nan')],
        ...     'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        ... }))
        <Tensor 0x7fb782a29b80>
        ├── a --> tensor([False,  True, False,  True, False])
        └── b --> <Tensor 0x7fb782a2d1f0>
            └── x --> tensor([[False,  True, False],
                              [ True, False, False]])
    """
    return torch.isinf(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.isnan)
@func_treelize()
def isnan(input):
    """
    In ``treetensor``, you get a tree of new tensors with boolean elements representing
    if each element of ``input`` is NaN or not

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.isnan(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([False, False, False, False,  True])

        >>> ttorch.isnan(ttorch.tensor({
        ...     'a': [1, float('inf'), 2, float('-inf'), float('nan')],
        ...     'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        ... }))
        <Tensor 0x7fb782a2d0a0>
        ├── a --> tensor([False, False, False, False,  True])
        └── b --> <Tensor 0x7fb782a29d90>
            └── x --> tensor([[False, False, False],
                              [False, False,  True]])
    """
    return torch.isnan(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.abs)
@func_treelize()
def abs(input, *args, **kwargs):
    return torch.abs(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.abs_)
@return_self
@func_treelize()
def abs_(input):
    return torch.abs_(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.clamp)
@func_treelize()
def clamp(input, *args, **kwargs):
    return torch.clamp(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.clamp_)
@return_self
@func_treelize()
def clamp_(input, *args, **kwargs):
    return torch.clamp_(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.sign)
@func_treelize()
def sign(input, *args, **kwargs):
    return torch.sign(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.round)
@func_treelize()
def round(input, *args, **kwargs):
    return torch.round(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.round_)
@return_self
@func_treelize()
def round_(input):
    return torch.round_(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.floor)
@func_treelize()
def floor(input, *args, **kwargs):
    return torch.floor(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.floor_)
@return_self
@func_treelize()
def floor_(input):
    return torch.floor_(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.ceil)
@func_treelize()
def ceil(input, *args, **kwargs):
    return torch.ceil(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.ceil_)
@return_self
@func_treelize()
def ceil_(input):
    return torch.ceil_(input)


# noinspection PyShadowingBuiltins
@doc_from(torch.sigmoid)
@func_treelize()
def sigmoid(input, *args, **kwargs):
    """
    Get a tree of new tensors with the sigmoid of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.tensor([1.0, 2.0, -1.5]).sigmoid()
        tensor([0.7311, 0.8808, 0.1824])

        >>> ttorch.tensor({
        ...     'a': [1.0, 2.0, -1.5],
        ...     'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        ... }).sigmoid()
        <Tensor 0x7f973a312820>
        ├── a --> tensor([0.7311, 0.8808, 0.1824])
        └── b --> <Tensor 0x7f973a3128b0>
            └── x --> tensor([[0.6225, 0.7685],
                              [0.0759, 0.5622]])
    """
    return torch.sigmoid(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.sigmoid_)
@return_self
@func_treelize()
def sigmoid_(input):
    """
    In-place version of :func:`treetensor.torch.sigmoid`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([1.0, 2.0, -1.5])
        >>> ttorch.sigmoid_(t)
        >>> t
        tensor([0.7311, 0.8808, 0.1824])

        >>> t = ttorch.tensor({
        ...     'a': [1.0, 2.0, -1.5],
        ...     'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        ... })
        >>> ttorch.sigmoid_(t)
        >>> t
        <Tensor 0x7f68fea8d040>
        ├── a --> tensor([0.7311, 0.8808, 0.1824])
        └── b --> <Tensor 0x7f68fea8ee50>
            └── x --> tensor([[0.6225, 0.7685],
                              [0.0759, 0.5622]])
    """
    return torch.sigmoid_(input)
