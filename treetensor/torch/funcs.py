import builtins
import sys

import torch
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import BaseTree
from treevalue.utils import post_process

from .base import _auto_torch
from .tensor import Tensor, tireduce
from ..common import Object, ireduce, return_self
from ..utils import doc_from_base as original_doc_from_base
from ..utils import replaceable_partial, args_mapping, module_autoremove

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
    'isfinite', 'isinf', 'isnan', 'isclose',
    'abs', 'abs_', 'clamp', 'clamp_', 'sign', 'sigmoid', 'sigmoid_',
    'round', 'round_', 'floor', 'floor_', 'ceil', 'ceil_',
    'add', 'sub', 'mul', 'div', 'pow', 'neg', 'neg_',
    'exp', 'exp_', 'exp2', 'exp2_', 'sqrt', 'sqrt_',
    'log', 'log_', 'log2', 'log2_', 'log10', 'log10_',
    'cat', 'split', 'stack', 'reshape', 'where', 'squeeze', 'unsqueeze',
]

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, BaseTree, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize, return_type=Tensor)
)
doc_from_base = replaceable_partial(original_doc_from_base, base=torch)
auto_tensor = replaceable_partial(_auto_torch, cls=Tensor)


@doc_from_base()
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
@doc_from_base()
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


@doc_from_base()
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
@doc_from_base()
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


@doc_from_base()
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
@doc_from_base()
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


@doc_from_base()
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
@doc_from_base()
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


@doc_from_base()
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
@doc_from_base()
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


@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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


@doc_from_base()
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
    return torch.clone(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
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
@doc_from_base()
@func_treelize()
def isclose(input, other, *args, **kwargs):
    """
    Returns a new tensor with boolean elements representing
    if each element of ``input`` is “close” to the corresponding element of ``other``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> import math
        >>> ttorch.isclose(
        ...     ttorch.tensor((1., 2, 3)),
        ...     ttorch.tensor((1 + 1e-10, 3, 4))
        ... )
        tensor([ True, False, False])

        >>> ttorch.isclose(
        ...     ttorch.tensor({
        ...         'a': [1., 2, 3],
        ...         'b': {'x': [[float('inf'), 4, 1e20],
        ...                     [-math.inf, 2.2943, 9483.32]]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [1 + 1e-10, 3, 4],
        ...         'b': {'x': [[math.inf, 6, 1e20+1],
        ...                     [-float('inf'), 2.294300000001, 9484.32]]},
        ...     }),
        ... )
        <Tensor 0x7f5b3219f370>
        ├── a --> tensor([ True, False, False])
        └── b --> <Tensor 0x7f5b3219f550>
            └── x --> tensor([[ True, False,  True],
                              [ True,  True, False]])
    """
    return torch.isclose(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def abs(input, *args, **kwargs):
    """
    Computes the absolute value of each element in ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.abs(ttorch.tensor([12, 0, -3]))
        tensor([12,  0,  3])

        >>> ttorch.abs(ttorch.tensor({
        ...     'a': [12, 0, -3],
        ...     'b': {'x': [[-3, 1], [0, -2]]},
        ... }))
        <Tensor 0x7f1c81d78ee0>
        ├── a --> tensor([12,  0,  3])
        └── b --> <Tensor 0x7f1c81d78d90>
            └── x --> tensor([[3, 1],
                              [0, 2]])
    """
    return torch.abs(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def abs_(input):
    """
    In-place version of :func:`treetensor.torch.abs`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([12, 0, -3])
        >>> ttorch.abs_(t)
        >>> t
        tensor([12,  0,  3])

        >>> t = ttorch.tensor({
        ...     'a': [12, 0, -3],
        ...     'b': {'x': [[-3, 1], [0, -2]]},
        ... })
        >>> ttorch.abs_(t)
        >>> t
        <Tensor 0x7f1c81d07ca0>
        ├── a --> tensor([12,  0,  3])
        └── b --> <Tensor 0x7f1c81d07d30>
            └── x --> tensor([[3, 1],
                              [0, 2]])
    """
    return torch.abs_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def clamp(input, *args, **kwargs):
    """
    Clamp all elements in ``input`` into the range `[` ``min``, ``max`` `]`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.clamp(ttorch.tensor([-1.7120,  0.1734, -0.0478, 2.0922]), min=-0.5, max=0.5)
        tensor([-0.5000,  0.1734, -0.0478,  0.5000])

        >>> ttorch.clamp(ttorch.tensor({
        ...     'a': [-1.7120,  0.1734, -0.0478, 2.0922],
        ...     'b': {'x': [[-0.9049, 1.7029, -0.3697], [0.0489, -1.3127, -1.0221]]},
        ... }), min=-0.5, max=0.5)
        <Tensor 0x7fbf5332a7c0>
        ├── a --> tensor([-0.5000,  0.1734, -0.0478,  0.5000])
        └── b --> <Tensor 0x7fbf5332a880>
            └── x --> tensor([[-0.5000,  0.5000, -0.3697],
                              [ 0.0489, -0.5000, -0.5000]])
    """
    return torch.clamp(input, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyUnresolvedReferences
@doc_from_base()
@return_self
@func_treelize()
def clamp_(input, *args, **kwargs):
    """
    In-place version of :func:`treetensor.torch.clamp`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-1.7120,  0.1734, -0.0478, 2.0922])
        >>> ttorch.clamp_(t, min=-0.5, max=0.5)
        >>> t
        tensor([-0.5000,  0.1734, -0.0478,  0.5000])

        >>> t = ttorch.tensor({
        ...     'a': [-1.7120,  0.1734, -0.0478, 2.0922],
        ...     'b': {'x': [[-0.9049, 1.7029, -0.3697], [0.0489, -1.3127, -1.0221]]},
        ... })
        >>> ttorch.clamp_(t, min=-0.5, max=0.5)
        >>> t
        <Tensor 0x7fbf53327730>
        ├── a --> tensor([-0.5000,  0.1734, -0.0478,  0.5000])
        └── b --> <Tensor 0x7fbf533277f0>
            └── x --> tensor([[-0.5000,  0.5000, -0.3697],
                              [ 0.0489, -0.5000, -0.5000]])
    """
    return torch.clamp_(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def sign(input, *args, **kwargs):
    """
    Returns a tree of new tensors with the signs of the elements of input.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.sign(ttorch.tensor([12, 0, -3]))
        tensor([ 1,  0, -1])

        >>> ttorch.sign(ttorch.tensor({
        ...     'a': [12, 0, -3],
        ...     'b': {'x': [[-3, 1], [0, -2]]},
        ... }))
        <Tensor 0x7f1c81d02d30>
        ├── a --> tensor([ 1,  0, -1])
        └── b --> <Tensor 0x7f1c81d02a60>
            └── x --> tensor([[-1,  1],
                              [ 0, -1]])
    """
    return torch.sign(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def round(input, *args, **kwargs):
    """
    Returns a tree of new tensors with each of the elements of ``input``
    rounded to the closest integer.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.round(ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]))
        tensor([[ 1., -2.],
                [-2.,  3.]])

        >>> ttorch.round(ttorch.tensor({
        ...     'a': [[1.2, -1.8], [-2.3, 2.8]],
        ...     'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        ... }))
        <Tensor 0x7fbf5333bc10>
        ├── a --> tensor([[ 1., -2.],
        │                 [-2.,  3.]])
        └── b --> <Tensor 0x7fbf5333bcd0>
            └── x --> tensor([[ 1., -4.,  1.],
                              [-5., -2.,  3.]])
    """
    return torch.round(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def round_(input):
    """
    In-place version of :func:`treetensor.torch.round`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        >>> ttorch.round_(t)
        >>> t
        tensor([[ 1., -2.],
                [-2.,  3.]])

        >>> t = ttorch.tensor({
        ...     'a': [[1.2, -1.8], [-2.3, 2.8]],
        ...     'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        ... })
        >>> ttorch.round_(t)
        >>> t
        <Tensor 0x7fbf5332a460>
        ├── a --> tensor([[ 1., -2.],
        │                 [-2.,  3.]])
        └── b --> <Tensor 0x7fbf5332a1f0>
            └── x --> tensor([[ 1., -4.,  1.],
                              [-5., -2.,  3.]])
    """
    return torch.round_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def floor(input, *args, **kwargs):
    """
    Returns a tree of new tensors with the floor of the elements of ``input``,
    the largest integer less than or equal to each element.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.floor(ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]))
        tensor([[ 1., -2.],
                [-3.,  2.]])

        >>> ttorch.floor(ttorch.tensor({
        ...     'a': [[1.2, -1.8], [-2.3, 2.8]],
        ...     'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        ... }))
        <Tensor 0x7fbf53334250>
        ├── a --> tensor([[ 1., -2.],
        │                 [-3.,  2.]])
        └── b --> <Tensor 0x7fbf53334f10>
            └── x --> tensor([[ 1., -4.,  1.],
                              [-5., -2.,  2.]])
    """
    return torch.floor(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def floor_(input):
    """
    In-place version of :func:`treetensor.torch.floor`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        >>> ttorch.floor_(t)
        >>> t
        tensor([[ 1., -2.],
                [-3.,  2.]])

        >>> t = ttorch.tensor({
        ...     'a': [[1.2, -1.8], [-2.3, 2.8]],
        ...     'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        ... })
        >>> ttorch.floor_(t)
        >>> t
        <Tensor 0x7fbf53396d90>
        ├── a --> tensor([[ 1., -2.],
        │                 [-3.,  2.]])
        └── b --> <Tensor 0x7fbf533a0250>
            └── x --> tensor([[ 1., -4.,  1.],
                              [-5., -2.,  2.]])
    """
    return torch.floor_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def ceil(input, *args, **kwargs):
    """
    Returns a tree of new tensors with the ceil of the elements of ``input``,
    the smallest integer greater than or equal to each element.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.ceil(ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]))
        tensor([[ 2., -1.],
                [-2.,  3.]])

        >>> ttorch.ceil(ttorch.tensor({
        ...     'a': [[1.2, -1.8], [-2.3, 2.8]],
        ...     'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        ... }))
        <Tensor 0x7f1c81d021c0>
        ├── a --> tensor([[ 2., -1.],
        │                 [-2.,  3.]])
        └── b --> <Tensor 0x7f1c81d02280>
            └── x --> tensor([[ 1., -3.,  2.],
                              [-4., -2.,  3.]])
    """
    return torch.ceil(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def ceil_(input):
    """
    In-place version of :func:`treetensor.torch.ceil`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        >>> ttorch.ceil_(t)
        >>> t
        tensor([[ 2., -1.],
                [-2.,  3.]])

        >>> t = ttorch.tensor({
        ...     'a': [[1.2, -1.8], [-2.3, 2.8]],
        ...     'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        ... })
        >>> ttorch.ceil_(t)
        >>> t
        <Tensor 0x7f1c81d78040>
        ├── a --> tensor([[ 2., -1.],
        │                 [-2.,  3.]])
        └── b --> <Tensor 0x7f1c81d780d0>
            └── x --> tensor([[ 1., -3.,  2.],
                              [-4., -2.,  3.]])
    """
    return torch.ceil_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def sigmoid(input, *args, **kwargs):
    """
    Returns a tree of new tensors with the sigmoid of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.sigmoid(ttorch.tensor([1.0, 2.0, -1.5]))
        tensor([0.7311, 0.8808, 0.1824])

        >>> ttorch.sigmoid(ttorch.tensor({
        ...     'a': [1.0, 2.0, -1.5],
        ...     'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        ... }))
        <Tensor 0x7f973a312820>
        ├── a --> tensor([0.7311, 0.8808, 0.1824])
        └── b --> <Tensor 0x7f973a3128b0>
            └── x --> tensor([[0.6225, 0.7685],
                              [0.0759, 0.5622]])
    """
    return torch.sigmoid(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
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


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def add(input, other, *args, **kwargs):
    """
    Adds the scalar ``other`` to each element of the ``input`` input and
    returns a new resulting tree tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.add(
        ...     ttorch.tensor([1, 2, 3]),
        ...     ttorch.tensor([3, 5, 11]),
        ... )
        tensor([ 4,  7, 14])

        >>> ttorch.add(
        ...     ttorch.tensor({
        ...         'a': [1, 2, 3],
        ...         'b': {'x': [[3, 5], [9, 12]]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [3, 5, 11],
        ...         'b': {'x': [[31, -15], [13, 23]]},
        ...     })
        ... )
        <Tensor 0x7f11b139c710>
        ├── a --> tensor([ 4,  7, 14])
        └── b --> <Tensor 0x7f11b139c630>
            └── x --> tensor([[ 34, -10],
                              [ 22,  35]])
    """
    return torch.add(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def sub(input, other, *args, **kwargs):
    """
    Subtracts ``other``, scaled by ``alpha``, from ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.sub(
        ...     ttorch.tensor([1, 2, 3]),
        ...     ttorch.tensor([3, 5, 11]),
        ... )
        tensor([-2, -3, -8])

        >>> ttorch.sub(
        ...     ttorch.tensor({
        ...         'a': [1, 2, 3],
        ...         'b': {'x': [[3, 5], [9, 12]]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [3, 5, 11],
        ...         'b': {'x': [[31, -15], [13, 23]]},
        ...     })
        ... )
        <Tensor 0x7f11b139ccc0>
        ├── a --> tensor([-2, -3, -8])
        └── b --> <Tensor 0x7f11b139cc18>
            └── x --> tensor([[-28,  20],
                              [ -4, -11]])
    """
    return torch.sub(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def mul(input, other, *args, **kwargs):
    """
    Multiplies each element of the input ``input`` with the scalar ``other`` and
    returns a new resulting tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.mul(
        ...     ttorch.tensor([1, 2, 3]),
        ...     ttorch.tensor([3, 5, 11]),
        ... )
        tensor([ 3, 10, 33])

        >>> ttorch.mul(
        ...     ttorch.tensor({
        ...         'a': [1, 2, 3],
        ...         'b': {'x': [[3, 5], [9, 12]]},
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [3, 5, 11],
        ...         'b': {'x': [[31, -15], [13, 23]]},
        ...     })
        ... )
        <Tensor 0x7f11b139ca58>
        ├── a --> tensor([ 3, 10, 33])
        └── b --> <Tensor 0x7f11b139cb00>
            └── x --> tensor([[ 93, -75],
                              [117, 276]])
    """
    return torch.mul(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def div(input, other, *args, **kwargs):
    """
    Divides each element of the input ``input`` by the corresponding element of ``other``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.div(ttorch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637]), 0.5)
        tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

        >>> ttorch.div(
        ...     ttorch.tensor([1.3119, 0.0928, 0.4158, 0.7494, 0.3870]),
        ...     ttorch.tensor([-1.7501, -1.4652,  0.1379, -1.1252,  0.0380]),
        ... )
        tensor([-0.7496, -0.0633,  3.0152, -0.6660, 10.1842])

        >>> ttorch.div(
        ...     ttorch.tensor({
        ...         'a': [ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637],
        ...         'b': {
        ...             'x': [1.3119, 0.0928, 0.4158, 0.7494, 0.3870],
        ...             'y': [[[ 1.9579, -0.0335,  0.1178],
        ...                    [ 0.8287,  1.4520, -0.4696]],
        ...                   [[-2.1659, -0.5831,  0.4080],
        ...                    [ 0.1400,  0.8122,  0.5380]]],
        ...         },
        ...     }),
        ...     ttorch.tensor({
        ...         'a': 0.5,
        ...         'b': {
        ...             'x': [-1.7501, -1.4652,  0.1379, -1.1252,  0.0380],
        ...             'y': [[[-1.3136,  0.7785, -0.7290],
        ...                    [ 0.6025,  0.4635, -1.1882]],
        ...                   [[ 0.2756, -0.4483, -0.2005],
        ...                    [ 0.9587,  1.4623, -2.8323]]],
        ...         },
        ...     }),
        ... )
        <Tensor 0x7f11b139c198>
        ├── a --> tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])
        └── b --> <Tensor 0x7f11b139c320>
            ├── x --> tensor([-0.7496, -0.0633,  3.0152, -0.6660, 10.1842])
            └── y --> tensor([[[-1.4905, -0.0430, -0.1616],
                               [ 1.3754,  3.1327,  0.3952]],

                              [[-7.8589,  1.3007, -2.0349],
                               [ 0.1460,  0.5554, -0.1900]]])
    """
    return torch.div(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def pow(input, exponent, *args, **kwargs):
    """
    Takes the power of each element in ``input`` with ``exponent`` and
    returns a tensor with the result.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.pow(
        ...     ttorch.tensor([4, 3, 2, 6, 2]),
        ...     ttorch.tensor([4, 2, 6, 4, 3]),
        ... )
        tensor([ 256,    9,   64, 1296,    8])

        >>> ttorch.pow(
        ...     ttorch.tensor({
        ...         'a': [4, 3, 2, 6, 2],
        ...         'b': {
        ...             'x': [[3, 4, 6],
        ...                   [6, 3, 5]],
        ...             'y': [[[3, 5, 5],
        ...                    [5, 7, 6]],
        ...                   [[4, 6, 5],
        ...                    [7, 2, 7]]],
        ...         },
        ...     }),
        ...     ttorch.tensor({
        ...         'a': [4, 2, 6, 4, 3],
        ...         'b': {
        ...             'x': [[7, 4, 6],
        ...                   [5, 2, 6]],
        ...             'y': [[[7, 2, 2],
        ...                    [2, 3, 2]],
        ...                   [[5, 2, 6],
        ...                    [7, 3, 4]]],
        ...         },
        ...     }),
        ... )
        <Tensor 0x7f11b13b6e48>
        ├── a --> tensor([ 256,    9,   64, 1296,    8])
        └── b --> <Tensor 0x7f11b13b6d68>
            ├── x --> tensor([[ 2187,   256, 46656],
            │                 [ 7776,     9, 15625]])
            └── y --> tensor([[[  2187,     25,     25],
                               [    25,    343,     36]],

                              [[  1024,     36,  15625],
                               [823543,      8,   2401]]])
    """
    return torch.pow(input, exponent, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def neg(input, *args, **kwargs):
    """
    Returns a new tensor with the negative of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.neg(ttorch.tensor([4, 3, 2, 6, 2]))
        tensor([-4, -3, -2, -6, -2])

        >>> ttorch.neg(ttorch.tensor({
        ...     'a': [4, 3, 2, 6, 2],
        ...     'b': {
        ...         'x': [[3, 4, 6],
        ...               [6, 3, 5]],
        ...         'y': [[[3, 5, 5],
        ...                [5, 7, 6]],
        ...               [[4, 6, 5],
        ...                [7, 2, 7]]],
        ...     },
        ... }))
        <Tensor 0x7f11b13b5860>
        ├── a --> tensor([-4, -3, -2, -6, -2])
        └── b --> <Tensor 0x7f11b13b5828>
            ├── x --> tensor([[-3, -4, -6],
            │                 [-6, -3, -5]])
            └── y --> tensor([[[-3, -5, -5],
                               [-5, -7, -6]],

                              [[-4, -6, -5],
                               [-7, -2, -7]]])
    """
    return torch.neg(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def neg_(input):
    """
    In-place version of :func:`treetensor.torch.neg`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([4, 3, 2, 6, 2])
        >>> ttorch.neg_(t)
        >>> t
        tensor([-4, -3, -2, -6, -2])

        >>> t = ttorch.tensor({
        ...     'a': [4, 3, 2, 6, 2],
        ...     'b': {
        ...         'x': [[3, 4, 6],
        ...               [6, 3, 5]],
        ...         'y': [[[3, 5, 5],
        ...                [5, 7, 6]],
        ...               [[4, 6, 5],
        ...                [7, 2, 7]]],
        ...     },
        ... })
        >>> ttorch.neg_(t)
        >>> t
        <Tensor 0x7f11b13b6fd0>
        ├── a --> tensor([-4, -3, -2, -6, -2])
        └── b --> <Tensor 0x7f11b13b60f0>
            ├── x --> tensor([[-3, -4, -6],
            │                 [-6, -3, -5]])
            └── y --> tensor([[[-3, -5, -5],
                               [-5, -7, -6]],

                              [[-4, -6, -5],
                               [-7, -2, -7]]])
    """
    return torch.neg_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def exp(input, *args, **kwargs):
    """
    Returns a new tensor with the exponential of the elements of the input tensor ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.exp(ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]))
        tensor([1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03])

        >>> ttorch.exp(ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... }))
        <Tensor 0x7ff90a4b0a30>
        ├── a --> tensor([1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03])
        └── b --> <Tensor 0x7ff90a4b0af0>
            └── x --> tensor([[1.3534e-01, 3.3201e+00, 1.2840e+00],
                              [8.8861e+06, 4.2521e+01, 9.6328e-02]])
    """
    return torch.exp(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def exp_(input):
    """
    In-place version of :func:`exp`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        >>> ttorch.exp_(t)
        >>> t
        tensor([1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03])

        >>> t = ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... })
        >>> ttorch.exp_(t)
        >>> t
        <Tensor 0x7ff90a4bdb80>
        ├── a --> tensor([1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03])
        └── b --> <Tensor 0x7ff90a4bdc40>
            └── x --> tensor([[1.3534e-01, 3.3201e+00, 1.2840e+00],
                              [8.8861e+06, 4.2521e+01, 9.6328e-02]])
    """
    return torch.exp_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def exp2(input, *args, **kwargs):
    """
    Computes the base two exponential function of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.exp2(ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]))
        tensor([6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02])

        >>> ttorch.exp2(ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... }))
        <Tensor 0x7ff90a4c3af0>
        ├── a --> tensor([6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02])
        └── b --> <Tensor 0x7ff90a4c3be0>
            └── x --> tensor([[2.5000e-01, 2.2974e+00, 1.1892e+00],
                              [6.5536e+04, 1.3454e+01, 1.9751e-01]])
    """
    return torch.exp2(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def exp2_(input):
    """
    In-place version of :func:`exp2`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        >>> ttorch.exp2_(t)
        >>> t
        tensor([6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02])

        >>> t = ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... })
        >>> ttorch.exp2_(t)
        >>> t
        <Tensor 0x7ff90a4bd250>
        ├── a --> tensor([6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02])
        └── b --> <Tensor 0x7ff90a4bd130>
            └── x --> tensor([[2.5000e-01, 2.2974e+00, 1.1892e+00],
                              [6.5536e+04, 1.3454e+01, 1.9751e-01]])
    """
    return torch.exp2_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def sqrt(input, *args, **kwargs):
    """
    Returns a new tensor with the square-root of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.sqrt(ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]))
        tensor([   nan,    nan, 0.0000, 1.4142, 2.1909, 2.8284])

        >>> ttorch.sqrt(ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... }))
        <Tensor 0x7ff90a4cb760>
        ├── a --> tensor([   nan,    nan, 0.0000, 1.4142, 2.1909, 2.8284])
        └── b --> <Tensor 0x7ff90a4cb5b0>
            └── x --> tensor([[   nan, 1.0954, 0.5000],
                              [4.0000, 1.9365,    nan]])
    """
    return torch.sqrt(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def sqrt_(input):
    """
    In-place version of :func:`sqrt`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        >>> ttorch.sqrt_(t)
        >>> t
        tensor([   nan,    nan, 0.0000, 1.4142, 2.1909, 2.8284])

        >>> t = ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... })
        >>> ttorch.sqrt_(t)
        >>> t
        <Tensor 0x7ff90a4b0af0>
        ├── a --> tensor([   nan,    nan, 0.0000, 1.4142, 2.1909, 2.8284])
        └── b --> <Tensor 0x7ff90a4b04f0>
            └── x --> tensor([[   nan, 1.0954, 0.5000],
                              [4.0000, 1.9365,    nan]])
    """
    return torch.sqrt_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def log(input, *args, **kwargs):
    """
    Returns a new tensor with the natural logarithm of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.log(ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]))
        tensor([   nan,    nan,   -inf, 0.6931, 1.5686, 2.0794])

        >>> ttorch.log(ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... }))
        <Tensor 0x7ff90a4c9ca0>
        ├── a --> tensor([   nan,    nan,   -inf, 0.6931, 1.5686, 2.0794])
        └── b --> <Tensor 0x7ff90a4c9e50>
            └── x --> tensor([[    nan,  0.1823, -1.3863],
                              [ 2.7726,  1.3218,     nan]])
    """
    return torch.log(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def log_(input):
    """
    In-place version of :func:`log`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        >>> ttorch.log_(t)
        >>> t
        tensor([   nan,    nan,   -inf, 0.6931, 1.5686, 2.0794])

        >>> t = ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... })
        >>> ttorch.log_(t)
        >>> t
        <Tensor 0x7ff90a4bdf70>
        ├── a --> tensor([   nan,    nan,   -inf, 0.6931, 1.5686, 2.0794])
        └── b --> <Tensor 0x7ff90a4bdcd0>
            └── x --> tensor([[    nan,  0.1823, -1.3863],
                              [ 2.7726,  1.3218,     nan]])
    """
    return torch.log_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def log2(input, *args, **kwargs):
    """
    Returns a new tensor with the logarithm to the base 2 of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.log2(ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]))
        tensor([   nan,    nan,   -inf, 1.0000, 2.2630, 3.0000])

        >>> ttorch.log2(ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... }))
        <Tensor 0x7ff90a4cff70>
        ├── a --> tensor([   nan,    nan,   -inf, 1.0000, 2.2630, 3.0000])
        └── b --> <Tensor 0x7ff90a4bc070>
            └── x --> tensor([[    nan,  0.2630, -2.0000],
                              [ 4.0000,  1.9069,     nan]])
    """
    return torch.log2(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def log2_(input):
    """
    In-place version of :func:`log2`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        >>> ttorch.log2_(t)
        >>> t
        tensor([   nan,    nan,   -inf, 1.0000, 2.2630, 3.0000])

        >>> t = ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... })
        >>> ttorch.log2_(t)
        >>> t
        <Tensor 0x7ff90a4cbbe0>
        ├── a --> tensor([   nan,    nan,   -inf, 1.0000, 2.2630, 3.0000])
        └── b --> <Tensor 0x7ff90a4cb940>
            └── x --> tensor([[    nan,  0.2630, -2.0000],
                              [ 4.0000,  1.9069,     nan]])
    """
    return torch.log2_(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def log10(input, *args, **kwargs):
    """
    Returns a new tensor with the logarithm to the base 10 of the elements of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.log10(ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]))
        tensor([   nan,    nan,   -inf, 0.3010, 0.6812, 0.9031])

        >>> ttorch.log10(ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... }))
        <Tensor 0x7ff90a4bc4f0>
        ├── a --> tensor([   nan,    nan,   -inf, 0.3010, 0.6812, 0.9031])
        └── b --> <Tensor 0x7ff90a4bc5b0>
            └── x --> tensor([[    nan,  0.0792, -0.6021],
                              [ 1.2041,  0.5740,     nan]])
    """
    return torch.log10(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def log10_(input):
    """
    In-place version of :func:`log10`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        >>> ttorch.log10_(t)
        >>> t
        tensor([   nan,    nan,   -inf, 0.3010, 0.6812, 0.9031])

        >>> t = ttorch.tensor({
        ...     'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
        ...     'b': {'x': [[-2.0, 1.2, 0.25],
        ...                 [16.0, 3.75, -2.34]]},
        ... })
        >>> ttorch.log10_(t)
        >>> t
        <Tensor 0x7ff90a4acdc0>
        ├── a --> tensor([   nan,    nan,   -inf, 0.3010, 0.6812, 0.9031])
        └── b --> <Tensor 0x7ff90a4acf40>
            └── x --> tensor([[    nan,  0.0792, -0.6021],
                              [ 1.2041,  0.5740,     nan]])
    """
    return torch.log10_(input)


@doc_from_base()
@func_treelize(subside=dict(return_type=TreeValue))
def cat(tensors, *args, **kwargs):
    """
    Concatenates the given sequence of ``seq`` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    Examples:

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randint(10, 30, (2, 3))
        >>> t1
        tensor([[21, 29, 17],
                [16, 11, 16]])
        >>> t2 = torch.randint(30, 50, (2, 3))
        tensor([[46, 46, 46],
                [30, 47, 36]])
        >>> t2
        >>> t3 = torch.randint(50, 70, (2, 3))
        tensor([[51, 65, 65],
                [54, 67, 57]])
        >>> t3
        >>> ttorch.cat((t1, t2, t3))
        tensor([[21, 29, 17],
                [16, 11, 16],
                [46, 46, 46],
                [30, 47, 36],
                [51, 65, 65],
                [54, 67, 57]])

        >>> tt1 = ttorch.Tensor({
        ...    'a': t1,
        ...    'b': {'x': t2, 'y': t3},
        ... })
        >>> tt1
        <Tensor 0x7fed579acf60>
        ├── a --> tensor([[21, 29, 17],
        │                 [16, 11, 16]])
        └── b --> <Tensor 0x7fed579acf28>
            ├── x --> tensor([[46, 46, 46],
            │                 [30, 47, 36]])
            └── y --> tensor([[51, 65, 65],
                              [54, 67, 57]])
        >>> tt2 = ttorch.Tensor({
        ...    'a': t2,
        ...    'b': {'x': t3, 'y': t1},
        ... })
        >>> tt2
        <Tensor 0x7fed579d62e8>
        ├── a --> tensor([[46, 46, 46],
        │                 [30, 47, 36]])
        └── b --> <Tensor 0x7fed579d62b0>
            ├── x --> tensor([[51, 65, 65],
            │                 [54, 67, 57]])
            └── y --> tensor([[21, 29, 17],
                              [16, 11, 16]])
        >>> tt3 = ttorch.Tensor({
        ...    'a': t3,
        ...    'b': {'x': t1, 'y': t2},
        ... })
        >>> tt3
        <Tensor 0x7fed579d66a0>
        ├── a --> tensor([[51, 65, 65],
        │                 [54, 67, 57]])
        └── b --> <Tensor 0x7fed579d65f8>
            ├── x --> tensor([[21, 29, 17],
            │                 [16, 11, 16]])
            └── y --> tensor([[46, 46, 46],
                              [30, 47, 36]]
        >>> ttorch.cat((tt1, tt2, tt3))
        <Tensor 0x7fed579d6ac8>
        ├── a --> tensor([[21, 29, 17],
        │                 [16, 11, 16],
        │                 [46, 46, 46],
        │                 [30, 47, 36],
        │                 [51, 65, 65],
        │                 [54, 67, 57]])
        └── b --> <Tensor 0x7fed579d6a90>
            ├── x --> tensor([[46, 46, 46],
            │                 [30, 47, 36],
            │                 [51, 65, 65],
            │                 [54, 67, 57],
            │                 [21, 29, 17],
            │                 [16, 11, 16]])
            └── y --> tensor([[51, 65, 65],
                              [54, 67, 57],
                              [21, 29, 17],
                              [16, 11, 16],
                              [46, 46, 46],
                              [30, 47, 36]])
        >>> ttorch.cat((tt1, tt2, tt3), dim=1)
        <Tensor 0x7fed579644a8>
        ├── a --> tensor([[21, 29, 17, 46, 46, 46, 51, 65, 65],
        │                 [16, 11, 16, 30, 47, 36, 54, 67, 57]])
        └── b --> <Tensor 0x7fed57964438>
            ├── x --> tensor([[46, 46, 46, 51, 65, 65, 21, 29, 17],
            │                 [30, 47, 36, 54, 67, 57, 16, 11, 16]])
            └── y --> tensor([[51, 65, 65, 21, 29, 17, 46, 46, 46],
                              [54, 67, 57, 16, 11, 16, 30, 47, 36]])
    """
    return torch.cat(tensors, *args, **kwargs)


# noinspection PyShadowingNames
@doc_from_base()
@post_process(lambda r: tuple(map(auto_tensor, r)))
@func_treelize(return_type=TreeValue, rise=dict(template=[None]))
@post_process(lambda r: list(r))
def split(tensor, split_size_or_sections, *args, **kwargs):
    """
    Splits the tensor into chunks. Each chunk is a view of the original tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randint(100, (6, 2))
        >>> t1
        tensor([[59, 82],
                [86, 42],
                [71, 84],
                [61, 58],
                [82, 37],
                [14, 31]])
        >>> ttorch.split(t1, (1, 2, 3))
        (tensor([[59, 82]]), tensor([[86, 42],
                [71, 84]]), tensor([[61, 58],
                [82, 37],
                [14, 31]]))

        >>> tt1 = ttorch.randint(100, {
        ...     'a': (6, 2),
        ...     'b': {'x': (6, 2, 3)},
        ... })
        >>> tt1
        <Tensor 0x7f4c8d786400>
        ├── a --> tensor([[ 1, 65],
        │                 [68, 31],
        │                 [76, 73],
        │                 [74, 76],
        │                 [90,  0],
        │                 [95, 89]])
        └── b --> <Tensor 0x7f4c8d786320>
            └── x --> tensor([[[11, 20, 74],
                               [17, 85, 44]],

                              [[67, 37, 89],
                               [76, 28,  0]],

                              [[56, 12,  7],
                               [17, 63, 32]],

                              [[81, 75, 19],
                               [89, 21, 55]],

                              [[71, 53,  0],
                               [66, 82, 57]],

                              [[73, 81, 11],
                               [58, 54, 78]]])
        >>> ttorch.split(tt1, (1, 2, 3))
        (<Tensor 0x7f4c8d7861d0>
        ├── a --> tensor([[ 1, 65]])
        └── b --> <Tensor 0x7f4c8d786128>
            └── x --> tensor([[[11, 20, 74],
                               [17, 85, 44]]])
        , <Tensor 0x7f4c8d7860f0>
        ├── a --> tensor([[68, 31],
        │                 [76, 73]])
        └── b --> <Tensor 0x7f4c8d7860b8>
            └── x --> tensor([[[67, 37, 89],
                               [76, 28,  0]],

                              [[56, 12,  7],
                               [17, 63, 32]]])
        , <Tensor 0x7f4c8d7866d8>
        ├── a --> tensor([[74, 76],
        │                 [90,  0],
        │                 [95, 89]])
        └── b --> <Tensor 0x7f4c8d786668>
            └── x --> tensor([[[81, 75, 19],
                               [89, 21, 55]],

                              [[71, 53,  0],
                               [66, 82, 57]],

                              [[73, 81, 11],
                               [58, 54, 78]]])
        )
    """
    return torch.split(tensor, split_size_or_sections, *args, **kwargs)


@doc_from_base()
@func_treelize(subside=dict(return_type=TreeValue))
def stack(tensors, *args, **kwargs):
    """
    Concatenates a sequence of tensors along a new dimension.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randint(10, 30, (2, 3))
        >>> t1
        tensor([[17, 15, 27],
                [12, 17, 29]])
        >>> t2 = torch.randint(30, 50, (2, 3))
        >>> t2
        tensor([[45, 41, 47],
                [37, 37, 36]])
        >>> t3 = torch.randint(50, 70, (2, 3))
        >>> t3
        tensor([[60, 50, 55],
                [69, 54, 58]])
        >>> ttorch.stack((t1, t2, t3))
        tensor([[[17, 15, 27],
                 [12, 17, 29]],

                [[45, 41, 47],
                 [37, 37, 36]],

                [[60, 50, 55],
                 [69, 54, 58]]])

        >>> tt1 = ttorch.randint(10, 30, {
        ...     'a': (2,  3),
        ...     'b': {'x': (3, 4)},
        ... })
        >>> tt1
        <Tensor 0x7f4c8eba9630>
        ├── a --> tensor([[25, 22, 29],
        │                 [19, 21, 27]])
        └── b --> <Tensor 0x7f4c8eba9550>
            └── x --> tensor([[20, 17, 28, 10],
                              [28, 16, 27, 27],
                              [18, 21, 17, 12]])
        >>> tt2 = ttorch.randint(30, 50, {
        ...     'a': (2,  3),
        ...     'b': {'x': (3, 4)},
        ... })
        >>> tt2
        <Tensor 0x7f4c8eba97b8>
        ├── a --> tensor([[40, 44, 41],
        │                 [39, 44, 40]])
        └── b --> <Tensor 0x7f4c8eba9710>
            └── x --> tensor([[44, 42, 38, 44],
                              [30, 44, 42, 31],
                              [36, 30, 33, 31]])
        >>> ttorch.stack((tt1, tt2))
        <Tensor 0x7f4c8eb411d0>
        ├── a --> tensor([[[25, 22, 29],
        │                  [19, 21, 27]],
        │
        │                 [[40, 44, 41],
        │                  [39, 44, 40]]])
        └── b --> <Tensor 0x7f4c8eb410b8>
            └── x --> tensor([[[20, 17, 28, 10],
                               [28, 16, 27, 27],
                               [18, 21, 17, 12]],

                              [[44, 42, 38, 44],
                               [30, 44, 42, 31],
                               [36, 30, 33, 31]]])
        >>> ttorch.stack((tt1, tt2), dim=1)
        <Tensor 0x7f4c8eba9da0>
        ├── a --> tensor([[[25, 22, 29],
        │                  [40, 44, 41]],
        │
        │                 [[19, 21, 27],
        │                  [39, 44, 40]]])
        └── b --> <Tensor 0x7f4d01fb4898>
            └── x --> tensor([[[20, 17, 28, 10],
                               [44, 42, 38, 44]],

                              [[28, 16, 27, 27],
                               [30, 44, 42, 31]],

                              [[18, 21, 17, 12],
                               [36, 30, 33, 31]]])
    """
    return torch.stack(tensors, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def reshape(input, shape):
    """
    Returns a tensor with the same data and number of elements as ``input``,
    but with the specified shape. When possible, the returned tensor will be a view of ``input``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.reshape(torch.tensor([[1, 2], [3, 4]]), (-1, ))
        tensor([1, 2, 3, 4])

        >>> ttorch.reshape(ttorch.tensor({
        ...     'a': [[1, 2], [3, 4]],
        ...     'b': {'x': [[2], [3], [5], [7], [11], [13]]},
        ... }), (-1, ))
        <Tensor 0x7fc9efa3bda0>
        ├── a --> tensor([1, 2, 3, 4])
        └── b --> <Tensor 0x7fc9efa3bcf8>
            └── x --> tensor([ 2,  3,  5,  7, 11, 13])

    .. note::

        If the given ``shape`` is only one tuple, it should make sure that all the tensors
        in this tree can be reshaped to the given ``shape``. Or you can give a tree of tuples
        to reshape the tensors to different shapes.

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.reshape(ttorch.tensor({
            ...     'a': [[1, 2], [3, 4]],
            ...     'b': {'x': [[2], [3], [5], [7], [11], [13]]},
            ... }), {'a': (4, ), 'b': {'x': (3, 2)}})
            <Tensor 0x7fc9efa3bd68>
            ├── a --> tensor([1, 2, 3, 4])
            └── b --> <Tensor 0x7fc9efa3bf28>
                └── x --> tensor([[ 2,  3],
                                  [ 5,  7],
                                  [11, 13]])

    """
    return torch.reshape(input, shape)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def squeeze(input, *args, **kwargs):
    """
    Returns a tensor with all the dimensions of ``input`` of size 1 removed.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randint(100, (2, 1, 2, 1, 2))
        >>> t1.shape
        torch.Size([2, 1, 2, 1, 2])
        >>> ttorch.squeeze(t1).shape
        torch.Size([2, 2, 2])

        >>> tt1 = ttorch.randint(100, {
        ...     'a': (2, 1, 2, 1, 2),
        ...     'b': {'x': (2, 1, 1, 3)},
        ... })
        >>> tt1.shape
        <Size 0x7fa4c1b05410>
        ├── a --> torch.Size([2, 1, 2, 1, 2])
        └── b --> <Size 0x7fa4c1b05510>
            └── x --> torch.Size([2, 1, 1, 3])
        >>> ttorch.squeeze(tt1).shape
        <Size 0x7fa4c1b9f3d0>
        ├── a --> torch.Size([2, 2, 2])
        └── b --> <Size 0x7fa4c1afe710>
            └── x --> torch.Size([2, 3])
    """
    return torch.squeeze(input, *args, *kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def unsqueeze(input, dim):
    """
    Returns a new tensor with a dimension of size one inserted at the specified position.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randint(100, (100, ))
        >>> t1.shape
        torch.Size([100])
        >>> ttorch.unsqueeze(t1, 0).shape
        torch.Size([1, 100])

        >>> tt1 = ttorch.randint(100, {
        ...     'a': (2, 2, 2),
        ...     'b': {'x': (2, 3)},
        ... })
        >>> tt1.shape
        <Size 0x7f5d1a5741d0>
        ├── a --> torch.Size([2, 2, 2])
        └── b --> <Size 0x7f5d1a5740b8>
            └── x --> torch.Size([2, 3])
        >>> ttorch.unsqueeze(tt1, 1).shape
        <Size 0x7f5d1a5c98d0>
        ├── a --> torch.Size([2, 1, 2, 2])
        └── b --> <Size 0x7f5d1a5c99b0>
            └── x --> torch.Size([2, 1, 3])
    """
    return torch.unsqueeze(input, dim)


@doc_from_base()
@func_treelize()
def where(condition, x, y):
    """
    Return a tree of tensors of elements selected from either ``x`` or ``y``, depending on ``condition``.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.where(
        ...     torch.tensor([[True, False], [False, True]]),
        ...     torch.tensor([[2, 8], [16, 4]]),
        ...     torch.tensor([[3, 11], [5, 7]]),
        ... )
        tensor([[ 2, 11],
                [ 5,  4]])

        >>> tt1 = ttorch.randint(1, 99, {'a': (2, 3), 'b': {'x': (3, 2, 4)}})
        >>> tt1
        <Tensor 0x7f6760ad9908>
        ├── a --> tensor([[27, 90, 80],
        │                 [12, 59,  5]])
        └── b --> <Tensor 0x7f6760ad9860>
            └── x --> tensor([[[71, 52, 92, 79],
                               [48,  4, 13, 96]],

                              [[72, 89, 44, 62],
                               [32,  4, 29, 76]],

                              [[ 6,  3, 93, 89],
                               [44, 89, 85, 90]]])
        >>> ttorch.where(tt1 % 2 == 1, tt1, 0)
        <Tensor 0x7f6760ad9d30>
        ├── a --> tensor([[27,  0,  0],
        │                 [ 0, 59,  5]])
        └── b --> <Tensor 0x7f6760ad9f98>
            └── x --> tensor([[[71,  0,  0, 79],
                               [ 0,  0, 13,  0]],

                              [[ 0, 89,  0,  0],
                               [ 0,  0, 29,  0]],

                              [[ 0,  3, 93, 89],
                               [ 0, 89, 85,  0]]])
    """
    return torch.where(condition, x, y)


_current_module = sys.modules[__name__]
_current_module = module_autoremove(_current_module)
sys.modules[__name__] = _current_module
