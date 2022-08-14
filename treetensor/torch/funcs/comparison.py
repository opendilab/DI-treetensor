import builtins

import torch

from .base import doc_from_base, func_treelize
from ..stream import stream_call
from ...common import ireduce

__all__ = [
    'equal',
    'isfinite', 'isinf', 'isnan', 'isclose',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
]


# noinspection PyShadowingBuiltins
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
    return stream_call(torch.equal, input, other)


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
    return stream_call(torch.isfinite, input)


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
    return stream_call(torch.isinf, input)


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
    return stream_call(torch.isnan, input)


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
    return stream_call(torch.isclose, input, other, *args, **kwargs)


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
    return stream_call(torch.eq, input, other, *args, **kwargs)


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
    return stream_call(torch.ne, input, other, *args, **kwargs)


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
    return stream_call(torch.lt, input, other, *args, **kwargs)


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
    return stream_call(torch.le, input, other, *args, **kwargs)


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
    return stream_call(torch.gt, input, other, *args, **kwargs)


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
    return stream_call(torch.ge, input, other, *args, **kwargs)
