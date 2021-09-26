import torch

from .base import doc_from_base, func_treelize
from ..tensor import tireduce
from ...common import Object, ireduce

__all__ = [
    'all', 'any',
    'min', 'max', 'sum',
    'masked_select',
]


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
@ireduce(torch.cat, piter=tuple)
@func_treelize(return_type=Object)
def masked_select(input, mask, *args, **kwargs):
    """
    Returns a new 1-D tensor which indexes the ``input`` tensor
    according to the boolean mask ``mask`` which is a BoolTensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = torch.randn(3, 4)
        >>> t
        tensor([[ 0.0481,  0.1741,  0.9820, -0.6354],
                [ 0.8108, -0.7126,  0.1329,  1.0868],
                [-1.8267,  1.3676, -1.4490, -2.0224]])
        >>> ttorch.masked_select(t, t > 0.3)
        tensor([0.9820, 0.8108, 1.0868, 1.3676])

        >>> tt = ttorch.randn({
        ...     'a': (2, 3),
        ...     'b': {'x': (3, 4)},
        ... })
        >>> tt
        <Tensor 0x7f0be77bbc88>
        ├── a --> tensor([[ 1.1799,  0.4652, -1.7895],
        │                 [ 0.0423,  1.0866,  1.3533]])
        └── b --> <Tensor 0x7f0be77bbb70>
            └── x --> tensor([[ 0.8139, -0.6732,  0.0065,  0.9073],
                              [ 0.0596, -2.0621, -0.1598, -1.0793],
                              [-0.0496,  2.1392,  0.6403,  0.4041]])
        >>> ttorch.masked_select(tt, tt > 0.3)
        tensor([1.1799, 0.4652, 1.0866, 1.3533, 0.8139, 0.9073, 2.1392, 0.6403, 0.4041])
    """
    return torch.masked_select(input, mask, *args, **kwargs)
