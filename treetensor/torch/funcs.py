"""
Overview:
    Common functions, based on ``torch`` module.
"""

import builtins

import torch
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.utils import post_process

from .tensor import Tensor, tireduce
from ..common import TreeObject, ireduce
from ..utils import replaceable_partial, doc_from, args_mapping

__all__ = [
    'zeros', 'zeros_like',
    'randn', 'randn_like',
    'randint', 'randint_like',
    'ones', 'ones_like',
    'full', 'full_like',
    'empty', 'empty_like',
    'all', 'any',
    'eq', 'equal',
    'tensor',
]

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, TreeValue)) else x)))(
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
        torch.tensor([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])

        >>> ttorch.zeros({
        >>>     'a': (2, 3),
        >>>     'b': (4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]),
            'b': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        })
    """
    return torch.zeros(*args, **kwargs)


@doc_from(torch.zeros_like)
@func_treelize()
def zeros_like(input_, *args, **kwargs):
    """
    In ``treetensor``, you can use ``zeros_like`` to create a tree of tensors with all zeros like another tree.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.zeros_like(torch.randn(2, 3))  # the same as torch.zeros_like(torch.randn(2, 3))
        torch.tensor([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])

        >>> ttorch.zeros_like({
        >>>     'a': torch.randn(2, 3),
        >>>     'b': torch.randn(4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]),
            'b': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        })
    """
    return torch.zeros_like(input_, *args, **kwargs)


@doc_from(torch.randn)
@func_treelize()
def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs)


@doc_from(torch.randn_like)
@func_treelize()
def randn_like(input_, *args, **kwargs):
    return torch.randn_like(input_, *args, **kwargs)


@doc_from(torch.randint)
@func_treelize()
def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


@doc_from(torch.randint_like)
@func_treelize()
def randint_like(input_, *args, **kwargs):
    return torch.randint_like(input_, *args, **kwargs)


@doc_from(torch.ones)
@func_treelize()
def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@doc_from(torch.ones_like)
@func_treelize()
def ones_like(input_, *args, **kwargs):
    return torch.ones_like(input_, *args, **kwargs)


@doc_from(torch.full)
@func_treelize()
def full(*args, **kwargs):
    return torch.full(*args, **kwargs)


@doc_from(torch.full_like)
@func_treelize()
def full_like(input_, *args, **kwargs):
    return torch.full_like(input_, *args, **kwargs)


@doc_from(torch.empty)
@func_treelize()
def empty(*args, **kwargs):
    return torch.empty(*args, **kwargs)


@doc_from(torch.empty_like)
@func_treelize()
def empty_like(input_, *args, **kwargs):
    return torch.empty_like(input_, *args, **kwargs)


@doc_from(torch.all)
@tireduce(torch.all)
@func_treelize(return_type=TreeObject)
def all(input_, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``all`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.all(torch.tensor([True, True]))  # the same as torch.all
        torch.tensor(True)

        >>> ttorch.all(ttorch.tensor({
        >>>     'a': [True, True],
        >>>     'b': [True, True],
        >>> }))
        torch.tensor(True)

        >>> ttorch.all(ttorch.tensor({
        >>>     'a': [True, True],
        >>>     'b': [True, False],
        >>> }))
        torch.tensor(False)

    .. note::

        In this ``all`` function, the return value should be a tensor with single boolean value.

        If what you need is a tree of boolean tensors, you should do like this

            >>> ttorch.tensor({
            >>>     'a': [True, True],
            >>>     'b': [True, False],
            >>> }).map(torch.all)
            ttorch.tensor({
                'a': torch.tensor(True),
                'b': torch.tensor(False),
            })

    """
    return torch.all(input_, *args, **kwargs)


@doc_from(torch.any)
@tireduce(torch.any)
@func_treelize(return_type=TreeObject)
def any(input_, *args, **kwargs):
    """
    In ``treetensor``, you can get the ``any`` result of a whole tree with this function.

    Example::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.any(torch.tensor([False, False]))  # the same as torch.any
        torch.tensor(False)

        >>> ttorch.any(ttorch.tensor({
        >>>     'a': [True, False],
        >>>     'b': [False, False],
        >>> }))
        torch.tensor(True)

        >>> ttorch.any(ttorch.tensor({
        >>>     'a': [False, False],
        >>>     'b': [False, False],
        >>> }))
        torch.tensor(False)

    .. note::

        In this ``any`` function, the return value should be a tensor with single boolean value.

        If what you need is a tree of boolean tensors, you should do like this

            >>> ttorch.tensor({
            >>>     'a': [True, False],
            >>>     'b': [False, False],
            >>> }).map(torch.any)
            ttorch.tensor({
                'a': torch.tensor(True),
                'b': torch.tensor(False),
            })

    """
    return torch.any(input_, *args, **kwargs)


@doc_from(torch.eq)
@func_treelize()
def eq(input_, other, *args, **kwargs):
    return torch.eq(input_, other, *args, **kwargs)


@doc_from(torch.equal)
@ireduce(builtins.all)
@func_treelize()
def equal(input_, other, *args, **kwargs):
    return torch.equal(input_, other, *args, **kwargs)


@doc_from(torch.tensor)
@func_treelize()
def tensor(*args, **kwargs):
    """
    In ``treetensor``, you can create a tree tensor with simple data structure.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> ttorch.tensor(True)  # the same as torch.tensor(True)
        torch.tensor(True)

        >>> ttorch.tensor([1, 2, 3])  # the same as torch.tensor([1, 2, 3])
        torch.tensor([1, 2, 3])

        >>> ttorch.tensor({'a': 1, 'b': [1, 2, 3], 'c': [[True, False], [False, True]]})
        ttorch.Tensor({
            'a': torch.tensor(1),
            'b': torch.tensor([1, 2, 3]),
            'c': torch.tensor([[True, False], [False, True]]),
        })

    """
    return torch.tensor(*args, **kwargs)
