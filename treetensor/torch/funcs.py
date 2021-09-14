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
    'min', 'max', 'sum',
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
        torch.tensor([[-0.1524, -0.6836,  0.2071],
                      [-1.0407,  0.2497, -0.2317]])

        >>> ttorch.randn({
        >>>     'a': (2, 3),
        >>>     'b': (4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[-0.2399, -1.3437,  0.0656],
                               [-0.3137, -0.3177, -3.0176]])
            'b': torch.tensor([-1.3047,  0.0188, -0.3311,  0.3112]),
        })
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
        torch.tensor([[-1.3912,  2.3161,  1.0146],
                      [-0.3242,  0.5288,  2.4341]])

        >>> ttorch.randn_like({
        >>>     'a': torch.ones(2, 3),
        >>>     'b': torch.ones(4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[ 1.0548, -0.4282,  2.2030],
                               [-0.5305, -0.2601, -1.2560]])
            'b': torch.tensor([ 0.4502,  0.3977, -0.5329,  0.3459]),
        })
    """
    return torch.randn_like(input, *args, **kwargs)


@doc_from(torch.randint)
@func_treelize()
def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.randint_like)
@func_treelize()
def randint_like(input, *args, **kwargs):
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
        torch.tensor([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]])

        >>> ttorch.ones({
        >>>     'a': (2, 3),
        >>>     'b': (4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]]),
            'b': torch.tensor([1.0, 1.0, 1.0, 1.0]),
        })
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
        torch.tensor([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]])

        >>> ttorch.ones_like({
        >>>     'a': torch.randn(2, 3),
        >>>     'b': torch.randn(4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]]),
            'b': torch.tensor([1.0, 1.0, 1.0, 1.0]),
        })
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
        torch.tensor([[2.3, 2.3, 2.3],
                      [2.3, 2.3, 2.3]])

        >>> ttorch.ones({
        >>>     'a': (2, 3),
        >>>     'b': (4, ),
        >>> }, 2.3)
        ttorch.tensor({
            'a': torch.tensor([[2.3, 2.3, 2.3],
                               [2.3, 2.3, 2.3]]),
            'b': torch.tensor([2.3, 2.3, 2.3, 2.3]),
        })
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
        torch.tensor([[2.3, 2.3, 2.3],
                      [2.3, 2.3, 2.3]])

        >>> ttorch.full_like({
        >>>     'a': torch.randn(2, 3),
        >>>     'b': torch.randn(4, ),
        >>> }, 2.3)
        ttorch.tensor({
            'a': torch.tensor([[2.3, 2.3, 2.3],
                               [2.3, 2.3, 2.3]]),
            'b': torch.tensor([2.3, 2.3, 2.3, 2.3]),
        })
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
        torch.tensor([[ 6.6531e-39,  8.2002e-35,  1.3593e-43],
                      [ 0.0000e+00, -4.0271e-20,  4.5887e-41]])

        >>> ttorch.empty({
        >>>     'a': (2, 3),
        >>>     'b': (4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[-1.1736e+27,  3.0918e-41, -1.1758e+27],
                               [ 3.0918e-41,  8.9683e-44,  0.0000e+00]]),
            'b': torch.tensor([-4.0271e-20,  4.5887e-41, -1.1213e+27,  3.0918e-41]),
        })
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
        torch.tensor([[-4.0271e-20,  4.5887e-41, -4.0271e-20],
                      [ 4.5887e-41,  4.4842e-44,  0.0000e+00]])

        >>> ttorch.empty_like({
        >>>     'a': torch.randn(2, 3),
        >>>     'b': torch.randn(4, ),
        >>> })
        ttorch.tensor({
            'a': torch.tensor([[-1.1978e+27,  3.0918e-41, -1.1976e+27],
                               [ 3.0918e-41,  8.9683e-44,  0.0000e+00]]),
            'b': torch.tensor([-4.0271e-20,  4.5887e-41, -4.0271e-20,  4.5887e-41]),
        })
    """
    return torch.empty_like(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.all)
@tireduce(torch.all)
@func_treelize(return_type=TreeObject)
def all(input, *args, **kwargs):
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
    return torch.all(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.any)
@tireduce(torch.any)
@func_treelize(return_type=TreeObject)
def any(input, *args, **kwargs):
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
    return torch.any(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.min)
@tireduce(torch.min)
@func_treelize(return_type=TreeObject)
def min(input, *args, **kwargs):
    return torch.min(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.max)
@tireduce(torch.max)
@func_treelize(return_type=TreeObject)
def max(input, *args, **kwargs):
    return torch.max(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.sum)
@tireduce(torch.sum)
@func_treelize(return_type=TreeObject)
def sum(input, *args, **kwargs):
    return torch.sum(input, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from(torch.eq)
@func_treelize()
def eq(input, other, *args, **kwargs):
    return torch.eq(input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyArgumentList
@doc_from(torch.equal)
@ireduce(builtins.all)
@func_treelize()
def equal(input, other, *args, **kwargs):
    return torch.equal(input, other, *args, **kwargs)


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
