import torch

from .base import doc_from_base, func_treelize
from ..stream import stream_call

__all__ = [
    'dot', 'matmul', 'mm',
]


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
    return stream_call(torch.dot, input, other, *args, **kwargs)


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
    return stream_call(torch.matmul, input, other, *args, **kwargs)


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
    return stream_call(torch.mm, input, mat2, *args, **kwargs)
