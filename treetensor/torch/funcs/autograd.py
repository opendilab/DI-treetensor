import torch

from .base import doc_from_base, func_treelize
from ...common import return_self

__all__ = [
    'detach', 'detach_'
]


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def detach(input):
    """
    Detach tensor from calculation graph.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> tt = ttorch.randn({
        ...     'a': (2, 3),
        ...     'b': {'x': (3, 4)},
        ... })
        >>> tt.requires_grad_(True)
        >>> tt
        <Tensor 0x7f5881338eb8>
        ├── a --> tensor([[ 2.5262,  0.7398,  0.7966],
        │                 [ 1.3164,  1.2248, -2.2494]], requires_grad=True)
        └── b --> <Tensor 0x7f5881338e10>
            └── x --> tensor([[ 0.3578,  0.4611, -0.6668,  0.5356],
                              [-1.4392, -1.2899, -0.0394,  0.8457],
                              [ 0.4492, -0.5188, -0.2375, -1.2649]], requires_grad=True)

        >>> ttorch.detach(tt)
        <Tensor 0x7f588133a588>
        ├── a --> tensor([[ 2.5262,  0.7398,  0.7966],
        │                 [ 1.3164,  1.2248, -2.2494]])
        └── b --> <Tensor 0x7f588133a4e0>
            └── x --> tensor([[ 0.3578,  0.4611, -0.6668,  0.5356],
                              [-1.4392, -1.2899, -0.0394,  0.8457],
                              [ 0.4492, -0.5188, -0.2375, -1.2649]])
    """
    return torch.detach(input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@return_self
@func_treelize()
def detach_(input):
    """
    In-place version of :func:`treetensor.torch.detach`.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> tt = ttorch.randn({
        ...     'a': (2, 3),
        ...     'b': {'x': (3, 4)},
        ... })
        >>> tt.requires_grad_(True)
        >>> tt
        <Tensor 0x7f588133aba8>
        ├── a --> tensor([[-0.1631, -1.1573,  1.3109],
        │                 [ 2.7277, -0.0745, -1.2577]], requires_grad=True)
        └── b --> <Tensor 0x7f588133ab00>
            └── x --> tensor([[-0.5876,  0.9836,  1.9584, -0.1513],
                              [ 0.5369, -1.3986,  0.9361,  0.6765],
                              [ 0.6465, -0.2212,  1.5499, -1.2156]], requires_grad=True)

        >>> ttorch.detach_(tt)
        <Tensor 0x7f588133aba8>
        ├── a --> tensor([[-0.1631, -1.1573,  1.3109],
        │                 [ 2.7277, -0.0745, -1.2577]])
        └── b --> <Tensor 0x7f588133ab00>
            └── x --> tensor([[-0.5876,  0.9836,  1.9584, -0.1513],
                              [ 0.5369, -1.3986,  0.9361,  0.6765],
                              [ 0.6465, -0.2212,  1.5499, -1.2156]])
    """
    return torch.detach_(input)
