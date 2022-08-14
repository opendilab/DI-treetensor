import torch

from .base import doc_from_base, func_treelize
from ..stream import stream_call
from ...common import return_self

__all__ = [
    'abs', 'abs_', 'clamp', 'clamp_', 'sign', 'sigmoid', 'sigmoid_',
    'round', 'round_', 'floor', 'floor_', 'ceil', 'ceil_',
    'add', 'sub', 'mul', 'div', 'pow', 'neg', 'neg_',
    'exp', 'exp_', 'exp2', 'exp2_', 'sqrt', 'sqrt_',
    'log', 'log_', 'log2', 'log2_', 'log10', 'log10_',
    'dist', 'norm',
]


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
    return stream_call(torch.abs, input, *args, **kwargs)


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
    return stream_call(torch.abs_, input)


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
    return stream_call(torch.clamp, input, *args, **kwargs)


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
    return stream_call(torch.clamp_, input, *args, **kwargs)


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
    return stream_call(torch.sign, input, *args, **kwargs)


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
    return stream_call(torch.round, input, *args, **kwargs)


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
    return stream_call(torch.round_, input)


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
    return stream_call(torch.floor, input, *args, **kwargs)


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
    return stream_call(torch.floor_, input)


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
    return stream_call(torch.ceil, input, *args, **kwargs)


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
    return stream_call(torch.ceil_, input)


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
    return stream_call(torch.sigmoid, input, *args, **kwargs)


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
    return stream_call(torch.sigmoid_, input)


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
    return stream_call(torch.add, input, other, *args, **kwargs)


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
    return stream_call(torch.sub, input, other, *args, **kwargs)


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
    return stream_call(torch.mul, input, other, *args, **kwargs)


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
    return stream_call(torch.div, input, other, *args, **kwargs)


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
    return stream_call(torch.pow, input, exponent, *args, **kwargs)


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
    return stream_call(torch.neg, input, *args, **kwargs)


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
    return stream_call(torch.neg_, input)


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
    return stream_call(torch.exp, input, *args, **kwargs)


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
    return stream_call(torch.exp_, input)


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
    return stream_call(torch.exp2, input, *args, **kwargs)


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
    return stream_call(torch.exp2_, input)


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
    return stream_call(torch.sqrt, input, *args, **kwargs)


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
    return stream_call(torch.sqrt_, input)


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
    return stream_call(torch.log, input, *args, **kwargs)


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
    return stream_call(torch.log_, input)


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
    return stream_call(torch.log2, input, *args, **kwargs)


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
    return stream_call(torch.log2_, input)


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
    return stream_call(torch.log10, input, *args, **kwargs)


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
    return stream_call(torch.log10_, input)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def dist(input, other, *args, **kwargs):
    """
    Returns the p-norm of (``input`` - ``other``)

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randn(5)
        >>> t1
        tensor([-0.6566,  1.2243,  1.5018, -0.1492,  0.8947])
        >>> t2 = torch.randn(5)
        >>> t2
        tensor([0.5898, 0.6839, 0.0388, 0.4649, 0.7964])
        >>> ttorch.dist(t1, t2)
        tensor(2.0911)

        >>> tt1 = ttorch.randn({'a': (5, ), 'b': {'x': (6, )}})
        >>> tt1
        <Tensor 0x7f95f68495f8>
        ├── a --> tensor([-0.5491,  1.5006, -0.0483,  1.2282, -1.4837])
        └── b --> <Tensor 0x7f95f68494e0>
            └── x --> tensor([-1.8414,  1.2913,  0.0943,  0.3473,  1.2717,  0.6013])
        >>> tt2 = ttorch.randn({'a': (5, ), 'b': {'x': (6, )}})
        >>> tt2
        <Tensor 0x7f95f68ef2b0>
        ├── a --> tensor([ 0.1389, -0.7804, -1.3048, -1.1066,  1.3225])
        └── b --> <Tensor 0x7f95f6849dd8>
            └── x --> tensor([ 1.4873,  0.2218, -0.1063, -0.8726, -0.6756,  0.4805])
        >>> ttorch.dist(tt1, tt2)
        <Tensor 0x7f95f6849358>
        ├── a --> tensor(4.5366)
        └── b --> <Tensor 0x7f95f68494a8>
            └── x --> tensor(4.1904)
    """
    return stream_call(torch.dist, input, other, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def norm(input, *args, **kwargs):
    """
    Returns the matrix norm or vector norm of a given tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t1 = torch.randn(3, 4)
        >>> t1
        tensor([[ 0.0363, -1.7385,  1.0669,  2.6967],
                [ 0.0848,  0.2735,  0.3538,  0.2271],
                [-0.1014,  1.1351, -0.5761, -1.2671]])
        >>> ttorch.norm(t1)
        tensor(3.8638)

        >>> tt1 = ttorch.randn({
        ...     'a': (2, 3),
        ...     'b': {'x': (3, 4)},
        ... })
        >>> tt1
        <Tensor 0x7f95f684f4a8>
        ├── a --> tensor([[-0.5012,  2.0900,  0.0151],
        │                 [-0.5035,  0.2144,  0.8370]])
        └── b --> <Tensor 0x7f95f684f400>
            └── x --> tensor([[ 0.3911,  0.3557, -2.2156,  0.3653],
                              [-0.3503,  1.2182, -0.2364, -0.2854],
                              [-1.5770, -0.7349,  0.8391, -0.2845]])
        >>> ttorch.norm(tt1)
        <Tensor 0x7f95f684fa20>
        ├── a --> tensor(2.3706)
        └── b --> <Tensor 0x7f95f684f978>
            └── x --> tensor(3.2982)
    """
    return stream_call(torch.norm, input, *args, **kwargs)
