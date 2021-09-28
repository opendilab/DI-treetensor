import torch

from .base import doc_from_base, func_treelize
from ..base import rmreduce, post_reduce, auto_reduce
from ...common import Object

__all__ = [
    'all', 'any',
    'min', 'max', 'sum', 'mean', 'std',
    'masked_select',
]


# noinspection PyShadowingBuiltins,PyUnusedLocal
@post_reduce(torch.all)
@func_treelize(return_type=Object)
def _all_r(input, *args, **kwargs):
    return input


# noinspection PyShadowingBuiltins
@func_treelize()
def _all_nr(input, *args, **kwargs):
    return torch.all(input, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@doc_from_base()
@auto_reduce(_all_r, _all_nr)
def all(input, *args, reduce=None, **kwargs):
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

        >>> ttorch.all(ttorch.tensor({'a': [True, True], 'b': {'x': [True, False]}}), reduce=False)
        <Tensor 0x7fcda55652b0>
        ├── a --> tensor(True)
        └── b --> <Tensor 0x7fcda5565208>
            └── x --> tensor(False)

        >>> ttorch.all(ttorch.tensor({'a': [True, True], 'b': {'x': [True, False]}}), dim=0)
        <Tensor 0x7fcda5565780>
        ├── a --> tensor(True)
        └── b --> <Tensor 0x7fcda55656d8>
            └── x --> tensor(False)

    """
    pass  # pragma: no cover


# noinspection PyShadowingBuiltins,PyUnusedLocal
@post_reduce(torch.any)
@func_treelize(return_type=Object)
def _any_r(input, *args, **kwargs):
    return input


# noinspection PyShadowingBuiltins
@func_treelize()
def _any_nr(input, *args, **kwargs):
    return torch.any(input, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@doc_from_base()
@auto_reduce(_any_r, _any_nr)
def any(input, *args, reduce=None, **kwargs):
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
    pass  # pragma: no cover


# noinspection PyShadowingBuiltins
@doc_from_base()
@rmreduce(torch.min)
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
@rmreduce(torch.max)
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
@rmreduce(torch.sum)
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


# noinspection PyShadowingBuiltins,PyUnusedLocal
@post_reduce(torch.mean)
@func_treelize(return_type=Object)
def _mean_r(input, *args, **kwargs):
    return input


# noinspection PyShadowingBuiltins
@func_treelize()
def _mean_nr(input, *args, **kwargs):
    return torch.mean(input, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@doc_from_base()
@auto_reduce(_mean_r, _mean_nr)
def mean(input, *args, reduce=None, **kwargs):
    """
    Returns the mean value of all elements in the ``input`` tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = torch.randn((2, 3)) * 30
        >>> t
        tensor([[ 26.6598,  27.8008, -59.4753],
                [-79.1833,   3.3349,  20.1665]])
        >>> ttorch.mean(t)
        tensor(-10.1161)

        >>> tt = ttorch.randn({
        ...     'a': (2, 3),
        ...     'b': {'x': (3, 4)},
        ... }) * 30
        >>> tt
        <Tensor 0x7f2f5b9f6cf8>
        ├── a --> tensor([[ 25.2702,  37.4206, -37.1401],
        │                 [ -7.7245, -91.3234, -27.9402]])
        └── b --> <Tensor 0x7f2f5b9f6c18>
            └── x --> tensor([[  3.2028, -14.0720,  18.1739,   8.5944],
                              [ 41.7761,  36.9908, -20.5495,   5.6480],
                              [ -9.3438,  -0.7416,  47.2113,   6.9325]])
        >>> ttorch.mean(tt)
        tensor(1.2436)
        >>> ttorch.mean(tt, reduce=False)
        <Tensor 0x7f1321caf080>
        ├── a --> tensor(-16.9062)
        └── b --> <Tensor 0x7f1321caf048>
            └── x --> tensor(10.3186)
        >>> ttorch.mean(tt, dim=1)
        <Tensor 0x7f63dbbc9828>
        ├── a --> tensor([  8.5169, -42.3294])
        └── b --> <Tensor 0x7f63dbbc9780>
            └── x --> tensor([ 3.9748, 15.9663, 11.0146])

    """
    pass  # pragma: no cover


# noinspection PyShadowingBuiltins,PyUnusedLocal
@post_reduce(torch.std)
@func_treelize(return_type=Object)
def _std_r(input, *args, **kwargs):
    return input


# noinspection PyShadowingBuiltins
@func_treelize()
def _std_nr(input, *args, **kwargs):
    return torch.std(input, *args, **kwargs)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@doc_from_base()
@auto_reduce(_std_r, _std_nr)
def std(input, *args, reduce=None, **kwargs):
    """
    Returns the standard-deviation of all elements in the ``input`` tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = torch.randn((2, 3)) * 30
        >>> t
        tensor([[ 25.5133,  24.2050,   8.1067],
                [ 22.7316, -17.8863, -37.9171]])
        >>> ttorch.std(t)
        tensor(26.3619)

        >>> tt = ttorch.randn({
        ...     'a': (2, 3),
        ...     'b': {'x': (3, 4)},
        ... }) * 30
        >>> tt
        <Tensor 0x7f7c7288ca58>
        ├── a --> tensor([[-48.6580,  30.9506, -16.1800],
        │                 [ 37.6667,  10.3850,  -5.7679]])
        └── b --> <Tensor 0x7f7c7288c978>
            └── x --> tensor([[-17.9371,   8.4873, -49.0445,   4.7368],
                              [ 21.3990, -11.2385, -15.9331, -41.6838],
                              [ -7.1814, -38.1301,  -2.2320,  10.1392]])
        >>> ttorch.std()
        tensor(25.6854)
        >>> ttorch.std(tt, reduce=False)
        <Tensor 0x7f7c7288c470>
        ├── a --> tensor(32.0483)
        └── b --> <Tensor 0x7f7c7288c3c8>
            └── x --> tensor(22.1754)
        >>> ttorch.std(tt, dim=1)
        <Tensor 0x7f1321ca1c50>
        ├── a --> tensor([40.0284, 21.9536])
        └── b --> <Tensor 0x7f1321ca1fd0>
            └── x --> tensor([26.4519, 25.9011, 20.5223])

    """
    pass  # pragma: no cover


# noinspection PyShadowingBuiltins
@doc_from_base()
@rmreduce()
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
