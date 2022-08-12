import torch
from hbutils.reflection import post_process
from treevalue import TreeValue

from .base import doc_from_base, func_treelize, auto_tensor
from ..stream import stream_call

__all__ = [
    'cat', 'split', 'chunk', 'stack',
    'reshape', 'where', 'squeeze', 'unsqueeze',
    'index_select',
]


@doc_from_base()
@func_treelize(subside=True)
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
    return stream_call(torch.cat, tensors, *args, **kwargs)


# noinspection PyShadowingNames
@doc_from_base()
@post_process(auto_tensor)
@func_treelize(return_type=TreeValue, rise=True)
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
    return stream_call(torch.split, tensor, split_size_or_sections, *args, **kwargs)


# noinspection PyShadowingBuiltins
@doc_from_base()
@post_process(auto_tensor)
@func_treelize(return_type=TreeValue, rise=True)
def chunk(input, chunks, *args, **kwargs):
    """
    Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = torch.randint(100, (4, 5))
        >>> t
        tensor([[54, 97, 12, 48, 62],
                [92, 87, 28, 53, 54],
                [65, 82, 40, 26, 61],
                [75, 43, 86, 99,  7]])
        >>> ttorch.chunk(t, 2)
        (tensor([[54, 97, 12, 48, 62],
                [92, 87, 28, 53, 54]]), tensor([[65, 82, 40, 26, 61],
                [75, 43, 86, 99,  7]]))

        >>> tt = ttorch.randint(100, {
        ...     'a': (4, 5),
        ...     'b': {'x': (2, 3, 4)},
        ... })
        >>> tt
        <Tensor 0x7f667e2fb358>
        ├── a --> tensor([[80,  2, 15, 45, 48],
        │                 [38, 89, 34, 10, 34],
        │                 [18, 99, 33, 38, 20],
        │                 [43, 21, 35, 43, 37]])
        └── b --> <Tensor 0x7f667e2fb278>
            └── x --> tensor([[[19, 17, 39, 68],
                               [41, 69, 33, 89],
                               [31, 88, 39, 14]],

                              [[27, 81, 84, 35],
                               [29, 65, 17, 72],
                               [53, 50, 75,  0]]])
        >>> ttorch.chunk(tt, 2)
        (<Tensor 0x7f667e9b7eb8>
        ├── a --> tensor([[80,  2, 15, 45, 48],
        │                 [38, 89, 34, 10, 34]])
        └── b --> <Tensor 0x7f667e2e7cf8>
            └── x --> tensor([[[19, 17, 39, 68],
                               [41, 69, 33, 89],
                               [31, 88, 39, 14]]])
        , <Tensor 0x7f66f176dac8>
        ├── a --> tensor([[18, 99, 33, 38, 20],
        │                 [43, 21, 35, 43, 37]])
        └── b --> <Tensor 0x7f668030ba58>
            └── x --> tensor([[[27, 81, 84, 35],
                               [29, 65, 17, 72],
                               [53, 50, 75,  0]]])
    """
    return stream_call(torch.chunk, input, chunks, *args, **kwargs)


@doc_from_base()
@func_treelize(subside=True)
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
    return stream_call(torch.stack, tensors, *args, **kwargs)


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
    return stream_call(torch.reshape, input, shape)


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
    return stream_call(torch.squeeze, input, *args, *kwargs)


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
    return stream_call(torch.unsqueeze, input, dim)


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
    return stream_call(torch.where, condition, x, y)


# noinspection PyShadowingBuiltins
@doc_from_base()
@func_treelize()
def index_select(input, dim, index, *args, **kwargs):
    """
    Returns a new tensor which indexes the ``input`` tensor
    along dimension ``dim`` using the entries in ``index`` which is a LongTensor.

    Examples::

        >>> import torch
        >>> import treetensor.torch as ttorch
        >>> t = torch.randn(3, 4)
        >>> t
        tensor([[ 0.2247, -0.1441, -1.2249, -0.2738],
                [-0.1496, -0.4883, -1.2442,  0.6374],
                [ 0.8017,  1.1220, -2.1013, -0.5951]])
        >>> ttorch.index_select(t, 1, torch.tensor([1, 2]))
        tensor([[-0.1441, -1.2249],
                [-0.4883, -1.2442],
                [ 1.1220, -2.1013]])

        >>> tt = ttorch.randn({
        ...     'a': (3, 4),
        ...     'b': {'x': (5, 6)},
        ... })
        >>> tt
        <Tensor 0x7f6b636c1cf8>
        ├── a --> tensor([[ 3.9724e-05, -3.3134e-01, -1.0441e+00,  7.9233e-01],
        │                 [-1.0035e-01,  2.3422e+00,  1.9307e+00, -1.7215e-01],
        │                 [ 1.9069e+00,  1.1852e+00, -1.0672e+00,  1.3463e+00]])
        └── b --> <Tensor 0x7f6b636c1be0>
            └── x --> tensor([[ 0.5200, -0.3595, -1.4235, -0.2655,  0.9504, -1.7564],
                              [-1.6577, -0.5516,  0.1660, -2.3273, -0.9811, -0.4677],
                              [ 0.7047, -1.6920,  0.3139,  0.6220,  0.4758, -1.2637],
                              [-0.3945, -2.1694,  0.8404, -0.4224, -1.4819,  0.3998],
                              [-0.0308,  0.9777, -0.7776, -0.0101, -1.0446, -1.1500]])
        >>> ttorch.index_select(tt, 1, torch.tensor([1, 2]))
        <Tensor 0x7f6b636c1f28>
        ├── a --> tensor([[-0.3313, -1.0441],
        │                 [ 2.3422,  1.9307],
        │                 [ 1.1852, -1.0672]])
        └── b --> <Tensor 0x7f6b636c1e80>
            └── x --> tensor([[-0.3595, -1.4235],
                              [-0.5516,  0.1660],
                              [-1.6920,  0.3139],
                              [-2.1694,  0.8404],
                              [ 0.9777, -0.7776]])

    .. note::

        If you need to select different indices in the tensors, just do like this.

        >>> ttorch.index_select(tt, 1, ttorch.tensor({'a': [1, 2], 'b': {'x': [1, 3, 5]}}))
        <Tensor 0x7f6b636dbf60>
        ├── a --> tensor([[-0.3313, -1.0441],
        │                 [ 2.3422,  1.9307],
        │                 [ 1.1852, -1.0672]])
        └── b --> <Tensor 0x7f6b636dbe80>
            └── x --> tensor([[-0.3595, -0.2655, -1.7564],
                              [-0.5516, -2.3273, -0.4677],
                              [-1.6920,  0.6220, -1.2637],
                              [-2.1694, -0.4224,  0.3998],
                              [ 0.9777, -0.0101, -1.1500]])
    """
    return stream_call(torch.index_select, input, dim, index, *args, **kwargs)
