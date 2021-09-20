from functools import wraps

import torch
from treevalue import TreeValue
from treevalue import func_treelize as original_func_treelize
from treevalue.tree.common import BaseTree
from treevalue.utils import post_process

from .base import Torch
from ..common import Object, clsmeta, ireduce
from ..utils import replaceable_partial, doc_from, current_names, args_mapping

func_treelize = post_process(post_process(args_mapping(
    lambda i, x: TreeValue(x) if isinstance(x, (dict, BaseTree, TreeValue)) else x)))(
    replaceable_partial(original_func_treelize)
)

__all__ = [
    'Size'
]


def _post_index(func):
    def _has_non_none(tree):
        if isinstance(tree, TreeValue):
            for _, value in tree:
                if _has_non_none(value):
                    return True

            return False
        else:
            return tree is not None

    @wraps(func)
    def _new_func(self, value, *args, **kwargs):
        _tree = func(self, value, *args, **kwargs)
        if not _has_non_none(_tree):
            raise ValueError(f'Can not find {repr(value)} in all the sizes.')
        else:
            return _tree

    return _new_func


# noinspection PyTypeChecker
@current_names()
class Size(Torch, metaclass=clsmeta(torch.Size, allow_dict=True)):
    def __init__(self, data):
        """
        In :class:`treetensor.torch.Size`, it's similar with the original :class:`torch.Size`.

        Examples::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.Size([1, 2, 3])
            torch.Size([1, 2, 3])

            >>> ttorch.Size({
            ...     'a': [1, 2, 3],
            ...     'b': {'x': [3, 4, ]},
            ...     'c': [5],
            ... })
            <Size 0x7fe00b115970>
            ├── a --> torch.Size([1, 2, 3])
            ├── b --> <Size 0x7fe00b115250>
            │   └── x --> torch.Size([3, 4])
            └── c --> torch.Size([5])
        """
        super(Torch, self).__init__(data)

    @doc_from(torch.Size.numel)
    @ireduce(sum)
    @func_treelize(return_type=Object)
    def numel(self: torch.Size) -> Object:
        """
        Get the numel sum of the sizes in this tree.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.Size({
            ...     'a': [1, 2],
            ...     'b': {'x': [3, 2, 4]},
            ... }).numel()
            26
        """
        return self.numel()

    @doc_from(torch.Size.index)
    @_post_index
    @func_treelize(return_type=Object)
    def index(self: torch.Size, value, *args, **kwargs) -> Object:
        """

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.Size({
            ...     'a': [1, 2],
            ...     'b': {'x': [3, 2, 4]},
            ...     'c': [3, 5],
            ... }).index(2)
            <Object 0x7fb412780e80>
            ├── a --> 1
            ├── b --> <Object 0x7fb412780eb8>
            │   └── x --> 1
            └── c --> None

        .. note::

            This method's behaviour is different from the :func:`torch.Size.index`.
            No :class:`ValueError` will be raised unless the value can not be found
            in any of the sizes, instead there will be nones returned in the tree.
        """
        try:
            return self.index(value, *args, **kwargs)
        except ValueError:
            return None

    @doc_from(torch.Size.count)
    @ireduce(sum)
    @func_treelize(return_type=Object)
    def count(self: torch.Size, *args, **kwargs) -> Object:
        """
        Get the occurrence count of the sizes in this tree.

        Example::

            >>> import torch
            >>> import treetensor.torch as ttorch
            >>> ttorch.Size({
            ...     'a': [1, 2],
            ...     'b': {'x': [3, 2, 4]},
            ... }).count(2)
            2
        """
        return self.count(*args, **kwargs)
