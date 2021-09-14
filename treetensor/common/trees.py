from abc import ABCMeta

from treevalue import general_tree_value

__all__ = [
    'BaseTreeStruct', "TreeObject",
]


class BaseTreeStruct(general_tree_value(), metaclass=ABCMeta):
    """
    Overview:
        Base structure of all the trees in ``treetensor``.
    """
    pass


class TreeObject(BaseTreeStruct):
    pass
