from abc import ABCMeta

from treevalue import general_tree_value, method_treelize

__all__ = [
    'BaseTreeStruct', "TreeData", 'TreeObject',
]


class BaseTreeStruct(general_tree_value(), metaclass=ABCMeta):
    """
    Overview:
        Base structure of all the trees in ``treetensor``.
    """
    pass


class TreeData(BaseTreeStruct, metaclass=ABCMeta):
    """
    Overview:
        In ``TreeData`` class, all the comparison operators will be override.
    """

    @method_treelize()
    def __eq__(self, other):
        return self == other

    @method_treelize()
    def __ne__(self, other):
        return self != other

    @method_treelize()
    def __lt__(self, other):
        return self < other

    @method_treelize()
    def __le__(self, other):
        return self <= other

    @method_treelize()
    def __gt__(self, other):
        return self > other

    @method_treelize()
    def __ge__(self, other):
        return self >= other


class TreeObject(BaseTreeStruct):
    pass
