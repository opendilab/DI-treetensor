from treevalue import general_tree_value, method_treelize

from ..common import TreeList


class TreeNumpy(general_tree_value()):
    """
    Overview:
        Real numpy tree.
    """

    tolist = method_treelize(return_type=TreeList)(lambda d: d.tolist())

    @property
    def size(self) -> int:
        return self \
            .map(lambda d: d.size) \
            .reduce(lambda **kwargs: sum(kwargs.values()))

    @property
    def nbytes(self) -> int:
        return self \
            .map(lambda d: d.nbytes) \
            .reduce(lambda **kwargs: sum(kwargs.values()))

    def sum(self):
        return self \
            .map(lambda d: d.sum()) \
            .reduce(lambda **kwargs: sum(kwargs.values()))
