from treevalue import general_tree_value


class TreeNumpy(general_tree_value()):
    """
    Overview:
        Real numpy tree.
    """

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
