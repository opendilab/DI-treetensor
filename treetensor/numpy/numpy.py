import numpy as np
from treevalue import method_treelize

from ..common import TreeObject, TreeData


class TreeNumpy(TreeData):
    """
    Overview:
        Real numpy tree.
    """

    @method_treelize(return_type=TreeObject)
    def tolist(self: np.ndarray):
        return self.tolist()

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
