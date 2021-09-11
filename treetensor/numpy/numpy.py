import numpy as np
from treevalue import method_treelize

from ..common import TreeObject, TreeData, ireduce
from ..utils import inherit_names, current_names

__all__ = [
    'TreeNumpy'
]


@current_names()
@inherit_names(TreeData)
class TreeNumpy(TreeData):
    """
    Overview:
        Real numpy tree.
    """

    @method_treelize(return_type=TreeObject)
    def tolist(self: np.ndarray):
        return self.tolist()

    @property
    @ireduce(sum)
    @method_treelize(return_type=TreeObject)
    def size(self: np.ndarray) -> int:
        return self.size

    @property
    @ireduce(sum)
    @method_treelize(return_type=TreeObject)
    def nbytes(self: np.ndarray) -> int:
        return self.nbytes

    @ireduce(sum)
    @method_treelize(return_type=TreeObject)
    def sum(self: np.ndarray, *args, **kwargs):
        return self.sum(*args, **kwargs)

    @ireduce(all)
    @method_treelize(return_type=TreeObject)
    def all(self: np.ndarray, *args, **kwargs):
        return self.all(*args, **kwargs)

    @ireduce(any)
    @method_treelize(return_type=TreeObject)
    def any(self: np.ndarray, *args, **kwargs):
        return self.any(*args, **kwargs)
