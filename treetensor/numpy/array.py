import numpy as np
from treevalue import method_treelize

from .base import TreeNumpy
from ..common import Object, ireduce
from ..utils import current_names

__all__ = [
    'ndarray'
]


# noinspection PyPep8Naming
@current_names()
class ndarray(TreeNumpy):
    """
    Overview:
        Real numpy tree.
    """

    @method_treelize(return_type=Object)
    def tolist(self: np.ndarray):
        return self.tolist()

    @property
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def size(self: np.ndarray) -> int:
        return self.size

    @property
    @ireduce(sum)
    @method_treelize(return_type=Object)
    def nbytes(self: np.ndarray) -> int:
        return self.nbytes

    @ireduce(sum)
    @method_treelize(return_type=Object)
    def sum(self: np.ndarray, *args, **kwargs):
        return self.sum(*args, **kwargs)

    @ireduce(all)
    @method_treelize(return_type=Object)
    def all(self: np.ndarray, *args, **kwargs):
        return self.all(*args, **kwargs)

    @ireduce(any)
    @method_treelize(return_type=Object)
    def any(self: np.ndarray, *args, **kwargs):
        return self.any(*args, **kwargs)

    @method_treelize()
    def __eq__(self, other):
        """
        See :func:`treetensor.numpy.eq`.
        """
        return self == other

    @method_treelize()
    def __ne__(self, other):
        """
        See :func:`treetensor.numpy.ne`.
        """
        return self != other

    @method_treelize()
    def __lt__(self, other):
        """
        See :func:`treetensor.numpy.lt`.
        """
        return self < other

    @method_treelize()
    def __gt__(self, other):
        """
        See :func:`treetensor.numpy.gt`.
        """
        return self > other

    @method_treelize()
    def __le__(self, other):
        """
        See :func:`treetensor.numpy.le`.
        """
        return self <= other

    @method_treelize()
    def __ge__(self, other):
        """
        See :func:`treetensor.numpy.ge`.
        """
        return self >= other
