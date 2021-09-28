from treevalue import tree_class

from .tensor import Tensor
from ..base import Torch


@tree_class(return_type=Tensor)
class TensorMethod(Torch):
    pass
