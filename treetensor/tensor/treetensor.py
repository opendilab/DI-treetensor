from torch import Tensor
from treevalue import general_tree_value, method_treelize

from ..numpy import TreeNumpy


# noinspection PyTypeChecker,PyShadowingBuiltins
class TreeTensor(general_tree_value()):
    def numel(self) -> int:
        return self \
            .map(lambda t: t.numel()) \
            .reduce(lambda **kws: sum(kws.values()))

    numpy = method_treelize(return_type=TreeNumpy)(Tensor.numpy)
    cpu = method_treelize()(Tensor.cpu)
    cuda = method_treelize()(Tensor.cuda)
    to = method_treelize()(Tensor.to)
