from functools import partial
from operator import __eq__

from torch import Tensor
from treevalue import general_tree_value, method_treelize, TreeValue

from ..common import TreeList
from ..numpy import TreeNumpy


def _same_merge(eq, hash_, **kwargs):
    kws = {
        key: value for key, value in kwargs.items()
        if not (isinstance(value, TreeValue) and not value)
    }

    class _Wrapper:
        def __init__(self, v):
            self.v = v

        def __hash__(self):
            return hash_(self.v)

        def __eq__(self, other):
            return eq(self.v, other.v)

    if len(set(_Wrapper(v) for v in kws.values())) == 1:
        return list(kws.values())[0]
    else:
        return TreeTensor(kws)


# noinspection PyTypeChecker,PyShadowingBuiltins
class TreeTensor(general_tree_value()):
    def numel(self) -> int:
        return self \
            .map(lambda t: t.numel()) \
            .reduce(lambda **kws: sum(kws.values()))

    @property
    def raw_shape(self):
        return self.map(lambda t: t.shape)

    @property
    def shape(self):
        return self.raw_shape.reduce(partial(_same_merge, __eq__, hash))

    numpy = method_treelize(return_type=TreeNumpy)(Tensor.numpy)
    tolist = method_treelize(return_type=TreeList)(Tensor.tolist)
    cpu = method_treelize()(Tensor.cpu)
    cuda = method_treelize()(Tensor.cuda)
    to = method_treelize()(Tensor.to)

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

    @method_treelize()
    def tensor_eq(self, other):
        return self == other

    @method_treelize()
    def tensor_ne(self, other):
        return self != other

    def all(self):
        return self.reduce(lambda **kws: all(kws.values()))

    def any(self):
        return self.reduce(lambda **kws: any(kws.values()))
