from treevalue import method_treelize, TreeValue
from treevalue.utils import post_process

from ..base import Torch, auto_torch
from ..tensor import Tensor
from ...utils import replaceable_partial

auto_tensor = replaceable_partial(auto_torch, cls=Tensor)


class TensorMethod(Torch):
    @post_process(auto_tensor)
    @method_treelize(return_type=TreeValue, rise=True)
    def __call__(self, *args, **kwargs):
        return self(*args, **kwargs)
