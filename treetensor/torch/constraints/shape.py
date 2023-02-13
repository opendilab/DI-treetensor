import torch

from ...common.constraints import ShapePrefixConstraint
from ...common.constraints import shape_prefix as _origin_shape_prefix

__all__ = [
    'TensorShapePrefixConstraint', 'shape_prefix',
]


class TensorShapePrefixConstraint(ShapePrefixConstraint):
    __type__ = torch.Tensor


def shape_prefix(*shape):
    return _origin_shape_prefix(*shape, type_=TensorShapePrefixConstraint)
