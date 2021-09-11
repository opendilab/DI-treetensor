import torch
from treevalue import func_treelize as original_func_treelize

from .tensor import TreeTensor, tireduce
from ..common import TreeObject
from ..utils import replaceable_partial

func_treelize = replaceable_partial(original_func_treelize, return_type=TreeTensor)

__all__ = [
    'zeros', 'zeros_like',
    'randn', 'randn_like',
    'randint', 'randint_like',
    'ones', 'ones_like',
    'full', 'full_like',
    'empty', 'empty_like',
    'all', 'any',
    'eq', 'equal',
]


@func_treelize()
def zeros(size, *args, **kwargs):
    return torch.zeros(*size, *args, **kwargs)


@func_treelize()
def zeros_like(input_, *args, **kwargs):
    return torch.zeros_like(input_, *args, **kwargs)


@func_treelize()
def randn(size, *args, **kwargs):
    return torch.randn(*size, *args, **kwargs)


@func_treelize()
def randn_like(input_, *args, **kwargs):
    return torch.randn_like(input_, *args, **kwargs)


@func_treelize()
def randint(size, *args, **kwargs):
    return torch.randint(*args, size, **kwargs)


@func_treelize()
def randint_like(input_, *args, **kwargs):
    return torch.randint_like(input_, *args, **kwargs)


@func_treelize()
def ones(size, *args, **kwargs):
    return torch.ones(*size, *args, **kwargs)


@func_treelize()
def ones_like(input_, *args, **kwargs):
    return torch.ones_like(input_, *args, **kwargs)


@func_treelize()
def full(size, *args, **kwargs):
    return torch.full(size, *args, **kwargs)


@func_treelize()
def full_like(input_, *args, **kwargs):
    return torch.full_like(input_, *args, **kwargs)


@func_treelize()
def empty(size, *args, **kwargs):
    return torch.empty(size, *args, **kwargs)


@func_treelize()
def empty_like(input_, *args, **kwargs):
    return torch.empty_like(input_, *args, **kwargs)


@tireduce(torch.all)
@func_treelize(return_type=TreeObject)
def all(input_, *args, **kwargs):
    return torch.all(input_, *args, **kwargs)


@tireduce(torch.any)
@func_treelize(return_type=TreeObject)
def any(input_, *args, **kwargs):
    return torch.any(input_, *args, **kwargs)


@func_treelize()
def eq(input_, other, *args, **kwargs):
    return torch.eq(input_, other, *args, **kwargs)


@func_treelize()
def equal(input_, other, *args, **kwargs):
    return torch.equal(input_, other, *args, **kwargs)
