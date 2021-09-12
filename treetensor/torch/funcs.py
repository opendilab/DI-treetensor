import builtins
from typing import List

import torch
from treevalue import func_treelize as original_func_treelize
from treevalue.utils import post_process

from .tensor import TreeTensor, tireduce
from ..common import TreeObject, ireduce
from ..utils import replaceable_partial, direct_doc, inherit_doc


def _doc_stripper(src, _, lines: List[str]):
    _name, _version = src.__name__, torch.__version__
    if lines:
        lines[0] = f'.. function:: {lines[0]}'

    return [
        f'.. note::',
        f'',
        f'    This documentation is based on '
        f'    `torch.{_name} <https://pytorch.org/docs/{_version}/generated/torch.{_name}.html>`_ '
        f'    in `torch v{_version} <https://pytorch.org/docs/{_version}/>`_.',
        f'    **Its arguments\' arrangements depend on the version of pytorch you installed**.',
        f'',
        *lines,
    ]


func_treelize = replaceable_partial(original_func_treelize, return_type=TreeTensor)
docs = post_process(post_process(direct_doc))(replaceable_partial(inherit_doc, stripper=_doc_stripper))

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


@docs(torch.zeros)
@func_treelize()
def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


@docs(torch.zeros_like)
@func_treelize()
def zeros_like(input_, *args, **kwargs):
    return torch.zeros_like(input_, *args, **kwargs)


@docs(torch.randn)
@func_treelize()
def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs)


@docs(torch.randn_like)
@func_treelize()
def randn_like(input_, *args, **kwargs):
    return torch.randn_like(input_, *args, **kwargs)


@docs(torch.randint)
@func_treelize()
def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


@docs(torch.randint_like)
@func_treelize()
def randint_like(input_, *args, **kwargs):
    return torch.randint_like(input_, *args, **kwargs)


@docs(torch.ones)
@func_treelize()
def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@docs(torch.ones_like)
@func_treelize()
def ones_like(input_, *args, **kwargs):
    return torch.ones_like(input_, *args, **kwargs)


@docs(torch.full)
@func_treelize()
def full(*args, **kwargs):
    return torch.full(*args, **kwargs)


@docs(torch.full_like)
@func_treelize()
def full_like(input_, *args, **kwargs):
    return torch.full_like(input_, *args, **kwargs)


@docs(torch.empty)
@func_treelize()
def empty(*args, **kwargs):
    return torch.empty(*args, **kwargs)


@docs(torch.empty_like)
@func_treelize()
def empty_like(input_, *args, **kwargs):
    return torch.empty_like(input_, *args, **kwargs)


@docs(torch.all)
@tireduce(torch.all)
@func_treelize(return_type=TreeObject)
def all(input_, *args, **kwargs):
    return torch.all(input_, *args, **kwargs)


@docs(torch.any)
@tireduce(torch.any)
@func_treelize(return_type=TreeObject)
def any(input_, *args, **kwargs):
    return torch.any(input_, *args, **kwargs)


@docs(torch.eq)
@func_treelize()
def eq(input_, other, *args, **kwargs):
    return torch.eq(input_, other, *args, **kwargs)


@docs(torch.equal)
@ireduce(builtins.all)
@func_treelize()
def equal(input_, other, *args, **kwargs):
    return torch.equal(input_, other, *args, **kwargs)
