import torch

from .base import doc_from_base, wrap_for_treelize

__all__ = [
    'vmap',
]


@doc_from_base()
@wrap_for_treelize()
def vmap(func, *args, **kwargs):
    return torch.vmap(func, *args, **kwargs)
