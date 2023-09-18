import torch
from hbutils.testing import vpip

from .base import doc_from_base, wrap_for_treelize

__all__ = [
    'vmap',
]

_is_torch_2 = vpip('torch') >= '2'


@doc_from_base()
@wrap_for_treelize()
def vmap(func, *args, **kwargs):
    if _is_torch_2:
        return torch.vmap(func, *args, **kwargs)
    else:
        raise NotImplementedError(f'Function vmap is not supported in torch {torch.__version__}.')
