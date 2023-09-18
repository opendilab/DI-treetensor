import torch

from .base import doc_from_base, wrap_for_treelize, _is_torch_2

__all__ = [
    'vmap',
]

if _is_torch_2:
    @doc_from_base()
    @wrap_for_treelize()
    def vmap(func, *args, **kwargs):
        return torch.vmap(func, *args, **kwargs)

else:
    def vmap(func, *args, **kwargs):
        """
        .. warning:
            :method:`treetensor.torch.vmap` is not supported for torch 1.x.
        """
        raise NotImplementedError(f'Function vmap is not supported in torch {torch.__version__}.')
