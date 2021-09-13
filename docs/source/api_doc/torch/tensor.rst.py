import importlib

import torch

import treetensor.torch as ttorch
from docs import print_title, current_module, print_block, get_origin, PropertyType, print_doc

_DOC_FROM_TAG = '__doc_from__'
_DOC_TAG = '__doc_names__'
_ttorch_tensor = importlib.import_module('treetensor.torch.tensor')
_torch_version = torch.__version__

if __name__ == '__main__':
    print_title(_ttorch_tensor.__name__, levelc='=')
    current_module(ttorch.__name__)
    with print_block('automodule', value=_ttorch_tensor.__name__):
        pass

    print_title(ttorch.Tensor.__name__, levelc='-')

    with print_block('autoclass', value=ttorch.Tensor.__name__) as tf:
        for _name in sorted(getattr(ttorch.Tensor, _DOC_TAG, [])):
            _item = getattr(ttorch.Tensor, _name)
            _origin = get_origin(_item)

            with print_block(
                    'autoproperty' if isinstance(_item, PropertyType) else 'automethod',
                    value=_name, file=tf
            ) as f:
                pass

            if _origin and (_origin.__doc__ or '').strip():
                with print_block('admonition', value='Torch Version Related', params={'class': 'tip'}, file=tf) as f:
                    print_doc(f"""
            This documentation is based on
            `torch.Tensor.{_name} <https://pytorch.org/docs/{_torch_version}/tensors.html#torch.Tensor.{_name}>`_
            in `torch v{_torch_version} <https://pytorch.org/docs/{_torch_version}/>`_.
            **Its arguments\' arrangements depend on the version of pytorch you installed**.

            If some arguments listed here are not working properly, please check your pytorch's version 
            with the following command and find its documentation.

            .. code-block:: shell
                :linenos:

                python -c 'import torch;print(torch.__version__)'

            The arguments and keyword arguments supported in torch v{_torch_version} is listed below.

                        """, file=f)
                print(file=tf)

                with print_block('', file=tf) as f:
                    print_doc(f'.. function:: {_origin.__doc__.lstrip()}', file=f)

            print(file=tf)
