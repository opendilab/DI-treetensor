import torch

import treetensor.torch as ttorch
from docs import print_title, current_module, get_origin, print_block, print_doc

_DOC_FROM_TAG = '__doc_from__'

if __name__ == '__main__':
    _torch_version = torch.__version__
    print_title(ttorch.funcs.__name__, levelc='=')
    current_module(ttorch.funcs.__name__)

    for _name in sorted(ttorch.funcs.__all__):
        _item = getattr(ttorch.funcs, _name)
        _origin = get_origin(_item)
        print_title(_name, levelc='-')

        print_block('', 'autofunction', value=_name)

        if _origin and (_origin.__doc__ or '').strip():
            print_block(f"""
This documentation is based on
`torch.{_name} <https://pytorch.org/docs/{_torch_version}/generated/torch.{_name}.html>`_
in `torch v{_torch_version} <https://pytorch.org/docs/{_torch_version}/>`_.
**Its arguments\' arrangements depend on the version of pytorch you installed**.

If some arguments listed here are not working properly, please check your pytorch's version 
with the following command and find its documentation.

.. code-block:: shell
    :linenos:

    python -c 'import torch;print(torch.__version__)'

The arguments and keyword arguments supported in torch v{_torch_version} is listed below.

            """, 'admonition', value='Torch Version Related', params={'class': 'tip'})
            print_doc(f'.. function:: {_origin.__doc__.lstrip()}')

        print()
