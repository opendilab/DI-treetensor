import codecs

import torch

import treetensor.torch as ttorch
from docs import print_title, current_module, get_origin, print_block, print_doc

_DOC_FROM_TAG = '__doc_from__'
_torch_version = torch.__version__

if __name__ == '__main__':
    print_title(f'Common Functions', levelc='=')
    current_module(ttorch.__name__)

    with print_block('toctree', params=dict(maxdepth=1)) as top_toctree:
        for _name in sorted(ttorch.funcs.__all__):
            _file_tag = f'funcs.{_name}.auto'
            _filename = f'{_file_tag}.rst'
            print(_file_tag, file=top_toctree)

            _item = getattr(ttorch.funcs, _name)
            _origin = get_origin(_item)
            with codecs.open(_filename, 'w') as p_func:
                print_title(_name, levelc='=', file=p_func)
                current_module(ttorch.__name__, file=p_func)

                print_title("Documentation", levelc='-', file=p_func)
                with print_block('autofunction', value=_name, file=p_func):
                    pass

                if _origin:
                    _has_origin_doc = (_origin.__doc__ or '').lstrip()
                    with print_block('admonition', value='Torch Version Related',
                                     params={'class': 'tip'}, file=p_func) as f_note:
                        print_doc(f"""
This documentation is based on
`torch.{_name} <https://pytorch.org/docs/{_torch_version}/generated/torch.{_name}.html>`_
in `torch v{_torch_version} <https://pytorch.org/docs/{_torch_version}/>`_.
**Its arguments\' arrangements depend on the version of pytorch you installed**.

If some arguments listed here are not working properly, please check your pytorch's version 
with the following command and find its documentation.

.. code-block:: shell
    :linenos:

    python -c 'import torch;print(torch.__version__)'

                    """, file=f_note)
                        if _has_origin_doc:
                            print_doc(f"The arguments and keyword arguments supported in "
                                      f"torch v{_torch_version} is listed below.", file=f_note)
                    print(file=p_func)

                    if _has_origin_doc:
                        print_title(f"Description From Torch v{_torch_version}", levelc='-', file=p_func)
                        current_module(torch.__name__, file=p_func)
                        with print_block('autofunction', f'{torch.__name__}.{_origin.__name__}',
                                         file=p_func):
                            pass

                print(file=p_func)
