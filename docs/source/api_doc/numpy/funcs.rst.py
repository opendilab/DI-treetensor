import codecs
import inspect
import os
import re

import numpy as np

import treetensor.numpy as tnp
from docs import print_title, current_module, get_origin, print_block, print_doc

_DOC_FROM_TAG = '__doc_from__'
_H2_PATTERN = re.compile('-{3,}')
_numpy_version = np.__version__
_short_version = '.'.join(_numpy_version.split('.')[:2])


def _doc_process(doc: str) -> str:
    _doc = doc
    _doc = re.sub(
        '( *)([^\\n]+)\\n( *)(-{3,})\\n',
        lambda x: x.group(2) + '\n' + '~' * len(x.group(4)) + '\n',
        _doc
    )
    _doc = _doc.replace(' : ', ' \\: ')
    return _doc


if __name__ == '__main__':
    print_title(f'Common Functions', levelc='=')
    current_module(tnp.__name__)

    with print_block('toctree', params=dict(maxdepth=1)) as top_toctree:
        for _name in sorted(tnp.funcs.__all__):
            _file_tag = f'funcs.{_name}.auto'
            _filename = f'{_file_tag}.rst'
            print(_file_tag, file=top_toctree)

            _item = getattr(tnp.funcs, _name)
            _origin = get_origin(_item)
            with codecs.open(_filename, 'w') as p_func:
                print_title(_name, levelc='=', file=p_func)
                current_module(tnp.__name__, file=p_func)

                print_title("Documentation", levelc='-', file=p_func)
                with print_block('autofunction', value=_name, file=p_func):
                    pass

                if _origin:
                    _has_origin_doc = (_origin.__doc__ or '').lstrip()
                    with print_block('admonition', value='Numpy Version Related',
                                     params={'class': 'tip'}, file=p_func) as f_note:
                        print_doc(f"""
This documentation is based on 
`numpy.{_name} <https://numpy.org/doc/{_short_version}/reference/generated/numpy.{_name}.html>`_ 
in `numpy v{_numpy_version} <https://numpy.org/doc/{_short_version}/>`_.
**Its arguments\' arrangements depend on the version of numpy you installed**.

If some arguments listed here are not working properly, please check your numpy's version 
with the following command and find its documentation.

.. code-block:: shell
    :linenos:

    python -c 'import numpy as np;print(np.__version__)'

                    """, file=f_note)
                        if _has_origin_doc:
                            print_doc(f"The arguments and keyword arguments supported in "
                                      f"numpy v{_numpy_version} is listed below.", file=f_note)
                    print(file=p_func)

                    if _has_origin_doc:
                        print_title(f"Description From Numpy v{_short_version}", levelc='-', file=p_func)
                        current_module(np.__name__, file=p_func)

                        _origin_doc = _doc_process(_origin.__doc__ or "").lstrip()
                        _doc_lines = _origin_doc.splitlines()
                        _first_line, _other_lines = _doc_lines[0], _doc_lines[1:]
                        if _first_line.strip():
                            _signature = _first_line
                            _main_doc = os.linesep.join(_other_lines)
                        else:
                            _signature = f'{_origin.__name__}{inspect.signature(_origin)}'
                            _main_doc = _origin_doc

                        print_doc(f'.. function:: {_signature}{os.linesep}'
                                  f'{os.linesep}'
                                  f'{_main_doc}', file=p_func)

                print(file=p_func)
