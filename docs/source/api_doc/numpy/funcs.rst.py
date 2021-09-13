import re

import numpy as np

import treetensor.numpy as tnp
from docs import print_title, current_module, get_origin, print_block, print_doc

_DOC_FROM_TAG = '__doc_from__'
_H2_PATTERN = re.compile('-{3,}')
_numpy_version = np.__version__
_short_version = '.'.join(_numpy_version.split('.')[:2])


def _raw_doc_process(doc: str) -> str:
    _doc = _H2_PATTERN.sub(lambda x: '~' * len(x.group(0)), doc)
    _doc = _doc.replace(' : ', ' \\: ')
    return _doc


if __name__ == '__main__':
    print_title(tnp.funcs.__name__, levelc='=')
    current_module(tnp.funcs.__name__)

    for _name in sorted(tnp.funcs.__all__):
        _item = getattr(tnp.funcs, _name)
        _origin = get_origin(_item)
        print_title(_name, levelc='-')

        with print_block('autofunction', value=_name):
            pass

        if _origin and (_origin.__doc__ or '').strip():
            with print_block('admonition', value='Numpy Version Related', params={'class': 'tip'}) as f:
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

The arguments and keyword arguments supported in numpy v{_numpy_version} is listed below.

            """, file=f)
            print()

            print_doc(_raw_doc_process(_origin.__doc__ or ''))
            print()
