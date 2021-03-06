import codecs
import importlib

import torch
from treevalue import TreeValue

import treetensor.torch as ttorch
from docs import print_title, current_module, print_block, get_origin, print_doc

_DOC_FROM_TAG = '__doc_from__'
_DOC_TAG = '__doc_names__'
_ttorch_tensor = importlib.import_module('treetensor.torch.tensor')
_torch_version = torch.__version__

if __name__ == '__main__':
    print_title("Tree Classes", levelc='=')
    current_module(ttorch.__name__)

    trees = filter(lambda x: isinstance(x, type) and issubclass(x, TreeValue),
                   map(lambda n: getattr(ttorch, n), ttorch.__all__))
    with print_block('toctree', params=dict(maxdepth=2)) as top_toctree:
        for _tree in trees:
            _class_file_tag = f'trees.{_tree.__name__}.auto'
            print(_class_file_tag, file=top_toctree)

            with codecs.open(f'{_class_file_tag}.rst', 'w') as p_class:
                print_title(_tree.__name__, levelc='=', file=p_class)
                current_module(ttorch.__name__, file=p_class)

                with print_block('toctree', params=dict(maxdepth=1), file=p_class) as cls_toctree:
                    for _name in sorted(getattr(_tree, _DOC_TAG, [])):
                        _item_file_tag = f'trees.{_tree.__name__}.{_name}.auto'
                        print(_item_file_tag, file=cls_toctree)

                        with codecs.open(f'{_item_file_tag}.rst', 'w') as p_item:
                            _item = getattr(_tree, _name)
                            _origin = get_origin(_item)

                            _has_own_doc = (_item.__doc__ or '').lstrip()
                            _has_origin_doc = (_origin is not None) and (_origin.__doc__ or '').strip()
                            print_title(
                                _name + (
                                    (' [Reprint Only]' if _has_origin_doc else ' [No Doc]')
                                    if not _has_own_doc else ''
                                ), levelc='=', file=p_item
                            )
                            current_module(ttorch.__name__, file=p_item)

                            print_title(f"Documentation", levelc='-', file=p_item)
                            with print_block('autoclass', value=_tree.__name__, file=p_item) as f_class:

                                with print_block(
                                        'autoproperty' if isinstance(_item, property) else 'automethod',
                                        value=_name, file=f_class):
                                    pass

                            if not _has_own_doc:
                                with print_block(
                                        'admonition',
                                        value=('Reprinted Documentation Only'
                                        if _has_origin_doc else "No Documentation"),
                                        params={'class': 'warning' if _has_origin_doc else "error"},
                                        file=p_item
                                ) as f_nodoc:
                                    print_doc('No ``__doc__`` is provided yet, need to be completed.', file=f_nodoc)

                            if _origin:
                                with print_block('admonition', value='Torch Version Related',
                                                 params={'class': 'tip'}, file=p_item) as f_note:
                                    print_doc(f"""
This documentation is based on
`torch.{_tree.__name__}.{_name} <https://pytorch.org/docs/{_torch_version}/tensors.html#torch.{_tree.__name__}.{_name}>`_
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
                                print(file=p_item)

                                if _has_origin_doc:
                                    print_title(f"Description From Torch v{_torch_version}", levelc='-',
                                                file=p_item)
                                    current_module(torch.__name__, file=p_item)
                                    with print_block('autoclass', f'{torch.__name__}.{_tree.__name__}',
                                                     dict(members=','.join([_name])), file=p_item):
                                        pass

                            print(file=p_item)
