import builtins
import io
import os
from functools import partial
from typing import Optional, Tuple, Callable
from typing import Type

from treevalue import func_treelize as original_func_treelize
from treevalue import general_tree_value, TreeValue
from treevalue.tree.common import BaseTree
from treevalue.tree.tree.tree import get_data_property
from treevalue.utils import post_process

from ..utils import replaceable_partial, args_mapping

__all__ = [
    'BaseTreeStruct',
    'print_tree', 'clsmeta',
]


def print_tree(tree: TreeValue, repr_: Callable = str,
               ascii_: bool = False, show_node_id: bool = True, file=None):
    """
    Overview:
        Print a tree structure to the given file.

    Arguments:
        - tree (:obj:`TreeValue`): Given tree object.
        - repr\\_ (:obj:`Callable`): Representation function, default is ``str``.
        - ascii\\_ (:obj:`bool`): Use ascii to print the tree, default is ``False``.
        - show_node_id (:obj:`bool`): Show node id of the tree, default is ``True``.
        - file: Output file of this print procedure, default is ``None`` which means to stdout. 
    """
    print_to_file = partial(builtins.print, file=file)
    node_ids = {}
    if ascii_:
        _HORI, _VECT, _CROS, _SROS = '|', '-', '+', '+'
    else:
        _HORI, _VECT, _CROS, _SROS = '\u2502', '\u2500', '\u251c', '\u2514'

    def _print_layer(node, path: Tuple[str, ...], prefixes: Tuple[str, ...],
                     current_key: Optional[str] = None, is_last_key: bool = True):
        # noinspection PyShadowingBuiltins
        def print(*args, pid: Optional[int] = -1, **kwargs, ):
            if pid is not None:
                print_to_file(prefixes[pid], end='')
            print_to_file(*args, **kwargs)

        _need_iter = True
        if isinstance(node, TreeValue):
            _node_id = id(get_data_property(node).actual())
            if show_node_id:
                _content = f'<{node.__class__.__name__} {hex(_node_id)}>'
            else:
                _content = f'<{node.__class__.__name__}>'
            if _node_id in node_ids.keys():
                _str_old_path = '.'.join(('<root>', *node_ids[_node_id]))
                _content = f'{_content}{os.linesep}(The same address as {_str_old_path})'
                _need_iter = False
            else:
                node_ids[_node_id] = path
                _need_iter = True
        else:
            _content = repr_(node)
            _need_iter = False

        if current_key:
            _key_arrow = f'{current_key} --> '
            _appended_prefix = (_HORI if _need_iter and len(node) > 0 else ' ') + ' ' * (len(_key_arrow) - 1)
            for index, line in enumerate(_content.splitlines()):
                if index == 0:
                    print(f'{_CROS if not is_last_key else _SROS}{_VECT * 2} {_key_arrow}', pid=-2, end='')
                else:
                    print(_appended_prefix, end='')
                print(line, pid=None)
        else:
            print(_content)

        if _need_iter:
            _length = len(node)
            for index, (key, value) in enumerate(sorted(node)):
                _is_last_line = index + 1 >= _length
                _new_prefixes = (*prefixes, prefixes[-1] + f'{_HORI if not _is_last_line else " "}   ')
                _new_path = (*path, key)
                _print_layer(value, _new_path, _new_prefixes, key, _is_last_line)

    if isinstance(tree, TreeValue):
        _print_layer(tree, (), ('', '',))
    else:
        print(repr_(tree), file=file)


class BaseTreeStruct(general_tree_value()):
    """
    Overview:
        Base structure of all the trees in ``treetensor``.
    """

    def __repr__(self):
        """
        Get the tree-based representation format of this object.

        Examples::

            >>> from treetensor.common import Object
            >>> repr(Object(1))  # Object is subclass of BaseTreeStruct
            '1'

            >>> repr(Object({'a': 1, 'b': 2, 'x': {'c': 3, 'd': 4}}))
            '<Object 0x7fe00b121220>\n├── a --> 1\n├── b --> 2\n└── x --> <Object 0x7fe00b121c10>\n    ├── c --> 3\n    └── d --> 4\n'

            >>> Object({'a': 1, 'b': 2, 'x': {'c': 3, 'd': 4}})
            <Object 0x7fe00b1271c0>
            ├── a --> 1
            ├── b --> 2
            └── x --> <Object 0x7fe00b127910>
                ├── c --> 3
                └── d --> 4
        """
        with io.StringIO() as sfile:
            print_tree(self, repr_=repr, ascii_=False, file=sfile)
            return sfile.getvalue()

    def __str__(self):
        """
        The same as :py:meth:`BaseTreeStruct.__repr__`.
        """
        return self.__repr__()


def clsmeta(func, allow_dict: bool = False) -> Type[type]:
    """
    Overview:
        Create a metaclass based on generating function.

        Used in :py:class:`treetensor.common.Object`,
        :py:class:`treetensor.torch.Tensor` and :py:class:`treetensor.torch.Size`.
        Can do modify onto the constructor function of the classes.

    Arguments:
        - func: Generating function.
        - allow_dict (:obj:`bool`): Auto transform dictionary to :py:class:`treevalue.TreeValue` class, \
                                    default is ``False``.
    Returns:
        - metaclass (:obj:`Type[type]`): Metaclass for creating a new class.
    """

    class _TempTreeValue(TreeValue):
        pass

    _types = (
        TreeValue, BaseTree,
        *((dict,) if allow_dict else ()),
    )
    func_treelize = post_process(post_process(args_mapping(
        lambda i, x: TreeValue(x) if isinstance(x, _types) else x)))(
        replaceable_partial(original_func_treelize, return_type=_TempTreeValue)
    )

    _wrapped_func = func_treelize()(func)

    class _MetaClass(type):
        def __call__(cls, *args, **kwargs):
            _result = _wrapped_func(*args, **kwargs)
            if isinstance(_result, _TempTreeValue):
                return type.__call__(cls, _result)
            else:
                return _result

    return _MetaClass
