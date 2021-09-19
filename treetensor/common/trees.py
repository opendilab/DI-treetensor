import builtins
import io
import os
from functools import partial
from typing import Optional, Tuple, Callable

from treevalue import func_treelize as original_func_treelize
from treevalue import general_tree_value, TreeValue
from treevalue.tree.common import BaseTree
from treevalue.tree.tree.tree import get_data_property
from treevalue.utils import post_process

from ..utils import replaceable_partial, args_mapping

__all__ = [
    'BaseTreeStruct', "Object",
    'print_tree', 'clsmeta',
]


def _tree_title(node: TreeValue):
    _tree = get_data_property(node)
    return "<{cls} {id}>".format(
        cls=node.__class__.__name__,
        id=hex(id(_tree.actual())),
    )


def print_tree(tree: TreeValue, repr_: Callable = str, ascii_: bool = False, file=None):
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
            _content = f'<{node.__class__.__name__} {hex(_node_id)}>'
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
        with io.StringIO() as sfile:
            print_tree(self, repr_=repr, ascii_=False, file=sfile)
            return sfile.getvalue()

    def __str__(self):
        return self.__repr__()


def clsmeta(cls: type, allow_dict: bool = False, allow_data: bool = True):
    class _TempTreeValue(TreeValue):
        pass

    _types = (
        TreeValue,
        *((dict,) if allow_dict else ()),
        *((BaseTree,) if allow_data else ()),
    )
    func_treelize = post_process(post_process(args_mapping(
        lambda i, x: TreeValue(x) if isinstance(x, _types) else x)))(
        replaceable_partial(original_func_treelize, return_type=_TempTreeValue)
    )

    _torch_size = func_treelize()(cls)

    class _MetaClass(type):
        def __call__(cls, *args, **kwargs):
            _result = _torch_size(*args, **kwargs)
            if isinstance(_result, _TempTreeValue):
                return type.__call__(cls, _result)
            else:
                return _result

    return _MetaClass


class Object(BaseTreeStruct):
    pass
