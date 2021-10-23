import inspect
from functools import wraps, lru_cache
from types import MethodType

from hbutils.reflection import post_process
from treevalue import method_treelize, TreeValue

from .trees import auto_tree
from .wrappers import return_self
from ..utils import doc_from_base as original_doc_from_base
from ..utils import replaceable_partial

__all__ = [
    'get_tree_proxy',
]


def get_tree_proxy(base, cls_mapper=None):
    doc_from_base = replaceable_partial(original_doc_from_base, base=base)
    outer_frame = inspect.currentframe().f_back
    outer_module = outer_frame.f_globals.get('__name__', None)

    class _TreeClassProxy:
        def __init__(self, cls):
            self.__cls = cls

        @lru_cache()
        def __getattr__(self, name):
            if hasattr(base, name) and not name.startswith('_') \
                    and callable(getattr(base, name)):
                _origin_func = getattr(base, name)
                return_self_deco = return_self if name.endswith('_') else (lambda x: x)
                auto_tree_cls = replaceable_partial(auto_tree, cls=cls_mapper or self.__cls)

                @doc_from_base()
                @return_self_deco
                @post_process(auto_tree_cls)
                @method_treelize(return_type=TreeValue, rise=True)
                @wraps(_origin_func, assigned=('__name__',), updated=())
                def _new_func(*args, **kwargs):
                    return _origin_func(*args, **kwargs)

                _new_func.__qualname__ = f'{self.__cls.__name__}.{name}'
                _new_func.__module__ = outer_module
                return _new_func
            else:
                raise AttributeError(f'Function {repr(name)} not found in {repr(base)}')

    class _TreeInstanceProxy:
        def __init__(self, proxy, s):
            self.__proxy = proxy
            self.__self = s

        @lru_cache()
        def __getattr__(self, name):
            return MethodType(getattr(self.__proxy, name), self.__self)

    return _TreeClassProxy, _TreeInstanceProxy
