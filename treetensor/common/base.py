from abc import ABCMeta
from functools import lru_cache

from treevalue import general_tree_value


@lru_cache()
def _merge_func(red):
    return lambda **kws: red(kws.values())


class BaseTreeStruct(general_tree_value(), metaclass=ABCMeta):
    def all(self) -> bool:
        return self.reduce(_merge_func(all))

    def any(self) -> bool:
        return self.reduce(_merge_func(any))

    def sum(self):
        return self.reduce(_merge_func(sum))
