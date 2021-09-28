from typing import Type

from treevalue import TreeValue, typetrans

from ...common import BaseTreeStruct

__all__ = ['Torch', 'auto_torch']


class Torch(BaseTreeStruct):
    pass


# noinspection PyArgumentList
def auto_torch(v, cls: Type[Torch]):
    if isinstance(v, TreeValue):
        return typetrans(v, cls)
    elif isinstance(v, (tuple, list, set)):
        return type(v)((auto_torch(item, cls) for item in v))
    elif isinstance(v, dict):
        return type(v)({key: auto_torch(value, cls) for key, value in v.items()})
    else:
        return v
