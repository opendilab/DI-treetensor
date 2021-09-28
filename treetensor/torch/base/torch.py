from typing import Type

from treevalue import TreeValue, typetrans

from ...common import BaseTreeStruct

__all__ = ['Torch', 'auto_torch']


class Torch(BaseTreeStruct):
    pass


def auto_torch(value, cls: Type[Torch]):
    return typetrans(value, cls) if isinstance(value, TreeValue) else value
