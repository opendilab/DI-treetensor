from typing import Type

from treevalue import typetrans, TreeValue

from ..common import BaseTreeStruct

__all__ = ['Torch']


class Torch(BaseTreeStruct):
    pass


def _auto_torch(value, cls: Type[Torch]):
    return typetrans(value, cls) if isinstance(value, TreeValue) else value
