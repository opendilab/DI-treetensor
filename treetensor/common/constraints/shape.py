from collections.abc import Sequence
from typing import Type, TypeVar, Optional

from treevalue.tree import ValueConstraint
from treevalue.tree.tree.constraint import TypeConstraint

__all__ = [
    'ShapePrefixConstraint',
    'shape_prefix',
]


class ShapePrefixConstraint(ValueConstraint, Sequence):
    __type__: Optional[type] = None

    def __init__(self, *prefix):
        ValueConstraint.__init__(self)
        self.__prefix = prefix

    @property
    def prefix(self):
        return self.__prefix

    def __getitem__(self, index):
        return self.__prefix[index]

    def __len__(self) -> int:
        return len(self.__prefix)

    def _validate_value(self, instance):
        if self.__type__ and not isinstance(instance, self.__type__):
            raise TypeError(f'Invalid type, {self.__type__.__name__!r} expected but {instance!r} found.')

        if not hasattr(instance, 'shape'):
            raise TypeError(f'Shape not found for instance {instance!r}.')
        shape = instance.shape
        if shape[:len(self.__prefix)] != self.__prefix:
            raise ValueError(f'Invalid shape prefix, {self.__prefix!r} expected but {shape!r} found.')

    def _features(self):
        return self.__prefix

    def _contains(self, other):
        if isinstance(other, ShapePrefixConstraint):
            return isinstance(self, type(other)) and self.__prefix[:len(other.__prefix)] == other.__prefix
        else:
            if self.__type__ and isinstance(other, TypeConstraint):
                return issubclass(self.__type__, other.type_)
            else:
                return False


_ShapePrefixType = TypeVar('_ShapePrefixType', bound=ShapePrefixConstraint)


def shape_prefix(*args, type_: Type[_ShapePrefixType] = ShapePrefixConstraint) -> _ShapePrefixType:
    return type_(*args)
