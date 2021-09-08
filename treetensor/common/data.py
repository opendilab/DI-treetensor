import operator

from treevalue import func_treelize

from .base import BaseTreeStruct

_OPERATORS = {}
for _op_name in getattr(operator, '__all__'):
    _OPERATORS[_op_name] = func_treelize()(getattr(operator, _op_name))


class TreeData(BaseTreeStruct):
    def __le__(self, other):
        return _OPERATORS['le'](self, other)

    def __lt__(self, other):
        return _OPERATORS['lt'](self, other)

    def __ge__(self, other):
        return _OPERATORS['ge'](self, other)

    def __gt__(self, other):
        return _OPERATORS['gt'](self, other)

    def __eq__(self, other):
        return _OPERATORS['eq'](self, other)

    def __ne__(self, other):
        return _OPERATORS['ne'](self, other)
