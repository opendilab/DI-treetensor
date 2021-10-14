import builtins

from treevalue import method_treelize

from .trees import BaseTreeStruct, clsmeta
from .wrappers import ireduce

__all__ = [
    "Object",
]


def _object(obj):
    return obj


class Object(BaseTreeStruct, metaclass=clsmeta(_object, allow_dict=True)):
    """
    Overview:
        Generic object tree class, used in :py:mod:`treetensor.numpy` and :py:mod:`treetensor.torch`.
    """

    def __init__(self, data):
        """
        In :class:`treetensor.common.Object`, object or object tree can be initialized.

        Examples::

            >>> from treetensor.common import Object
            >>> Object(1)
            1

            >>> Object({'a': 1, 'b': 2, 'x': {'c': 233}})
            <Object 0x7fe00b1153a0>
            ├── a --> 1
            ├── b --> 2
            └── x --> <Object 0x7fe00b115ee0>
                └── c --> 233
        """
        BaseTreeStruct.__init__(self, data)

    @ireduce(builtins.all, piter=list)
    @method_treelize()
    def all(self):
        """
        The values in this tree is all true or not.

        Examples::

            >>> from treetensor.common import Object
            >>> Object({'a': False, 'b': {'x': False}}).all()
            False
            >>> Object({'a': True, 'b': {'x': False}}).all()
            False
            >>> Object({'a': True, 'b': {'x': True}}).all()
            True

        """
        return not not self

    @ireduce(builtins.any, piter=list)
    @method_treelize()
    def any(self):
        """
        The values in this tree is not all False or yes.

        Examples::

            >>> from treetensor.common import Object
            >>> Object({'a': False, 'b': {'x': False}}).any()
            False
            >>> Object({'a': True, 'b': {'x': False}}).any()
            True
            >>> Object({'a': True, 'b': {'x': True}}).any()
            True

        """
        return not not self
