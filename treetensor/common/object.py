from .trees import BaseTreeStruct, clsmeta

__all__ = [
    "Object",
]


def _object(obj):
    return obj


class Object(BaseTreeStruct, metaclass=clsmeta(_object, allow_dict=True)):
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
        super(BaseTreeStruct, self).__init__(data)
