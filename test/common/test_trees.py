import io

import pytest
import torch
from treevalue import typetrans, TreeValue, general_tree_value

from treetensor.common import Object, print_tree


def text_compares(expected, actual):
    elines = expected.splitlines()
    alines = actual.splitlines()
    assert len(elines) == len(alines), f"""Lines not match,
Expected: {len(elines)} lines
  Actual: {len(alines)} lines
"""

    for i, (e, a) in enumerate(zip(elines, alines)):
        assert e.rstrip() == a.rstrip(), f"""Line {i} not match,
Expected: {e}
  Actual: {a}
"""


@pytest.mark.unittest
class TestCommonTrees:
    def test_object(self):
        t = Object(1)
        assert isinstance(t, int)
        assert t == 1

        assert Object({'a': 1, 'b': 2}) == typetrans(TreeValue({
            'a': 1, 'b': 2
        }), Object)

    def test_print_tree(self):
        class _TempTree(general_tree_value()):
            def __repr__(self):
                with io.StringIO() as sfile:
                    print_tree(self, repr_=repr, ascii_=False, show_node_id=False, file=sfile)
                    return sfile.getvalue()

            def __str__(self):
                return self.__repr__()

        text_compares("""<_TempTree>
├── a --> 1                                                           
└── b --> 2
        """.rstrip(), str(_TempTree({
            'a': 1, 'b': 2
        })).rstrip())

        class _TmpObject:
            def __init__(self, v):
                self.__v = v

            def __repr__(self):
                return self.__v

        tx = _TempTree({
            'a': 1, 'b': 2, 'c': torch.tensor([[1, 2, ], [3, 4]]),
            'd': {'x': torch.tensor([[1], [2], [3, ], [4]])},
            'e': _TmpObject('line after this\nhahahaha'),
        })
        tx.d.y = tx
        text_compares("""<_TempTree>                                                
├── a --> 1                                                           
├── b --> 2                                              
├── c --> tensor([[1, 2],                                
│                 [3, 4]])                                   
├── d --> <_TempTree>                                   
│   ├── x --> tensor([[1],                              
│   │                 [2],                              
│   │                 [3],                              
│   │                 [4]]) 
│   └── y --> <_TempTree> 
│             (The same address as <root>)
└── e --> line after this                                     
          hahahaha
        """.rstrip(), str(tx).rstrip())

        with io.StringIO() as sf:
            print_tree(1, file=sf)
            text_compares("1", sf.getvalue())
