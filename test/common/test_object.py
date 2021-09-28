import pytest
from treevalue import typetrans, TreeValue

from treetensor.common import Object


@pytest.mark.unittest
class TestCommonObject:
    def test_object(self):
        t = Object(1)
        assert isinstance(t, int)
        assert t == 1

        assert Object({'a': 1, 'b': 2}) == typetrans(TreeValue({
            'a': 1, 'b': 2
        }), Object)

    def test_all(self):
        assert not Object({'a': False, 'b': {'x': False}}).all()
        assert not Object({'a': True, 'b': {'x': False}}).all()
        assert Object({'a': True, 'b': {'x': True}}).all()

    def test_any(self):
        assert not Object({'a': False, 'b': {'x': False}}).any()
        assert Object({'a': True, 'b': {'x': False}}).any()
        assert Object({'a': True, 'b': {'x': True}}).any()
