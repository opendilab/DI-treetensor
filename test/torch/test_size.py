import pytest
import torch
from treevalue import typetrans, TreeValue

import treetensor.torch as ttorch
from treetensor.common import Object
from treetensor.utils import replaceable_partial
from ..tests import choose_mark_with_existence_check

choose_mark = replaceable_partial(choose_mark_with_existence_check, base=ttorch.Size)


class TestTorchSize:
    @choose_mark()
    def test___init__(self):
        t1 = ttorch.Size([1, 2, 3])
        assert isinstance(t1, torch.Size)
        assert t1 == torch.Size([1, 2, 3])

        t2 = ttorch.Size({
            'a': [1, 2, 3],
            'b': {'x': [3, 4, ]},
            'c': [5],
        })
        assert isinstance(t2, ttorch.Size)
        assert typetrans(t2, TreeValue) == TreeValue({
            'a': torch.Size([1, 2, 3]),
            'b': {'x': torch.Size([3, 4, ])},
            'c': torch.Size([5]),
        })

    @choose_mark()
    def test_numel(self):
        assert ttorch.Size({
            'a': [1, 2, 3],
            'b': {'x': [3, 4, ]},
            'c': [5],
        }).numel() == 23

    @choose_mark()
    def test_index(self):
        assert ttorch.Size({
            'a': [1, 2, 3],
            'b': {'x': [3, 4, ]},
            'c': [5],
        }).index(3) == Object({
            'a': 2,
            'b': {'x': 0},
            'c': None
        })

        with pytest.raises(ValueError):
            ttorch.Size({
                'a': [1, 2, 3],
                'b': {'x': [3, 4, ]},
                'c': [5],
            }).index(100)

    @choose_mark()
    def test_count(self):
        assert ttorch.Size({
            'a': [1, 2, 3],
            'b': {'x': [3, 4, ]},
            'c': [5],
        }).count(3) == 2
