import pytest
import torch
from treevalue import func_treelize, typetrans, TreeValue

import treetensor.torch as ttorch
from treetensor.common import Object

_all_is = func_treelize(return_type=ttorch.Tensor)(lambda x, y: x is y)


@pytest.mark.unittest
class TestTorchSize:
    def test_init(self):
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

    def test_numel(self):
        assert ttorch.Size({
            'a': [1, 2, 3],
            'b': {'x': [3, 4, ]},
            'c': [5],
        }).numel() == 23

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

    def test_count(self):
        assert ttorch.Size({
            'a': [1, 2, 3],
            'b': {'x': [3, 4, ]},
            'c': [5],
        }).count(3) == 2
