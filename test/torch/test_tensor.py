import numpy as np
import pytest
import torch
from treevalue import func_treelize, typetrans, TreeValue

import treetensor.numpy as tnp
import treetensor.torch as ttorch

_all_is = func_treelize(return_type=ttorch.Tensor)(lambda x, y: x is y)


@pytest.mark.unittest
class TestTorchTensor:
    _DEMO_1 = ttorch.Tensor({
        'a': [[1, 2, 3], [4, 5, 6]],
        'b': [[1, 2], [5, 6]],
        'x': {
            'c': [3, 5, 6, 7],
            'd': [[[1, 2], [8, 9]]],
        }
    })

    _DEMO_2 = ttorch.Tensor({
        'a': [[1, 2, 3], [4, 5, 6]],
        'b': [[1, 2], [5, 60]],
        'x': {
            'c': [3, 5, 6, 7],
            'd': [[[1, 2], [8, 9]]],
        }
    })

    def test_init(self):
        assert (ttorch.Tensor([1, 2, 3]) == torch.tensor([1, 2, 3])).all()
        assert (ttorch.Tensor([1, 2, 3], dtype=torch.float32) == torch.FloatTensor([1, 2, 3])).all()
        assert (self._DEMO_1 == typetrans(TreeValue({
            'a': ttorch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': ttorch.tensor([[1, 2], [5, 6]]),
            'x': {
                'c': ttorch.tensor([3, 5, 6, 7]),
                'd': ttorch.tensor([[[1, 2], [8, 9]]]),
            }
        }), ttorch.Tensor)).all()

    def test_numel(self):
        assert self._DEMO_1.numel() == 18

    def test_numpy(self):
        assert tnp.all(self._DEMO_1.numpy() == tnp.ndarray({
            'a': np.array([[1, 2, 3], [4, 5, 6]]),
            'b': np.array([[1, 2], [5, 6]]),
            'x': {
                'c': np.array([3, 5, 6, 7]),
                'd': np.array([[[1, 2], [8, 9]]]),
            }
        }))

    def test_cpu(self):
        assert ttorch.all(self._DEMO_1.cpu() == self._DEMO_1)
        assert _all_is(self._DEMO_1.cpu(), self._DEMO_1).reduce(lambda **kws: all(kws.values()))

    def test_to(self):
        assert ttorch.all(self._DEMO_1.to(torch.float32) == ttorch.Tensor({
            'a': torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.FloatTensor([[1, 2], [5, 6]]),
            'x': {
                'c': torch.FloatTensor([3, 5, 6, 7]),
                'd': torch.FloatTensor([[[1, 2], [8, 9]]]),
            }
        }))

    def test_all(self):
        assert (self._DEMO_1 == self._DEMO_1).all()
        assert not (self._DEMO_1 == self._DEMO_2).all()

    def test_tolist(self):
        pass
