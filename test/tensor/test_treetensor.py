import numpy as np
import pytest
import torch
from treevalue import func_treelize

from treetensor.numpy import all_array_equal, TreeNumpy
from treetensor.tensor import TreeTensor, all_equal

_all_is = func_treelize(return_type=TreeTensor)(lambda x, y: x is y)


@pytest.mark.unittest
class TestTensorTreetensor:
    _DEMO_1 = TreeTensor({
        'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        'b': torch.tensor([[1, 2], [5, 6]]),
        'x': {
            'c': torch.tensor([3, 5, 6, 7]),
            'd': torch.tensor([[[1, 2], [8, 9]]]),
        }
    })

    def test_numel(self):
        assert self._DEMO_1.numel() == 18

    def test_numpy(self):
        assert all_array_equal(self._DEMO_1.numpy(), TreeNumpy({
            'a': np.array([[1, 2, 3], [4, 5, 6]]),
            'b': np.array([[1, 2], [5, 6]]),
            'x': {
                'c': np.array([3, 5, 6, 7]),
                'd': np.array([[[1, 2], [8, 9]]]),
            }
        }))

    def test_cpu(self):
        assert all_equal(self._DEMO_1.cpu(), self._DEMO_1)
        assert _all_is(self._DEMO_1.cpu(), self._DEMO_1).reduce(lambda **kws: all(kws.values()))

    def test_to(self):
        assert all_equal(self._DEMO_1.to(torch.float32), TreeTensor({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
            'b': torch.tensor([[1, 2], [5, 6]], dtype=torch.float32),
            'x': {
                'c': torch.tensor([3, 5, 6, 7], dtype=torch.float32),
                'd': torch.tensor([[[1, 2], [8, 9]]], dtype=torch.float32),
            }
        }))
