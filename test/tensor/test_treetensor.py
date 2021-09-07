import pytest
import torch

from treetensor import TreeTensor


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
