import pytest
import torch

import treetensor.torch as ttorch
from treetensor.utils import replaceable_partial
from ..tests import choose_mark_with_existence_check

choose_mark = replaceable_partial(choose_mark_with_existence_check, base=ttorch.Size)


@pytest.mark.unittest
class TestTorchModule:
    def test_float32(self):
        assert ttorch.float32 is torch.float32

    def test_has_cuda(self):
        assert ttorch.has_cuda == torch.has_cuda

    def test_fxxk(self):
        with pytest.raises(AttributeError):
            _ = ttorch.fxxk

    def test___all__(self):
        assert 'has_cuda' not in ttorch.__all__
        assert 'float32' not in ttorch.__all__
        assert 'fxxk' not in ttorch.__all__

    def test_dir(self):
        assert 'has_cuda' not in dir(ttorch)
        assert 'float32' not in dir(ttorch)
        assert 'fxxk' not in dir(ttorch)
