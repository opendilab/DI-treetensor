import numpy as np
import pytest

from treetensor.numpy import TreeNumpy


@pytest.mark.unittest
class TestNumpyNumpy:
    _DEMO_1 = TreeNumpy({
        'a': np.array([[1, 2, 3], [4, 5, 6]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([[11], [23]]),
            'd': np.array([3, 9, 11.0])
        }
    })

    def test_size(self):
        assert self._DEMO_1.size == 15

    def test_nbytes(self):
        assert self._DEMO_1.nbytes == 120

    def test_sum(self):
        assert self._DEMO_1.sum() == 94.0
