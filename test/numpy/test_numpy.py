import pytest

from treetensor import TreeNumpy

try:
    import numpy as np
except ImportError:
    need_real = False
else:
    need_real = True
    from treetensor.numpy.numpy import TreeNumpy as RealTreeNumpy

unittest_mark = pytest.mark.unittest if need_real else pytest.mark.ignore


@unittest_mark
class TestNumpyReal:
    def test_base(self):
        assert TreeNumpy is RealTreeNumpy

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
