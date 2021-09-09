import numpy as np
import pytest

from treetensor.numpy import TreeNumpy, equal, array_equal
from treetensor.numpy import all as _numpy_all


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestNumpyFuncs:
    _DEMO_1 = TreeNumpy({
        'a': np.array([[1, 2, 3], [5, 6, 7]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    _DEMO_2 = TreeNumpy({
        'a': np.array([[1, 2, 3], [5, 6, 8]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    _DEMO_3 = TreeNumpy({
        'a': np.array([[1, 2, 3], [5, 6, 7]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    def test__numpy_all(self):
        assert not _numpy_all(self._DEMO_1 == self._DEMO_2)
        assert _numpy_all(self._DEMO_1 == self._DEMO_3)
        assert not _numpy_all(np.array([1, 2, 3]) == np.array([1, 2, 4]))
        assert _numpy_all(np.array([1, 2, 3]) == np.array([1, 2, 3]))

    def test_equal(self):
        assert _numpy_all(
            equal(self._DEMO_1, self._DEMO_2) == TreeNumpy({
                'a': np.array([[True, True, True], [True, True, False]]),
                'b': np.array([True, True, True, True]),
                'x': {
                    'c': np.array([True, True, True]),
                    'd': np.array([[True, True]]),
                }
            })
        )
        assert _numpy_all(
            equal(self._DEMO_1, self._DEMO_3) == TreeNumpy({
                'a': np.array([[True, True, True], [True, True, True]]),
                'b': np.array([True, True, True, True]),
                'x': {
                    'c': np.array([True, True, True]),
                    'd': np.array([[True, True]]),
                }
            })
        )

    def test_array_equal(self):
        assert array_equal(self._DEMO_1, self._DEMO_2) == TreeNumpy({
            'a': False,
            'b': True,
            'x': {
                'c': True,
                'd': True,
            }
        })
        assert array_equal(self._DEMO_1, self._DEMO_3) == TreeNumpy({
            'a': True,
            'b': True,
            'x': {
                'c': True,
                'd': True,
            }
        })
