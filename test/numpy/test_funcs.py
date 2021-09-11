import numpy as np
import pytest

import treetensor.numpy as tnp


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestNumpyFuncs:
    _DEMO_1 = tnp.TreeNumpy({
        'a': np.array([[1, 2, 3], [5, 6, 7]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    _DEMO_2 = tnp.TreeNumpy({
        'a': np.array([[1, 2, 3], [5, 6, 8]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    _DEMO_3 = tnp.TreeNumpy({
        'a': np.array([[1, 2, 3], [5, 6, 7]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    def test_all(self):
        assert tnp.all(np.array([True, True, True]))
        assert not tnp.all(np.array([True, True, False]))
        assert not tnp.all(np.array([False, False, False]))

        assert tnp.all(tnp.TreeNumpy({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, True]),
        }))
        assert not tnp.all(tnp.TreeNumpy({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, False]),
        }))
        assert not tnp.all(tnp.TreeNumpy({
            'a': np.array([False, False, False]),
            'b': np.array([False, False, False]),
        }))

    def test_any(self):
        assert tnp.any(np.array([True, True, True]))
        assert tnp.any(np.array([True, True, False]))
        assert not tnp.any(np.array([False, False, False]))

        assert tnp.any(tnp.TreeNumpy({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, True]),
        }))
        assert tnp.any(tnp.TreeNumpy({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, False]),
        }))
        assert not tnp.any(tnp.TreeNumpy({
            'a': np.array([False, False, False]),
            'b': np.array([False, False, False]),
        }))

    def test_equal(self):
        assert tnp.all(tnp.equal(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ))
        assert not tnp.all(tnp.equal(
            np.array([1, 2, 3]),
            np.array([1, 2, 4]),
        ))

        assert tnp.all(
            tnp.equal(self._DEMO_1, self._DEMO_2) == tnp.TreeNumpy({
                'a': np.array([[True, True, True], [True, True, False]]),
                'b': np.array([True, True, True, True]),
                'x': {
                    'c': np.array([True, True, True]),
                    'd': np.array([[True, True]]),
                }
            })
        )
        assert tnp.all(
            tnp.equal(self._DEMO_1, self._DEMO_3) == tnp.TreeNumpy({
                'a': np.array([[True, True, True], [True, True, True]]),
                'b': np.array([True, True, True, True]),
                'x': {
                    'c': np.array([True, True, True]),
                    'd': np.array([[True, True]]),
                }
            })
        )

    def test_array_equal(self):
        assert tnp.all(tnp.array_equal(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ))
        assert not tnp.all(tnp.array_equal(
            np.array([1, 2, 3]),
            np.array([1, 2, 4]),
        ))

        assert tnp.array_equal(self._DEMO_1, self._DEMO_2) == tnp.TreeNumpy({
            'a': False,
            'b': True,
            'x': {
                'c': True,
                'd': True,
            }
        })
        assert tnp.array_equal(self._DEMO_1, self._DEMO_3) == tnp.TreeNumpy({
            'a': True,
            'b': True,
            'x': {
                'c': True,
                'd': True,
            }
        })
