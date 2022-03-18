import numpy as np
import pytest

import treetensor.numpy as tnp


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestNumpyFuncs:
    _DEMO_1 = tnp.ndarray({
        'a': np.array([[1, 2, 3], [5, 6, 7]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    _DEMO_2 = tnp.ndarray({
        'a': np.array([[1, 2, 3], [5, 6, 8]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([3, 5, 7]),
            'd': np.array([[7, 9]]),
        }
    })

    _DEMO_3 = tnp.ndarray({
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

        assert tnp.all(tnp.ndarray({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, True]),
        }))
        assert not tnp.all(tnp.ndarray({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, False]),
        }))
        assert not tnp.all(tnp.ndarray({
            'a': np.array([False, False, False]),
            'b': np.array([False, False, False]),
        }))

    def test_any(self):
        assert tnp.any(np.array([True, True, True]))
        assert tnp.any(np.array([True, True, False]))
        assert not tnp.any(np.array([False, False, False]))

        assert tnp.any(tnp.ndarray({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, True]),
        }))
        assert tnp.any(tnp.ndarray({
            'a': np.array([True, True, True]),
            'b': np.array([True, True, False]),
        }))
        assert not tnp.any(tnp.ndarray({
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
            tnp.equal(self._DEMO_1, self._DEMO_2) == tnp.ndarray({
                'a': np.array([[True, True, True], [True, True, False]]),
                'b': np.array([True, True, True, True]),
                'x': {
                    'c': np.array([True, True, True]),
                    'd': np.array([[True, True]]),
                }
            })
        )
        assert tnp.all(
            tnp.equal(self._DEMO_1, self._DEMO_3) == tnp.ndarray({
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

        assert tnp.array_equal(self._DEMO_1, self._DEMO_2) == tnp.ndarray({
            'a': False,
            'b': True,
            'x': {
                'c': True,
                'd': True,
            }
        })
        assert tnp.array_equal(self._DEMO_1, self._DEMO_3) == tnp.ndarray({
            'a': True,
            'b': True,
            'x': {
                'c': True,
                'd': True,
            }
        })

    def test_zeros(self):
        zs = tnp.zeros((2, 3))
        assert isinstance(zs, np.ndarray)
        assert np.allclose(zs, np.zeros((2, 3)))

        zs = tnp.zeros({'a': (2, 3), 'c': {'x': (3, 4)}})
        assert tnp.allclose(zs, tnp.ndarray({
            'a': np.zeros((2, 3)),
            'c': {'x': np.zeros((3, 4))}
        }))

    def test_ones(self):
        zs = tnp.ones((2, 3))
        assert isinstance(zs, np.ndarray)
        assert np.allclose(zs, np.ones((2, 3)))

        zs = tnp.ones({'a': (2, 3), 'c': {'x': (3, 4)}})
        assert tnp.allclose(zs, tnp.ndarray({
            'a': np.ones((2, 3)),
            'c': {'x': np.zeros((3, 4))}
        }))

    def test_stack(self):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        nd = tnp.stack((a, b))
        assert isinstance(nd, np.ndarray)
        assert np.allclose(nd, np.array([[1, 2, 3],
                                         [2, 3, 4]]))

        a = tnp.array({
            'a': [1, 2, 3],
            'c': {'x': [11, 22, 33]},
        })
        b = tnp.array({
            'a': [2, 3, 4],
            'c': {'x': [22, 33, 44]},
        })
        nd = tnp.stack((a, b))
        assert tnp.allclose(nd, tnp.array({
            'a': [[1, 2, 3], [2, 3, 4]],
            'c': {'x': [[11, 22, 33], [22, 33, 44]]},
        }))

    def test_concatenate(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6]])
        nd = tnp.concatenate((a, b), axis=0)
        assert isinstance(nd, np.ndarray)
        assert np.allclose(nd, np.array([[1, 2],
                                         [3, 4],
                                         [5, 6]]))

        a = tnp.array({
            'a': [[1, 2], [3, 4]],
            'c': {'x': [[11, 22], [33, 44]]},
        })
        b = tnp.array({
            'a': [[5, 6]],
            'c': {'x': [[55, 66]]},
        })
        nd = tnp.concatenate((a, b), axis=0)
        assert tnp.allclose(nd, tnp.array({
            'a': [[1, 2], [3, 4], [5, 6]],
            'c': {'x': [[11, 22], [33, 44], [55, 66]]},
        }))

    def test_split(self):
        x = np.arange(9.0)
        ns = tnp.split(x, 3)
        assert len(ns) == 3
        assert isinstance(ns[0], np.ndarray)
        assert np.allclose(ns[0], np.array([0.0, 1.0, 2.0]))
        assert isinstance(ns[1], np.ndarray)
        assert np.allclose(ns[1], np.array([3.0, 4.0, 5.0]))
        assert isinstance(ns[2], np.ndarray)
        assert np.allclose(ns[2], np.array([6.0, 7.0, 8.0]))

        xx = tnp.arange(tnp.ndarray({'a': 9.0, 'c': {'x': 18.0}}))
        ns = tnp.split(xx, 3)
        assert len(ns) == 3
        assert tnp.allclose(ns[0], tnp.array({
            'a': [0.0, 1.0, 2.0],
            'c': {'x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
        }))
        assert tnp.allclose(ns[1], tnp.array({
            'a': [3.0, 4.0, 5.0],
            'c': {'x': [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]},
        }))
        assert tnp.allclose(ns[2], tnp.array({
            'a': [6.0, 7.0, 8.0],
            'c': {'x': [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]},
        }))
