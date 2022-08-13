import unittest

import numpy as np
import pytest
import torch
from hbutils.testing import OS

import treetensor.numpy as tnp
import treetensor.torch as ttorch
from treetensor.common import Object


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestNumpyArray:
    _DEMO_1 = tnp.ndarray({
        'a': np.array([[1, 2, 3], [4, 5, 6]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([[11], [23]]),
            'd': np.array([3, 9, 11.0])
        }
    })

    _DEMO_2 = tnp.ndarray({
        'a': np.array([[1, 22, 3], [4, 5, 6]]),
        'b': np.array([1, 3, 5, 7]),
        'x': {
            'c': np.array([[11], [0]]),
            'd': np.array([3, 9, 11.0])
        }
    })

    _DEMO_3 = tnp.ndarray({
        'a': np.array([[0, 0, 0], [0, 0, 0]]),
        'b': np.array([0, 0, 0, 0]),
        'x': {
            'c': np.array([[0], [0]]),
            'd': np.array([0, 0, 0.0])
        }
    })

    def test_array(self):
        assert (self._DEMO_1 == tnp.array({
            'a': [[1, 2, 3], [4, 5, 6]],
            'b': [1, 3, 5, 7],
            'x': {
                'c': [[11], [23]],
                'd': [3, 9, 11.0]
            }
        })).all()

    def test_size(self):
        assert self._DEMO_1.size == 15
        assert self._DEMO_2.size == 15
        assert self._DEMO_3.size == 15

    @unittest.skipUnless(OS.windows, 'Windows only')
    def test_nbytes_on_windows(self):
        assert self._DEMO_1.nbytes == 72
        assert self._DEMO_2.nbytes == 72
        assert self._DEMO_3.nbytes == 72

    @unittest.skipUnless(OS.linux or OS.macos, 'Linux or macos only')
    def test_nbytes_on_linux_or_macos(self):
        assert self._DEMO_1.nbytes == 120
        assert self._DEMO_2.nbytes == 120
        assert self._DEMO_3.nbytes == 120

    def test_sum(self):
        assert self._DEMO_1.sum() == 94.0
        assert self._DEMO_2.sum() == 91.0
        assert self._DEMO_3.sum() == 0.0

    def test_all(self):
        assert self._DEMO_1.all()
        assert not self._DEMO_2.all()
        assert not self._DEMO_3.all()
        assert tnp.ndarray({
            'a': np.array([[True, True, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [True]]),
                'd': np.array([True, True, True])
            }
        }).all()
        assert not tnp.ndarray({
            'a': np.array([[True, True, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [True]]),
                'd': np.array([True, True, False])
            }
        }).all()
        assert not tnp.ndarray({
            'a': np.array([[False, False, False], [False, False, False]]),
            'b': np.array([False, False, False, False]),
            'x': {
                'c': np.array([[False], [False]]),
                'd': np.array([False, False, False])
            }
        }).all()

    def test_any(self):
        assert self._DEMO_1.any()
        assert self._DEMO_2.any()
        assert not self._DEMO_3.any()
        assert tnp.ndarray({
            'a': np.array([[True, True, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [True]]),
                'd': np.array([True, True, True])
            }
        }).any()
        assert tnp.ndarray({
            'a': np.array([[True, True, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [True]]),
                'd': np.array([True, True, False])
            }
        }).any()
        assert not tnp.ndarray({
            'a': np.array([[False, False, False], [False, False, False]]),
            'b': np.array([False, False, False, False]),
            'x': {
                'c': np.array([[False], [False]]),
                'd': np.array([False, False, False])
            }
        }).any()

    def test_eq(self):
        assert (self._DEMO_1 == self._DEMO_1).all()
        assert (self._DEMO_2 == self._DEMO_2).all()
        assert not (self._DEMO_1 == self._DEMO_2).all()

    def test_ne(self):
        assert not (self._DEMO_1 != self._DEMO_1).any()
        assert not (self._DEMO_2 != self._DEMO_2).any()
        assert (self._DEMO_1 == self._DEMO_2).any()

    def test_gt(self):
        assert not (self._DEMO_1 > self._DEMO_1).any()
        assert not (self._DEMO_2 > self._DEMO_2).any()
        assert ((self._DEMO_1 > self._DEMO_2) == tnp.ndarray({
            'a': np.array([[False, False, False], [False, False, False]]),
            'b': np.array([False, False, False, False]),
            'x': {
                'c': np.array([[False], [True]]),
                'd': np.array([False, False, False])
            }
        })).all()
        assert ((self._DEMO_2 > self._DEMO_1) == tnp.ndarray({
            'a': np.array([[False, True, False], [False, False, False]]),
            'b': np.array([False, False, False, False]),
            'x': {
                'c': np.array([[False], [False]]),
                'd': np.array([False, False, False])
            }
        })).all()

    def test_ge(self):
        assert (self._DEMO_1 >= self._DEMO_1).all()
        assert (self._DEMO_2 >= self._DEMO_2).all()
        assert ((self._DEMO_1 >= self._DEMO_2) == tnp.ndarray({
            'a': np.array([[True, False, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [True]]),
                'd': np.array([True, True, True])
            }
        })).all()
        assert ((self._DEMO_2 >= self._DEMO_1) == tnp.ndarray({
            'a': np.array([[True, True, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [False]]),
                'd': np.array([True, True, True])
            }
        })).all()

    def test_lt(self):
        assert not (self._DEMO_1 < self._DEMO_1).any()
        assert not (self._DEMO_2 < self._DEMO_2).any()
        assert ((self._DEMO_1 < self._DEMO_2) == tnp.ndarray({
            'a': np.array([[False, True, False], [False, False, False]]),
            'b': np.array([False, False, False, False]),
            'x': {
                'c': np.array([[False], [False]]),
                'd': np.array([False, False, False])
            }
        })).all()
        assert ((self._DEMO_2 < self._DEMO_1) == tnp.ndarray({
            'a': np.array([[False, False, False], [False, False, False]]),
            'b': np.array([False, False, False, False]),
            'x': {
                'c': np.array([[False], [True]]),
                'd': np.array([False, False, False])
            }
        })).all()

    def test_le(self):
        assert (self._DEMO_1 <= self._DEMO_1).all()
        assert (self._DEMO_2 <= self._DEMO_2).all()
        assert ((self._DEMO_1 <= self._DEMO_2) == tnp.ndarray({
            'a': np.array([[True, True, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [False]]),
                'd': np.array([True, True, True])
            }
        })).all()
        assert ((self._DEMO_2 <= self._DEMO_1) == tnp.ndarray({
            'a': np.array([[True, False, True], [True, True, True]]),
            'b': np.array([True, True, True, True]),
            'x': {
                'c': np.array([[True], [True]]),
                'd': np.array([True, True, True])
            }
        })).all()

    def test_tolist(self):
        assert self._DEMO_1.tolist() == Object({
            'a': [[1, 2, 3], [4, 5, 6]],
            'b': [1, 3, 5, 7],
            'x': {
                'c': [[11], [23]],
                'd': [3, 9, 11.0],
            }
        })
        assert self._DEMO_2.tolist() == Object({
            'a': [[1, 22, 3], [4, 5, 6]],
            'b': [1, 3, 5, 7],
            'x': {
                'c': [[11], [0]],
                'd': [3, 9, 11.0],
            }
        })
        assert self._DEMO_3.tolist() == Object({
            'a': [[0, 0, 0], [0, 0, 0]],
            'b': [0, 0, 0, 0],
            'x': {
                'c': [[0], [0]],
                'd': [0, 0, 0.0],
            }
        })

    def test_tensor(self):
        assert ttorch.isclose(self._DEMO_1.tensor().double(), ttorch.Tensor({
            'a': ttorch.Tensor([[1, 2, 3], [4, 5, 6]]),
            'b': ttorch.Tensor([1, 3, 5, 7]),
            'x': {
                'c': ttorch.Tensor([[11], [23]]),
                'd': ttorch.Tensor([3, 9, 11.0])
            }
        }).double()).all()

        assert (self._DEMO_1.tensor(dtype=torch.float64) == ttorch.Tensor({
            'a': ttorch.Tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64),
            'b': ttorch.Tensor([1, 3, 5, 7], dtype=torch.float64),
            'x': {
                'c': ttorch.Tensor([[11], [23]], dtype=torch.float64),
                'd': ttorch.Tensor([3, 9, 11.0], dtype=torch.float64),
            }
        })).all()
