import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchTensorReduction:
    @choose_mark()
    def test_all(self):
        t1 = ttorch.Tensor({
            'a': [True, True],
            'b': {'x': [[True, True, ], [True, True, ]]}
        }).all()
        assert isinstance(t1, torch.Tensor)
        assert t1.dtype == torch.bool
        assert t1

        t2 = ttorch.Tensor({
            'a': [True, False],
            'b': {'x': [[True, True, ], [True, True, ]]}
        }).all()
        assert isinstance(t2, torch.Tensor)
        assert t2.dtype == torch.bool
        assert not t2

    @choose_mark()
    def test_any(self):
        t1 = ttorch.Tensor({
            'a': [True, False],
            'b': {'x': [[False, False, ], [False, False, ]]}
        }).any()
        assert isinstance(t1, torch.Tensor)
        assert t1.dtype == torch.bool
        assert t1

        t2 = ttorch.Tensor({
            'a': [False, False],
            'b': {'x': [[False, False, ], [False, False, ]]}
        }).any()
        assert isinstance(t2, torch.Tensor)
        assert t2.dtype == torch.bool
        assert not t2

    @choose_mark()
    def test_max(self):
        t1 = ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }).max()
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == 3

    @choose_mark()
    def test_min(self):
        t1 = ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }).min()
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == -1

    @choose_mark()
    def test_sum(self):
        t1 = ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }).sum()
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == 7
