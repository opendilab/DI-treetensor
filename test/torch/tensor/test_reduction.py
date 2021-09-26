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

    @choose_mark()
    def test_masked_select(self):
        tx = torch.tensor([[0.0481, 0.1741, 0.9820, -0.6354],
                           [0.8108, -0.7126, 0.1329, 1.0868],
                           [-1.8267, 1.3676, -1.4490, -2.0224]])
        t1 = tx.masked_select(tx > 0.3)
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([0.9820, 0.8108, 1.0868, 1.3676])).all()

        ttx = ttorch.tensor({
            'a': [[1.1799, 0.4652, -1.7895],
                  [0.0423, 1.0866, 1.3533]],
            'b': {
                'x': [[0.8139, -0.6732, 0.0065, 0.9073],
                      [0.0596, -2.0621, -0.1598, -1.0793],
                      [-0.0496, 2.1392, 0.6403, 0.4041]],
            }
        })
        tt1 = ttx.masked_select(ttx > 0.3)
        assert (tt1 == torch.tensor([1.1799, 0.4652, 1.0866, 1.3533, 0.8139,
                                     0.9073, 2.1392, 0.6403, 0.4041])).all()
