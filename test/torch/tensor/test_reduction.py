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
    def test_mean(self):
        t0 = torch.tensor([[26.6598, 27.8008, -59.4753],
                           [-79.1833, 3.3349, 20.1665]])
        t1 = t0.mean()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, torch.tensor(-10.1161), atol=1e-4).all()
        t2 = t0.mean(dim=1)
        assert isinstance(t2, torch.Tensor)
        assert ttorch.isclose(t2, torch.tensor([-1.6716, -18.5606]), atol=1e-4).all()

        tt0 = ttorch.tensor({
            'a': [[25.2702, 37.4206, -37.1401],
                  [-7.7245, -91.3234, -27.9402]],
            'b': {'x': [[3.2028, -14.0720, 18.1739, 8.5944],
                        [41.7761, 36.9908, -20.5495, 5.6480],
                        [-9.3438, -0.7416, 47.2113, 6.9325]]},
        })
        tt1 = tt0.mean()
        assert isinstance(tt1, torch.Tensor)
        assert ttorch.isclose(tt1, torch.tensor(1.2436), atol=1e-4).all()
        tt2 = tt0.mean(reduce=False)
        assert ttorch.isclose(tt2, ttorch.tensor({
            'a': -16.9062,
            'b': {'x': 10.3186},
        }), atol=1e-4).all()
        tt3 = tt0.mean(dim=1)
        assert ttorch.isclose(tt3, ttorch.tensor({
            'a': [8.5169, -42.3294],
            'b': {'x': [3.9748, 15.9663, 11.0146]}
        }), atol=1e-4).all()

    @choose_mark()
    def test_std(self):
        t0 = torch.tensor([[25.5133, 24.2050, 8.1067],
                           [22.7316, -17.8863, -37.9171]])
        t1 = t0.std()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, torch.tensor(26.3619), atol=1e-4).all()
        t2 = t0.std(dim=1)
        assert isinstance(t2, torch.Tensor)
        assert ttorch.isclose(t2, torch.tensor([9.6941, 30.9012]), atol=1e-4).all()

        tt0 = ttorch.tensor({
            'a': [[-48.6580, 30.9506, -16.1800],
                  [37.6667, 10.3850, -5.7679]],
            'b': {'x': [[-17.9371, 8.4873, -49.0445, 4.7368],
                        [21.3990, -11.2385, -15.9331, -41.6838],
                        [-7.1814, -38.1301, -2.2320, 10.1392]]},
        })
        tt1 = tt0.std()
        assert isinstance(tt1, torch.Tensor)
        assert ttorch.isclose(tt1, torch.tensor(25.6854), atol=1e-4).all()
        tt2 = tt0.std(reduce=False)
        assert ttorch.isclose(tt2, ttorch.tensor({
            'a': 32.0483,
            'b': {'x': 22.1754},
        }), atol=1e-4).all()
        tt3 = tt0.std(dim=1)
        assert ttorch.isclose(tt3, ttorch.tensor({
            'a': [40.0284, 21.9536],
            'b': {'x': [26.4519, 25.9011, 20.5223]},
        }), atol=1e-4).all()

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
