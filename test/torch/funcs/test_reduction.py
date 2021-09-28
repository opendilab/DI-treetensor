import pytest
import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchFuncsReduction:
    @choose_mark()
    def test_all(self):
        r1 = ttorch.all(torch.tensor([True, True, True]))
        assert torch.is_tensor(r1)
        assert r1 == torch.tensor(True)
        assert r1

        r2 = ttorch.all(torch.tensor([True, True, False]))
        assert torch.is_tensor(r2)
        assert r2 == torch.tensor(False)
        assert not r2

        r3 = ttorch.all(torch.tensor([False, False, False]))
        assert torch.is_tensor(r3)
        assert r3 == torch.tensor(False)
        assert not r3

        r4 = ttorch.all({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, True]),
        })
        assert torch.is_tensor(r4)
        assert r4 == torch.tensor(True)
        assert r4

        r5 = ttorch.all({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, False]),
        })
        assert torch.is_tensor(r5)
        assert r5 == torch.tensor(False)
        assert not r5

        r6 = ttorch.all({
            'a': torch.tensor([False, False, False]),
            'b': torch.tensor([False, False, False]),
        })
        assert torch.is_tensor(r6)
        assert r6 == torch.tensor(False)
        assert not r6

        r7 = ttorch.all(ttorch.tensor({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, False]),
        }), reduce=False)
        assert (r7 == ttorch.tensor({
            'a': True, 'b': False
        })).all()

        r8 = ttorch.all(ttorch.tensor({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, False]),
        }), dim=0)
        assert (r8 == ttorch.tensor({
            'a': True, 'b': False
        })).all()

        with pytest.warns(UserWarning):
            r9 = ttorch.all(ttorch.tensor({
                'a': torch.tensor([True, True, True]),
                'b': torch.tensor([True, True, False]),
            }), dim=0, reduce=True)
        assert (r9 == ttorch.tensor({
            'a': True, 'b': False
        })).all()

    @choose_mark()
    def test_any(self):
        r1 = ttorch.any(torch.tensor([True, True, True]))
        assert torch.is_tensor(r1)
        assert r1 == torch.tensor(True)
        assert r1

        r2 = ttorch.any(torch.tensor([True, True, False]))
        assert torch.is_tensor(r2)
        assert r2 == torch.tensor(True)
        assert r2

        r3 = ttorch.any(torch.tensor([False, False, False]))
        assert torch.is_tensor(r3)
        assert r3 == torch.tensor(False)
        assert not r3

        r4 = ttorch.any({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, True]),
        })
        assert torch.is_tensor(r4)
        assert r4 == torch.tensor(True)
        assert r4

        r5 = ttorch.any({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, False]),
        })
        assert torch.is_tensor(r5)
        assert r5 == torch.tensor(True)
        assert r5

        r6 = ttorch.any({
            'a': torch.tensor([False, False, False]),
            'b': torch.tensor([False, False, False]),
        })
        assert torch.is_tensor(r6)
        assert r6 == torch.tensor(False)
        assert not r6

        r7 = ttorch.any(ttorch.tensor({
            'a': torch.tensor([True, True, False]),
            'b': torch.tensor([False, False, False]),
        }), reduce=False)
        assert (r7 == ttorch.tensor({
            'a': True, 'b': False
        })).all()

        r8 = ttorch.any(ttorch.tensor({
            'a': torch.tensor([True, True, False]),
            'b': torch.tensor([False, False, False]),
        }), dim=0)
        assert (r8 == ttorch.tensor({
            'a': True, 'b': False
        })).all()

        with pytest.warns(UserWarning):
            r9 = ttorch.any(ttorch.tensor({
                'a': torch.tensor([True, True, False]),
                'b': torch.tensor([False, False, False]),
            }), dim=0, reduce=True)
        assert (r9 == ttorch.tensor({
            'a': True, 'b': False
        })).all()

    @choose_mark()
    def test_min(self):
        t1 = ttorch.min(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert t1 == torch.tensor(1.0)

        tt0 = ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })
        assert ttorch.isclose(ttorch.min(tt0), ttorch.tensor(0.9), atol=1e-4).all()

        tt1 = ttorch.min(tt0, reduce=False)
        assert ttorch.isclose(tt1, ttorch.tensor({
            'a': 1.0, 'b': 0.9,
        }), atol=1e-4).all()

        tt2_a, tt2_b = ttorch.min(tt0, dim=0)
        assert ttorch.isclose(tt2_a, ttorch.tensor({
            'a': 1.0, 'b': [1.3, 0.9],
        }), atol=1e-4).all()
        assert (tt2_b == ttorch.tensor({
            'a': 0, 'b': [1, 0],
        })).all()

    @choose_mark()
    def test_max(self):
        t1 = ttorch.max(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert t1 == torch.tensor(2.0)

        tt0 = ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })
        assert ttorch.isclose(ttorch.max(tt0), ttorch.tensor(2.5), atol=1e-4)

        tt1 = ttorch.max(tt0, reduce=False)
        assert ttorch.isclose(tt1, ttorch.tensor({
            'a': 2.0, 'b': 2.5,
        }), atol=1e-4).all()

        tt2_a, tt2_b = ttorch.max(tt0, dim=0)
        assert ttorch.isclose(tt2_a, ttorch.tensor({
            'a': 2.0, 'b': [1.8, 2.5],
        }), atol=1e-4).all()
        assert (tt2_b == ttorch.tensor({
            'a': 1, 'b': [0, 1],
        })).all()

    @choose_mark()
    def test_sum(self):
        assert ttorch.sum(torch.tensor([1.0, 2.0, 1.5])) == torch.tensor(4.5)

        tt0 = ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })
        assert ttorch.isclose(ttorch.sum(tt0), torch.tensor(11.0), atol=1e-4).all()
        assert ttorch.isclose(ttorch.sum(tt0, reduce=False), ttorch.tensor({
            'a': 4.5, 'b': {'x': 6.5},
        }), atol=1e-4).all()
        assert ttorch.isclose(ttorch.sum(tt0, dim=0), ttorch.tensor({
            'a': 4.5, 'b': {'x': [3.1, 3.4]},
        }), atol=1e-4).all()

    @choose_mark()
    def test_mean(self):
        t0 = torch.tensor([[26.6598, 27.8008, -59.4753],
                           [-79.1833, 3.3349, 20.1665]])
        t1 = ttorch.mean(t0)
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, torch.tensor(-10.1161), atol=1e-4).all()
        t2 = ttorch.mean(t0, dim=1)
        assert isinstance(t2, torch.Tensor)
        assert ttorch.isclose(t2, torch.tensor([-1.6716, -18.5606]), atol=1e-4).all()

        tt0 = ttorch.tensor({
            'a': [[25.2702, 37.4206, -37.1401],
                  [-7.7245, -91.3234, -27.9402]],
            'b': {'x': [[3.2028, -14.0720, 18.1739, 8.5944],
                        [41.7761, 36.9908, -20.5495, 5.6480],
                        [-9.3438, -0.7416, 47.2113, 6.9325]]},
        })
        tt1 = ttorch.mean(tt0)
        assert isinstance(tt1, torch.Tensor)
        assert ttorch.isclose(tt1, torch.tensor(1.2436), atol=1e-4).all()
        tt2 = ttorch.mean(tt0, reduce=False)
        assert ttorch.isclose(tt2, ttorch.tensor({
            'a': -16.9062,
            'b': {'x': 10.3186},
        }), atol=1e-4).all()
        tt3 = ttorch.mean(tt0, dim=1)
        assert ttorch.isclose(tt3, ttorch.tensor({
            'a': [8.5169, -42.3294],
            'b': {'x': [3.9748, 15.9663, 11.0146]}
        }), atol=1e-4).all()

        with pytest.warns(UserWarning):
            tt4 = ttorch.mean(tt0, dim=1, reduce=True)
        assert ttorch.isclose(tt4, ttorch.tensor({
            'a': [8.5169, -42.3294],
            'b': {'x': [3.9748, 15.9663, 11.0146]}
        }), atol=1e-4).all()

    @choose_mark()
    def test_std(self):
        t0 = torch.tensor([[25.5133, 24.2050, 8.1067],
                           [22.7316, -17.8863, -37.9171]])
        t1 = ttorch.std(t0)
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, torch.tensor(26.3619), atol=1e-4).all()
        t2 = ttorch.std(t0, dim=1)
        assert isinstance(t2, torch.Tensor)
        assert ttorch.isclose(t2, torch.tensor([9.6941, 30.9012]), atol=1e-4).all()

        tt0 = ttorch.tensor({
            'a': [[-48.6580, 30.9506, -16.1800],
                  [37.6667, 10.3850, -5.7679]],
            'b': {'x': [[-17.9371, 8.4873, -49.0445, 4.7368],
                        [21.3990, -11.2385, -15.9331, -41.6838],
                        [-7.1814, -38.1301, -2.2320, 10.1392]]},
        })
        tt1 = ttorch.std(tt0)
        assert isinstance(tt1, torch.Tensor)
        assert ttorch.isclose(tt1, torch.tensor(25.6854), atol=1e-4).all()
        tt2 = ttorch.std(tt0, reduce=False)
        assert ttorch.isclose(tt2, ttorch.tensor({
            'a': 32.0483,
            'b': {'x': 22.1754},
        }), atol=1e-4).all()
        tt3 = ttorch.std(tt0, dim=1)
        assert ttorch.isclose(tt3, ttorch.tensor({
            'a': [40.0284, 21.9536],
            'b': {'x': [26.4519, 25.9011, 20.5223]},
        }), atol=1e-4).all()

        with pytest.warns(UserWarning):
            tt4 = ttorch.std(tt0, dim=1, reduce=True)
        assert ttorch.isclose(tt4, ttorch.tensor({
            'a': [40.0284, 21.9536],
            'b': {'x': [26.4519, 25.9011, 20.5223]},
        }), atol=1e-4).all()

    @choose_mark()
    def test_masked_select(self):
        tx = torch.tensor([[0.0481, 0.1741, 0.9820, -0.6354],
                           [0.8108, -0.7126, 0.1329, 1.0868],
                           [-1.8267, 1.3676, -1.4490, -2.0224]])
        t1 = ttorch.masked_select(tx, tx > 0.3)
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
        tt1 = ttorch.masked_select(ttx, ttx > 0.3)
        assert ttorch.isclose(tt1, torch.tensor([1.1799, 0.4652, 1.0866, 1.3533, 0.8139,
                                                 0.9073, 2.1392, 0.6403, 0.4041]), atol=1e-4).all()
        tt2 = ttorch.masked_select(ttx, ttx > 0.3, reduce=False)
        assert ttorch.isclose(tt2, ttorch.tensor({
            'a': [1.1799, 0.4652, 1.0866, 1.3533],
            'b': {'x': [0.8139, 0.9073, 2.1392, 0.6403, 0.4041]},
        }), atol=1e-4).all()
