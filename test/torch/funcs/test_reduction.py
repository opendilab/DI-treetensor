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
        }).all()
        assert torch.is_tensor(r4)
        assert r4 == torch.tensor(True)
        assert r4

        r5 = ttorch.all({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, False]),
        }).all()
        assert torch.is_tensor(r5)
        assert r5 == torch.tensor(False)
        assert not r5

        r6 = ttorch.all({
            'a': torch.tensor([False, False, False]),
            'b': torch.tensor([False, False, False]),
        }).all()
        assert torch.is_tensor(r6)
        assert r6 == torch.tensor(False)
        assert not r6

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
        }).all()
        assert torch.is_tensor(r4)
        assert r4 == torch.tensor(True)
        assert r4

        r5 = ttorch.any({
            'a': torch.tensor([True, True, True]),
            'b': torch.tensor([True, True, False]),
        }).all()
        assert torch.is_tensor(r5)
        assert r5 == torch.tensor(True)
        assert r5

        r6 = ttorch.any({
            'a': torch.tensor([False, False, False]),
            'b': torch.tensor([False, False, False]),
        }).all()
        assert torch.is_tensor(r6)
        assert r6 == torch.tensor(False)
        assert not r6

    @choose_mark()
    def test_min(self):
        t1 = ttorch.min(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert t1 == torch.tensor(1.0)

        assert ttorch.isclose(ttorch.min(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })), ttorch.tensor(0.9), atol=1e-4)

    @choose_mark()
    def test_max(self):
        t1 = ttorch.max(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert t1 == torch.tensor(2.0)

        assert ttorch.isclose(ttorch.max(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })), ttorch.tensor(2.5), atol=1e-4)

    @choose_mark()
    def test_sum(self):
        assert ttorch.sum(torch.tensor([1.0, 2.0, 1.5])) == torch.tensor(4.5)
        assert (ttorch.sum(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })) == torch.tensor(11.0)).all()

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
        assert (tt1 == torch.tensor([1.1799, 0.4652, 1.0866, 1.3533, 0.8139,
                                     0.9073, 2.1392, 0.6403, 0.4041])).all()
