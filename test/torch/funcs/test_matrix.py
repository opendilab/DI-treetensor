import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchFuncsMatrix:
    @choose_mark()
    def test_dot(self):
        t1 = ttorch.dot(torch.tensor([1, 2]), torch.tensor([2, 3]))
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == 8

        t2 = ttorch.dot(
            ttorch.tensor({
                'a': [1, 2, 3],
                'b': {'x': [3, 4]},
            }),
            ttorch.tensor({
                'a': [5, 6, 7],
                'b': {'x': [1, 2]},
            })
        )
        assert (t2 == ttorch.tensor({'a': 38, 'b': {'x': 11}})).all()

    @choose_mark()
    def test_matmul(self):
        t1 = ttorch.matmul(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 2]]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([[19, 10], [43, 26]])).all()

        t2 = ttorch.matmul(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': {'x': [3, 4, 5, 6]},
            }),
            ttorch.tensor({
                'a': [[5, 6], [7, 2]],
                'b': {'x': [4, 3, 2, 1]},
            }),
        )
        assert (t2 == ttorch.tensor({
            'a': [[19, 10], [43, 26]],
            'b': {'x': 40}
        })).all()

    @choose_mark()
    def test_mm(self):
        t1 = ttorch.mm(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 2]]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([[19, 10], [43, 26]])).all()

        t2 = ttorch.mm(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': {'x': [[3, 4, 5], [6, 7, 8]]},
            }),
            ttorch.tensor({
                'a': [[5, 6], [7, 2]],
                'b': {'x': [[6, 5], [4, 3], [2, 1]]},
            }),
        )
        assert (t2 == ttorch.tensor({
            'a': [[19, 10], [43, 26]],
            'b': {'x': [[44, 32], [80, 59]]},
        })).all()
