import math

import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchTensorComparison:
    @choose_mark()
    def test_isfinite(self):
        t1 = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite()
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([True, False, True, False, False])).all()

        t2 = ttorch.tensor({
            'a': [1, float('inf'), 2, float('-inf'), float('nan')],
            'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        }).isfinite()
        assert (t2 == ttorch.tensor({
            'a': [True, False, True, False, False],
            'b': {'x': [[True, False, True], [False, True, False]]},
        }))

    @choose_mark()
    def test_isinf(self):
        t1 = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf()
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([False, True, False, True, False])).all()

        t2 = ttorch.tensor({
            'a': [1, float('inf'), 2, float('-inf'), float('nan')],
            'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        }).isinf()
        assert (t2 == ttorch.tensor({
            'a': [False, True, False, True, False],
            'b': {'x': [[False, True, False], [True, False, False]]},
        }))

    @choose_mark()
    def test_isnan(self):
        t1 = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan()
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([False, False, False, False, True])).all()

        t2 = ttorch.tensor({
            'a': [1, float('inf'), 2, float('-inf'), float('nan')],
            'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        }).isnan()
        assert (t2 == ttorch.tensor({
            'a': [False, False, False, False, True],
            'b': {'x': [[False, False, False], [False, False, True]]},
        })).all()

    @choose_mark()
    def test_isclose(self):
        t1 = ttorch.tensor((1., 2, 3)).isclose(ttorch.tensor((1 + 1e-10, 3, 4)))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([True, False, False])).all()

        t2 = ttorch.tensor({
            'a': [1., 2, 3],
            'b': {'x': [[float('inf'), 4, 1e20],
                        [-math.inf, 2.2943, 9483.32]]},
        }).isclose(ttorch.tensor({
            'a': [1 + 1e-10, 3, 4],
            'b': {'x': [[math.inf, 6, 1e20 + 1],
                        [-float('inf'), 2.294300000001, 9484.32]]},
        }))
        assert (t2 == ttorch.tensor({
            'a': [True, False, False],
            'b': {'x': [[True, False, True],
                        [True, True, False]]},
        })).all()
