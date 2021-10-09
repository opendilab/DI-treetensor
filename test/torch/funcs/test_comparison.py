import math

import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchFuncsComparison:
    @choose_mark()
    def test_equal(self):
        p1 = ttorch.equal(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
        assert isinstance(p1, bool)
        assert p1

        p2 = ttorch.equal(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4]))
        assert isinstance(p2, bool)
        assert not p2

        p3 = ttorch.equal(ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }), ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }))
        assert isinstance(p3, bool)
        assert p3

        p4 = ttorch.equal(ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }), ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 5]),
        }))
        assert isinstance(p4, bool)
        assert not p4

    @choose_mark()
    def test_eq(self):
        assert ttorch.eq(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])).all()
        assert not ttorch.eq(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 2])).all()
        assert ttorch.eq(torch.tensor([1, 1, 1]), 1).all()
        assert not ttorch.eq(torch.tensor([1, 1, 2]), 1).all()

        assert ttorch.eq(ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }), ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        })).all()
        assert not ttorch.eq(ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }), ttorch.Tensor({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 5]),
        })).all()

    @choose_mark()
    def test_ne(self):
        assert (ttorch.ne(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 1], [4, 4]]),
        ) == torch.tensor([[False, True],
                           [True, False]])).all()

        assert (ttorch.ne(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': [1.0, 1.5, 2.0],
            }),
            ttorch.tensor({
                'a': [[1, 1], [4, 4]],
                'b': [1.3, 1.2, 2.0],
            }),
        ) == ttorch.tensor({
            'a': [[False, True], [True, False]],
            'b': [True, True, False],
        })).all()

    @choose_mark()
    def test_lt(self):
        assert (ttorch.lt(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 1], [4, 4]]),
        ) == torch.tensor([[False, False],
                           [True, False]])).all()

        assert (ttorch.lt(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': [1.0, 1.5, 2.0],
            }),
            ttorch.tensor({
                'a': [[1, 1], [4, 4]],
                'b': [1.3, 1.2, 2.0],
            }),
        ) == ttorch.tensor({
            'a': [[False, False], [True, False]],
            'b': [True, False, False],
        })).all()

    @choose_mark()
    def test_le(self):
        assert (ttorch.le(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 1], [4, 4]]),
        ) == torch.tensor([[True, False],
                           [True, True]])).all()

        assert (ttorch.le(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': [1.0, 1.5, 2.0],
            }),
            ttorch.tensor({
                'a': [[1, 1], [4, 4]],
                'b': [1.3, 1.2, 2.0],
            }),
        ) == ttorch.tensor({
            'a': [[True, False], [True, True]],
            'b': [True, False, True],
        })).all()

    @choose_mark()
    def test_gt(self):
        assert (ttorch.gt(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 1], [4, 4]]),
        ) == torch.tensor([[False, True],
                           [False, False]])).all()

        assert (ttorch.gt(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': [1.0, 1.5, 2.0],
            }),
            ttorch.tensor({
                'a': [[1, 1], [4, 4]],
                'b': [1.3, 1.2, 2.0],
            }),
        ) == ttorch.tensor({
            'a': [[False, True], [False, False]],
            'b': [False, True, False],
        })).all()

    @choose_mark()
    def test_ge(self):
        assert (ttorch.ge(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 1], [4, 4]]),
        ) == torch.tensor([[True, True],
                           [False, True]])).all()

        assert (ttorch.ge(
            ttorch.tensor({
                'a': [[1, 2], [3, 4]],
                'b': [1.0, 1.5, 2.0],
            }),
            ttorch.tensor({
                'a': [[1, 1], [4, 4]],
                'b': [1.3, 1.2, 2.0],
            }),
        ) == ttorch.tensor({
            'a': [[True, True], [False, True]],
            'b': [False, True, True],
        })).all()

    @choose_mark()
    def test_isfinite(self):
        t1 = ttorch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([True, False, True, False, False])).all()

        t2 = ttorch.isfinite(ttorch.tensor({
            'a': [1, float('inf'), 2, float('-inf'), float('nan')],
            'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        }))
        assert (t2 == ttorch.tensor({
            'a': [True, False, True, False, False],
            'b': {'x': [[True, False, True], [False, True, False]]},
        }))

    @choose_mark()
    def test_isinf(self):
        t1 = ttorch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([False, True, False, True, False])).all()

        t2 = ttorch.isinf(ttorch.tensor({
            'a': [1, float('inf'), 2, float('-inf'), float('nan')],
            'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        }))
        assert (t2 == ttorch.tensor({
            'a': [False, True, False, True, False],
            'b': {'x': [[False, True, False], [True, False, False]]},
        }))

    @choose_mark()
    def test_isnan(self):
        t1 = ttorch.isnan(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([False, False, False, False, True])).all()

        t2 = ttorch.isnan(ttorch.tensor({
            'a': [1, float('inf'), 2, float('-inf'), float('nan')],
            'b': {'x': [[1, float('inf'), -2], [float('-inf'), 3, float('nan')]]}
        }))
        assert (t2 == ttorch.tensor({
            'a': [False, False, False, False, True],
            'b': {'x': [[False, False, False], [False, False, True]]},
        })).all()

    @choose_mark()
    def test_isclose(self):
        t1 = ttorch.isclose(
            ttorch.tensor((1., 2, 3)),
            ttorch.tensor((1 + 1e-10, 3, 4))
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([True, False, False])).all()

        t2 = ttorch.isclose(
            ttorch.tensor({
                'a': [1., 2, 3],
                'b': {'x': [[float('inf'), 4, 1e20],
                            [-math.inf, 2.2943, 9483.32]]},
            }),
            ttorch.tensor({
                'a': [1 + 1e-10, 3, 4],
                'b': {'x': [[math.inf, 6, 1e20 + 1],
                            [-float('inf'), 2.294300000001, 9484.32]]},
            }),
        )
        assert (t2 == ttorch.tensor({
            'a': [True, False, False],
            'b': {'x': [[True, False, True],
                        [True, True, False]]},
        })).all()
