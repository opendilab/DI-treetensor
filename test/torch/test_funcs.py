import torch

import treetensor.torch as ttorch
from treetensor.utils import replaceable_partial
from ..tests import choose_mark_with_existence_check

choose_mark = replaceable_partial(choose_mark_with_existence_check, base=ttorch)


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchFuncs:
    @choose_mark()
    def test_tensor(self):
        t1 = ttorch.tensor(True)
        assert isinstance(t1, torch.Tensor)
        assert t1

        t2 = ttorch.tensor([[1, 2, 3], [4, 5, 6]])
        assert isinstance(t2, torch.Tensor)
        assert (t2 == torch.tensor([[1, 2, 3], [4, 5, 6]])).all()

        t3 = ttorch.tensor({
            'a': [1, 2],
            'b': [[3, 4], [5, 6.2]],
            'x': {
                'c': True,
                'd': [False, True],
            }
        })
        assert isinstance(t3, ttorch.Tensor)
        assert (t3 == ttorch.Tensor({
            'a': torch.tensor([1, 2]),
            'b': torch.tensor([[3, 4], [5, 6.2]]),
            'x': {
                'c': torch.tensor(True),
                'd': torch.tensor([False, True]),
            }
        })).all()

    @choose_mark()
    def test_zeros(self):
        assert ttorch.all(ttorch.zeros(2, 3) == torch.zeros(2, 3))
        assert ttorch.all(ttorch.zeros({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }) == ttorch.Tensor({
            'a': torch.zeros(2, 3),
            'b': torch.zeros(5, 6),
            'x': {
                'c': torch.zeros(2, 3, 4),
            }
        }))

    @choose_mark()
    def test_zeros_like(self):
        assert ttorch.all(
            ttorch.zeros_like(torch.tensor([[1, 2, 3], [4, 5, 6]])) ==
            torch.tensor([[0, 0, 0], [0, 0, 0]]),
        )
        assert ttorch.all(
            ttorch.zeros_like({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            }) == ttorch.Tensor({
                'a': torch.tensor([[0, 0, 0], [0, 0, 0]]),
                'b': torch.tensor([0, 0, 0, 0]),
                'x': {
                    'c': torch.tensor([0, 0, 0]),
                    'd': torch.tensor([[[0, 0]]]),
                }
            })
        )

    @choose_mark()
    def test_ones(self):
        assert ttorch.all(ttorch.ones(2, 3) == torch.ones(2, 3))
        assert ttorch.all(ttorch.ones({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }) == ttorch.Tensor({
            'a': torch.ones(2, 3),
            'b': torch.ones(5, 6),
            'x': {
                'c': torch.ones(2, 3, 4),
            }
        }))

    @choose_mark()
    def test_ones_like(self):
        assert ttorch.all(
            ttorch.ones_like(torch.tensor([[1, 2, 3], [4, 5, 6]])) ==
            torch.tensor([[1, 1, 1], [1, 1, 1]])
        )
        assert ttorch.all(
            ttorch.ones_like({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            }) == ttorch.Tensor({
                'a': torch.tensor([[1, 1, 1], [1, 1, 1]]),
                'b': torch.tensor([1, 1, 1, 1]),
                'x': {
                    'c': torch.tensor([1, 1, 1]),
                    'd': torch.tensor([[[1, 1]]]),
                }
            })
        )

    @choose_mark()
    def test_randn(self):
        _target = ttorch.randn(200, 300)
        assert -0.02 <= _target.view(60000).mean().tolist() <= 0.02
        assert 0.98 <= _target.view(60000).std().tolist() <= 1.02
        assert _target.shape == torch.Size([200, 300])

        _target = ttorch.randn({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    @choose_mark()
    def test_randn_like(self):
        _target = ttorch.randn_like(torch.ones(200, 300))
        assert -0.02 <= _target.view(60000).mean().tolist() <= 0.02
        assert 0.98 <= _target.view(60000).std().tolist() <= 1.02
        assert _target.shape == torch.Size([200, 300])

        _target = ttorch.randn_like({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
            'b': torch.tensor([1, 2, 3, 4], dtype=torch.float32),
            'x': {
                'c': torch.tensor([5, 6, 7], dtype=torch.float32),
                'd': torch.tensor([[[8, 9]]], dtype=torch.float32),
            }
        })
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    @choose_mark()
    def test_randint(self):
        _target = ttorch.randint(-10, 10, {
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })
        assert ttorch.all(_target < 10)
        assert ttorch.all(-10 <= _target)
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

        _target = ttorch.randint(10, {
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })
        assert ttorch.all(_target < 10)
        assert ttorch.all(0 <= _target)
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    @choose_mark()
    def test_randint_like(self):
        _target = ttorch.randint_like({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        }, -10, 10)
        assert ttorch.all(_target < 10)
        assert ttorch.all(-10 <= _target)
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

        _target = ttorch.randint_like({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        }, 10)
        assert ttorch.all(_target < 10)
        assert ttorch.all(0 <= _target)
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    @choose_mark()
    def test_full(self):
        _target = ttorch.full({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }, 233)
        assert ttorch.all(_target == 233)
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    @choose_mark()
    def test_full_like(self):
        _target = ttorch.full_like({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        }, 233)
        assert ttorch.all(_target == 233)
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    @choose_mark()
    def test_empty(self):
        _target = ttorch.empty({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    @choose_mark()
    def test_empty_like(self):
        _target = ttorch.empty_like({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        })
        assert _target.shape == ttorch.Size({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

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
    def test_eq(self):
        assert ttorch.eq(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])).all()
        assert not ttorch.eq(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 2])).all()
        assert ttorch.eq(torch.tensor([1, 1, 1]), 1).all()
        assert not ttorch.eq(torch.tensor([1, 1, 2]), 1).all()

        assert ttorch.eq({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }, ({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        })).all()
        assert not ttorch.eq({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }, ({
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
    def test_equal(self):
        p1 = ttorch.equal(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
        assert isinstance(p1, bool)
        assert p1

        p2 = ttorch.equal(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4]))
        assert isinstance(p2, bool)
        assert not p2

        p3 = ttorch.equal({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }, ({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }))
        assert isinstance(p3, bool)
        assert p3

        p4 = ttorch.equal({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
        }, ({
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 5]),
        }))
        assert isinstance(p4, bool)
        assert not p4

    @choose_mark()
    def test_min(self):
        t1 = ttorch.min(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert t1 == torch.tensor(1.0)

        assert ttorch.min(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })) == ttorch.tensor({
            'a': 1.0,
            'b': {'x': 0.9},
        })

    @choose_mark()
    def test_max(self):
        t1 = ttorch.max(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert t1 == torch.tensor(2.0)

        assert ttorch.max(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })) == ttorch.tensor({
            'a': 2.0,
            'b': {'x': 2.5, }
        })

    @choose_mark()
    def test_sum(self):
        assert ttorch.sum(torch.tensor([1.0, 2.0, 1.5])) == torch.tensor(4.5)
        assert ttorch.sum(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })) == torch.tensor(11.0)

    @choose_mark()
    def test_clone(self):
        t1 = ttorch.clone(torch.tensor([1.0, 2.0, 1.5]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([1.0, 2.0, 1.5])).all()

        t2 = ttorch.clone(ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        }))
        assert (t2 == ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })).all()

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
    def test_abs(self):
        t1 = ttorch.abs(ttorch.tensor([12, 0, -3]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([12, 0, 3])).all()

        t2 = ttorch.abs(ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        }))
        assert (t2 == ttorch.tensor({
            'a': [12, 0, 3],
            'b': {'x': [[3, 1], [0, 2]]},
        })).all()

    @choose_mark()
    def test_abs_(self):
        t1 = ttorch.tensor([12, 0, -3])
        assert isinstance(t1, torch.Tensor)

        t1r = ttorch.abs_(t1)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([12, 0, 3])).all()

        t2 = ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        })
        t2r = ttorch.abs_(t2)
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [12, 0, 3],
            'b': {'x': [[3, 1], [0, 2]]},
        })).all()

    @choose_mark()
    def test_clamp(self):
        t1 = ttorch.clamp(ttorch.tensor([-1.7120, 0.1734, -0.0478, 2.0922]), min=-0.5, max=0.5)
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([-0.5000, 0.1734, -0.0478, 0.5000])) < 1e-6).all()

        t2 = ttorch.clamp(ttorch.tensor({
            'a': [-1.7120, 0.1734, -0.0478, 2.0922],
            'b': {'x': [[-0.9049, 1.7029, -0.3697], [0.0489, -1.3127, -1.0221]]},
        }), min=-0.5, max=0.5)
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [-0.5000, 0.1734, -0.0478, 0.5000],
            'b': {'x': [[-0.5000, 0.5000, -0.3697],
                        [0.0489, -0.5000, -0.5000]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_clamp_(self):
        t1 = ttorch.tensor([-1.7120, 0.1734, -0.0478, 2.0922])
        t1r = ttorch.clamp_(t1, min=-0.5, max=0.5)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([-0.5000, 0.1734, -0.0478, 0.5000])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [-1.7120, 0.1734, -0.0478, 2.0922],
            'b': {'x': [[-0.9049, 1.7029, -0.3697], [0.0489, -1.3127, -1.0221]]},
        })
        t2r = ttorch.clamp_(t2, min=-0.5, max=0.5)
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [-0.5000, 0.1734, -0.0478, 0.5000],
            'b': {'x': [[-0.5000, 0.5000, -0.3697],
                        [0.0489, -0.5000, -0.5000]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_sign(self):
        t1 = ttorch.sign(ttorch.tensor([12, 0, -3]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([1, 0, -1])).all()

        t2 = ttorch.sign(ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        }))
        assert (t2 == ttorch.tensor({
            'a': [1, 0, -1],
            'b': {'x': [[-1, 1],
                        [0, -1]]},
        })).all()

    @choose_mark()
    def test_round(self):
        t1 = ttorch.round(ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]))
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.round(ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        }))
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-2., 3.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_round_(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        t1r = ttorch.round_(t1)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        })
        t2r = ttorch.round_(t2)
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-2., 3.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_floor(self):
        t1 = ttorch.floor(ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]))
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-3., 2.]])) < 1e-6).all()

        t2 = ttorch.floor(ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        }))
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-3., 2.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 2.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_floor_(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        t1r = ttorch.floor_(t1)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-3., 2.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        })
        t2r = ttorch.floor_(t2)
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-3., 2.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 2.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_ceil(self):
        t1 = ttorch.ceil(ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]))
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[2., -1.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.ceil(ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        }))
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[2., -1.],
                  [-2., 3.]],
            'b': {'x': [[1., -3., 2.],
                        [-4., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_ceil_(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        t1r = ttorch.ceil_(t1)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[2., -1.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        })
        t2r = ttorch.ceil_(t2)
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[2., -1.],
                  [-2., 3.]],
            'b': {'x': [[1., -3., 2.],
                        [-4., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_sigmoid(self):
        t1 = ttorch.sigmoid(ttorch.tensor([1.0, 2.0, -1.5]))
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([0.7311, 0.8808, 0.1824])) < 1e-4).all()

        t2 = ttorch.sigmoid(ttorch.tensor({
            'a': [1.0, 2.0, -1.5],
            'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        }))
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [0.7311, 0.8808, 0.1824],
            'b': {'x': [[0.6225, 0.7685],
                        [0.0759, 0.5622]]},
        })) < 1e-4).all()

    @choose_mark()
    def test_sigmoid_(self):
        t1 = ttorch.tensor([1.0, 2.0, -1.5])
        t1r = ttorch.sigmoid_(t1)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([0.7311, 0.8808, 0.1824])) < 1e-4).all()

        t2 = ttorch.tensor({
            'a': [1.0, 2.0, -1.5],
            'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        })
        t2r = ttorch.sigmoid_(t2)
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [0.7311, 0.8808, 0.1824],
            'b': {'x': [[0.6225, 0.7685],
                        [0.0759, 0.5622]]},
        })) < 1e-4).all()

    @choose_mark()
    def test_add(self):
        t1 = ttorch.add(
            ttorch.tensor([1, 2, 3]),
            ttorch.tensor([3, 5, 11]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([4, 7, 14])).all()

        t2 = ttorch.add(
            ttorch.tensor({
                'a': [1, 2, 3],
                'b': {'x': [[3, 5], [9, 12]]},
            }),
            ttorch.tensor({
                'a': [3, 5, 11],
                'b': {'x': [[31, -15], [13, 23]]},
            })
        )
        assert (t2 == ttorch.tensor({
            'a': [4, 7, 14],
            'b': {'x': [[34, -10],
                        [22, 35]]},
        })).all()

    @choose_mark()
    def test_sub(self):
        t1 = ttorch.sub(
            ttorch.tensor([1, 2, 3]),
            ttorch.tensor([3, 5, 11]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([-2, -3, -8])).all()

        t2 = ttorch.sub(
            ttorch.tensor({
                'a': [1, 2, 3],
                'b': {'x': [[3, 5], [9, 12]]},
            }),
            ttorch.tensor({
                'a': [3, 5, 11],
                'b': {'x': [[31, -15], [13, 23]]},
            })
        )
        assert (t2 == ttorch.tensor({
            'a': [-2, -3, -8],
            'b': {'x': [[-28, 20],
                        [-4, -11]]},
        })).all()

    @choose_mark()
    def test_mul(self):
        t1 = ttorch.mul(
            ttorch.tensor([1, 2, 3]),
            ttorch.tensor([3, 5, 11]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([3, 10, 33])).all()

        t2 = ttorch.mul(
            ttorch.tensor({
                'a': [1, 2, 3],
                'b': {'x': [[3, 5], [9, 12]]},
            }),
            ttorch.tensor({
                'a': [3, 5, 11],
                'b': {'x': [[31, -15], [13, 23]]},
            })
        )
        assert (t2 == ttorch.tensor({
            'a': [3, 10, 33],
            'b': {'x': [[93, -75],
                        [117, 276]]},
        })).all()

    @choose_mark()
    def test_div(self):
        t1 = ttorch.div(ttorch.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637]), 0.5)
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([0.7620, 2.5548, -0.5944, -0.7438, 0.9274])).all()

        t2 = ttorch.div(
            ttorch.tensor([1.3119, 0.0928, 0.4158, 0.7494, 0.3870]),
            ttorch.tensor([-1.7501, -1.4652, 0.1379, -1.1252, 0.0380]),
        )
        assert isinstance(t2, torch.Tensor)
        assert (ttorch.abs(t2 - ttorch.tensor([-0.7496, -0.0633, 3.0152, -0.6660, 10.1842])) < 1e-4).all()

        t3 = ttorch.div(
            ttorch.tensor({
                'a': [0.3810, 1.2774, -0.2972, -0.3719, 0.4637],
                'b': {
                    'x': [1.3119, 0.0928, 0.4158, 0.7494, 0.3870],
                    'y': [[[1.9579, -0.0335, 0.1178],
                           [0.8287, 1.4520, -0.4696]],
                          [[-2.1659, -0.5831, 0.4080],
                           [0.1400, 0.8122, 0.5380]]],
                },
            }),
            ttorch.tensor({
                'a': 0.5,
                'b': {
                    'x': [-1.7501, -1.4652, 0.1379, -1.1252, 0.0380],
                    'y': [[[-1.3136, 0.7785, -0.7290],
                           [0.6025, 0.4635, -1.1882]],
                          [[0.2756, -0.4483, -0.2005],
                           [0.9587, 1.4623, -2.8323]]],
                },
            }),
        )
        assert (ttorch.abs(t3 - ttorch.tensor({
            'a': [0.7620, 2.5548, -0.5944, -0.7438, 0.9274],
            'b': {
                'x': [-0.7496, -0.0633, 3.0152, -0.6660, 10.1842],
                'y': [[[-1.4905, -0.0430, -0.1616],
                       [1.3754, 3.1327, 0.3952]],

                      [[-7.8589, 1.3007, -2.0349],
                       [0.1460, 0.5554, -0.1900]]],
            }
        })) < 1e-4).all()

    @choose_mark()
    def test_pow(self):
        t1 = ttorch.pow(
            ttorch.tensor([4, 3, 2, 6, 2]),
            ttorch.tensor([4, 2, 6, 4, 3]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([256, 9, 64, 1296, 8])).all()

        t2 = ttorch.pow(
            ttorch.tensor({
                'a': [4, 3, 2, 6, 2],
                'b': {
                    'x': [[3, 4, 6],
                          [6, 3, 5]],
                    'y': [[[3, 5, 5],
                           [5, 7, 6]],
                          [[4, 6, 5],
                           [7, 2, 7]]],
                },
            }),
            ttorch.tensor({
                'a': [4, 2, 6, 4, 3],
                'b': {
                    'x': [[7, 4, 6],
                          [5, 2, 6]],
                    'y': [[[7, 2, 2],
                           [2, 3, 2]],
                          [[5, 2, 6],
                           [7, 3, 4]]],
                },
            }),
        )
        assert (t2 == ttorch.tensor({
            'a': [256, 9, 64, 1296, 8],
            'b': {
                'x': [[2187, 256, 46656],
                      [7776, 9, 15625]],
                'y': [[[2187, 25, 25],
                       [25, 343, 36]],

                      [[1024, 36, 15625],
                       [823543, 8, 2401]]],
            }
        })).all()

    @choose_mark()
    def test_neg(self):
        t1 = ttorch.neg(ttorch.tensor([4, 3, 2, 6, 2]))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([-4, -3, -2, -6, -2])).all()

        t2 = ttorch.neg(ttorch.tensor({
            'a': [4, 3, 2, 6, 2],
            'b': {
                'x': [[3, 4, 6],
                      [6, 3, 5]],
                'y': [[[3, 5, 5],
                       [5, 7, 6]],
                      [[4, 6, 5],
                       [7, 2, 7]]],
            },
        }))
        assert (t2 == ttorch.tensor({
            'a': [-4, -3, -2, -6, -2],
            'b': {
                'x': [[-3, -4, -6],
                      [-6, -3, -5]],
                'y': [[[-3, -5, -5],
                       [-5, -7, -6]],
                      [[-4, -6, -5],
                       [-7, -2, -7]]],
            }
        }))

    @choose_mark()
    def test_neg_(self):
        t1 = ttorch.tensor([4, 3, 2, 6, 2])
        t1r = ttorch.neg_(t1)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([-4, -3, -2, -6, -2])).all()

        t2 = ttorch.tensor({
            'a': [4, 3, 2, 6, 2],
            'b': {
                'x': [[3, 4, 6],
                      [6, 3, 5]],
                'y': [[[3, 5, 5],
                       [5, 7, 6]],
                      [[4, 6, 5],
                       [7, 2, 7]]],
            },
        })
        t2r = ttorch.neg_(t2)
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [-4, -3, -2, -6, -2],
            'b': {
                'x': [[-3, -4, -6],
                      [-6, -3, -5]],
                'y': [[[-3, -5, -5],
                       [-5, -7, -6]],
                      [[-4, -6, -5],
                       [-7, -2, -7]]],
            }
        }))
