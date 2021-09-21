import numpy as np
import pytest
import torch
from treevalue import func_treelize, typetrans, TreeValue

import treetensor.numpy as tnp
import treetensor.torch as ttorch
from treetensor.common import Object

_all_is = func_treelize(return_type=ttorch.Tensor)(lambda x, y: x is y)


# noinspection PyUnresolvedReferences
@pytest.mark.unittest
class TestTorchTensor:
    _DEMO_1 = ttorch.Tensor({
        'a': [[1, 2, 3], [4, 5, 6]],
        'b': [[1, 2], [5, 6]],
        'x': {
            'c': [3, 5, 6, 7],
            'd': [[[1, 2], [8, 9]]],
        }
    })

    _DEMO_2 = ttorch.Tensor({
        'a': [[1, 2, 3], [4, 5, 6]],
        'b': [[1, 2], [5, 60]],
        'x': {
            'c': [3, 5, 6, 7],
            'd': [[[1, 2], [8, 9]]],
        }
    })

    def test_init(self):
        assert (ttorch.Tensor([1, 2, 3]) == torch.tensor([1, 2, 3])).all()
        assert (ttorch.Tensor([1, 2, 3], dtype=torch.float32) == torch.FloatTensor([1, 2, 3])).all()
        assert (self._DEMO_1 == typetrans(TreeValue({
            'a': ttorch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': ttorch.tensor([[1, 2], [5, 6]]),
            'x': {
                'c': ttorch.tensor([3, 5, 6, 7]),
                'd': ttorch.tensor([[[1, 2], [8, 9]]]),
            }
        }), ttorch.Tensor)).all()

    def test_numel(self):
        assert self._DEMO_1.numel() == 18

    def test_numpy(self):
        assert tnp.all(self._DEMO_1.numpy() == tnp.ndarray({
            'a': np.array([[1, 2, 3], [4, 5, 6]]),
            'b': np.array([[1, 2], [5, 6]]),
            'x': {
                'c': np.array([3, 5, 6, 7]),
                'd': np.array([[[1, 2], [8, 9]]]),
            }
        }))

    def test_cpu(self):
        assert ttorch.all(self._DEMO_1.cpu() == self._DEMO_1)
        assert _all_is(self._DEMO_1.cpu(), self._DEMO_1).reduce(lambda **kws: all(kws.values()))

    def test_to(self):
        assert ttorch.all(self._DEMO_1.to(torch.float32) == ttorch.Tensor({
            'a': torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.FloatTensor([[1, 2], [5, 6]]),
            'x': {
                'c': torch.FloatTensor([3, 5, 6, 7]),
                'd': torch.FloatTensor([[[1, 2], [8, 9]]]),
            }
        }))

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

    def test_tolist(self):
        assert self._DEMO_1.tolist() == Object({
            'a': [[1, 2, 3], [4, 5, 6]],
            'b': [[1, 2], [5, 6]],
            'x': {
                'c': [3, 5, 6, 7],
                'd': [[[1, 2], [8, 9]]],
            }
        })

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

    def test_max(self):
        t1 = ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }).max()
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == 3

    def test_min(self):
        t1 = ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }).min()
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == -1

    def test_sum(self):
        t1 = ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }).sum()
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == 7

    def test_eq(self):
        assert ((ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }) == ttorch.Tensor({
            'a': [1, 21],
            'b': {'x': [[-1, 3], [12, -10]]}
        })) == ttorch.Tensor({
            'a': [True, False],
            'b': {'x': [[False, True], [False, False]]}
        })).all()

    def test_ne(self):
        assert ((ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }) != ttorch.Tensor({
            'a': [1, 21],
            'b': {'x': [[-1, 3], [12, -10]]}
        })) == ttorch.Tensor({
            'a': [False, True],
            'b': {'x': [[True, False], [True, True]]}
        })).all()

    def test_lt(self):
        assert ((ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }) < ttorch.Tensor({
            'a': [1, 21],
            'b': {'x': [[-1, 3], [12, -10]]}
        })) == ttorch.Tensor({
            'a': [False, True],
            'b': {'x': [[False, False], [True, False]]}
        })).all()

    def test_le(self):
        assert ((ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }) <= ttorch.Tensor({
            'a': [1, 21],
            'b': {'x': [[-1, 3], [12, -10]]}
        })) == ttorch.Tensor({
            'a': [True, True],
            'b': {'x': [[False, True], [True, False]]}
        })).all()

    def test_gt(self):
        assert ((ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }) > ttorch.Tensor({
            'a': [1, 21],
            'b': {'x': [[-1, 3], [12, -10]]}
        })) == ttorch.Tensor({
            'a': [False, False],
            'b': {'x': [[True, False], [False, True]]}
        })).all()

    def test_ge(self):
        assert ((ttorch.Tensor({
            'a': [1, 2],
            'b': {'x': [[0, 3], [2, -1]]}
        }) >= ttorch.Tensor({
            'a': [1, 21],
            'b': {'x': [[-1, 3], [12, -10]]}
        })) == ttorch.Tensor({
            'a': [True, False],
            'b': {'x': [[True, True], [False, True]]}
        })).all()

    def test_clone(self):
        t1 = ttorch.tensor([1.0, 2.0, 1.5]).clone()
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([1.0, 2.0, 1.5])).all()

        t2 = ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        }).clone()
        assert (t2 == ttorch.tensor({
            'a': [1.0, 2.0, 1.5],
            'b': {'x': [[1.8, 0.9], [1.3, 2.5]]},
        })).all()

    def test_dot(self):
        t1 = torch.tensor([1, 2]).dot(torch.tensor([2, 3]))
        assert isinstance(t1, torch.Tensor)
        assert t1.tolist() == 8

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [3, 4]},
        }).dot(
            ttorch.tensor({
                'a': [5, 6, 7],
                'b': {'x': [1, 2]},
            })
        )
        assert (t2 == ttorch.tensor({'a': 38, 'b': {'x': 11}})).all()

    def test_matmul(self):
        t1 = torch.tensor([[1, 2], [3, 4]]).matmul(
            torch.tensor([[5, 6], [7, 2]]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([[19, 10], [43, 26]])).all()

        t2 = ttorch.tensor({
            'a': [[1, 2], [3, 4]],
            'b': {'x': [3, 4, 5, 6]},
        }).matmul(
            ttorch.tensor({
                'a': [[5, 6], [7, 2]],
                'b': {'x': [4, 3, 2, 1]},
            }),
        )
        assert (t2 == ttorch.tensor({
            'a': [[19, 10], [43, 26]],
            'b': {'x': 40}
        })).all()

    def test_mm(self):
        t1 = torch.tensor([[1, 2], [3, 4]]).mm(
            torch.tensor([[5, 6], [7, 2]]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == torch.tensor([[19, 10], [43, 26]])).all()

        t2 = ttorch.tensor({
            'a': [[1, 2], [3, 4]],
            'b': {'x': [[3, 4, 5], [6, 7, 8]]},
        }).mm(
            ttorch.tensor({
                'a': [[5, 6], [7, 2]],
                'b': {'x': [[6, 5], [4, 3], [2, 1]]},
            }),
        )
        assert (t2 == ttorch.tensor({
            'a': [[19, 10], [43, 26]],
            'b': {'x': [[44, 32], [80, 59]]},
        })).all()

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
