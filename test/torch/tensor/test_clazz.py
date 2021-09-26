import numpy as np
import torch
from treevalue import typetrans, TreeValue, func_treelize

import treetensor.numpy as tnp
import treetensor.torch as ttorch
from treetensor.common import Object
from .base import choose_mark

_all_is = func_treelize(return_type=ttorch.Tensor)(lambda x, y: x is y)


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchTensorClass:
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

    @choose_mark()
    def test___init__(self):
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

    @choose_mark()
    def test_numel(self):
        assert self._DEMO_1.numel() == 18

    @choose_mark()
    def test_numpy(self):
        assert tnp.all(self._DEMO_1.numpy() == tnp.ndarray({
            'a': np.array([[1, 2, 3], [4, 5, 6]]),
            'b': np.array([[1, 2], [5, 6]]),
            'x': {
                'c': np.array([3, 5, 6, 7]),
                'd': np.array([[[1, 2], [8, 9]]]),
            }
        }))

    @choose_mark()
    def test_cpu(self):
        assert ttorch.all(self._DEMO_1.cpu() == self._DEMO_1)
        assert _all_is(self._DEMO_1.cpu(), self._DEMO_1).reduce(lambda **kws: all(kws.values()))

    @choose_mark()
    def test_to(self):
        assert ttorch.all(self._DEMO_1.to(torch.float32) == ttorch.Tensor({
            'a': torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.FloatTensor([[1, 2], [5, 6]]),
            'x': {
                'c': torch.FloatTensor([3, 5, 6, 7]),
                'd': torch.FloatTensor([[[1, 2], [8, 9]]]),
            }
        }))

    @choose_mark()
    def test_tolist(self):
        assert self._DEMO_1.tolist() == Object({
            'a': [[1, 2, 3], [4, 5, 6]],
            'b': [[1, 2], [5, 6]],
            'x': {
                'c': [3, 5, 6, 7],
                'd': [[[1, 2], [8, 9]]],
            }
        })

    @choose_mark()
    def test___eq__(self):
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

    @choose_mark()
    def test___ne__(self):
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

    @choose_mark()
    def test___lt__(self):
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

    @choose_mark()
    def test___le__(self):
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

    @choose_mark()
    def test___gt__(self):
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

    @choose_mark()
    def test___ge__(self):
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

    @choose_mark()
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
