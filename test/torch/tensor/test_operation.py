import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchTensorOperation:

    @choose_mark()
    def test_split(self):
        t1 = torch.tensor([[59, 82],
                           [86, 42],
                           [71, 84],
                           [61, 58],
                           [82, 37],
                           [14, 31]])
        t1_a, t1_b, t1_c = t1.split((1, 2, 3))
        assert (t1_a == torch.tensor([[59, 82]])).all()
        assert (t1_b == torch.tensor([[86, 42],
                                      [71, 84]])).all()
        assert (t1_c == torch.tensor([[61, 58],
                                      [82, 37],
                                      [14, 31]])).all()

        tt1 = ttorch.tensor({
            'a': [[1, 65],
                  [68, 31],
                  [76, 73],
                  [74, 76],
                  [90, 0],
                  [95, 89]],
            'b': {'x': [[[11, 20, 74],
                         [17, 85, 44]],

                        [[67, 37, 89],
                         [76, 28, 0]],

                        [[56, 12, 7],
                         [17, 63, 32]],

                        [[81, 75, 19],
                         [89, 21, 55]],

                        [[71, 53, 0],
                         [66, 82, 57]],

                        [[73, 81, 11],
                         [58, 54, 78]]]},
        })

        tt1_a, tt1_b, tt1_c = tt1.split((1, 2, 3))
        assert (tt1_a == ttorch.tensor({
            'a': [[1, 65]],
            'b': [[[11, 20, 74],
                   [17, 85, 44]]]
        })).all()
        assert (tt1_b == ttorch.tensor({
            'a': [[68, 31],
                  [76, 73]],
            'b': [[[67, 37, 89],
                   [76, 28, 0]],

                  [[56, 12, 7],
                   [17, 63, 32]]]
        })).all()
        assert (tt1_c == ttorch.tensor({
            'a': [[74, 76],
                  [90, 0],
                  [95, 89]],
            'b': [[[81, 75, 19],
                   [89, 21, 55]],

                  [[71, 53, 0],
                   [66, 82, 57]],

                  [[73, 81, 11],
                   [58, 54, 78]]]
        })).all()

    @choose_mark()
    def test_reshape(self):
        t1 = torch.tensor([[1, 2], [3, 4]]).reshape((-1,))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([1, 2, 3, 4])).all()

        t2 = ttorch.tensor({
            'a': [[1, 2], [3, 4]],
            'b': {'x': [[2], [3], [5], [7], [11], [13]]},
        }).reshape((-1,))
        assert (t2 == ttorch.tensor({
            'a': [1, 2, 3, 4],
            'b': {'x': [2, 3, 5, 7, 11, 13]},
        })).all()

    @choose_mark()
    def test_squeeze(self):
        t1 = torch.randint(100, (2, 1, 2, 1, 2))
        assert t1.shape == torch.Size([2, 1, 2, 1, 2])
        assert t1.squeeze().shape == torch.Size([2, 2, 2])

        t2 = ttorch.randint(100, {
            'a': (2, 1, 2, 1, 2),
            'b': {'x': (2, 1, 1, 3)},
        })
        assert t2.shape == ttorch.Size({
            'a': (2, 1, 2, 1, 2),
            'b': {'x': (2, 1, 1, 3)},
        })
        assert t2.squeeze().shape == ttorch.Size({
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })

    @choose_mark()
    def test_squeeze_(self):
        t1 = torch.randint(100, (2, 1, 2, 1, 2))
        assert t1.shape == torch.Size([2, 1, 2, 1, 2])
        t1r = t1.squeeze_()
        assert t1r is t1
        assert t1.shape == torch.Size([2, 2, 2])

        t2 = ttorch.randint(100, {
            'a': (2, 1, 2, 1, 2),
            'b': {'x': (2, 1, 1, 3)},
        })
        assert t2.shape == ttorch.Size({
            'a': (2, 1, 2, 1, 2),
            'b': {'x': (2, 1, 1, 3)},
        })
        t2r = t2.squeeze_()
        assert t2r is t2
        assert t2.shape == ttorch.Size({
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })

    @choose_mark()
    def test_unsqueeze(self):
        t1 = torch.randint(100, (100,))
        assert t1.shape == torch.Size([100])
        assert t1.unsqueeze(0).shape == torch.Size([1, 100])

        tt1 = ttorch.randint(100, {
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })
        assert tt1.shape == ttorch.Size({
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })
        assert tt1.unsqueeze(1).shape == ttorch.Size({
            'a': (2, 1, 2, 2),
            'b': {'x': (2, 1, 3)},
        })

    @choose_mark()
    def test_unsqueeze_(self):
        t1 = torch.randint(100, (100,))
        assert t1.shape == torch.Size([100])
        t1r = t1.unsqueeze_(0)
        assert t1r is t1
        assert t1.shape == torch.Size([1, 100])

        tt1 = ttorch.randint(100, {
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })
        assert tt1.shape == ttorch.Size({
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })
        tt1r = tt1.unsqueeze_(1)
        assert tt1r is tt1
        assert tt1.shape == ttorch.Size({
            'a': (2, 1, 2, 2),
            'b': {'x': (2, 1, 3)},
        })

    @choose_mark()
    def test_where(self):
        t1 = torch.tensor([[2, 8], [16, 4]]).where(
            torch.tensor([[True, False], [False, True]]),
            torch.tensor([[3, 11], [5, 7]]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([[2, 11],
                                     [5, 4]])).all()

        t2 = ttorch.tensor({
            'a': [[27, 90, 80],
                  [12, 59, 5]],
            'b': {'x': [[[71, 52, 92, 79],
                         [48, 4, 13, 96]],

                        [[72, 89, 44, 62],
                         [32, 4, 29, 76]],

                        [[6, 3, 93, 89],
                         [44, 89, 85, 90]]]},
        })
        assert (t2.where(t2 % 2 == 1,
                         ttorch.zeros({'a': (2, 3), 'b': {'x': (3, 2, 4)}}, dtype=torch.long)) ==
                ttorch.tensor({
                    'a': [[27, 0, 0],
                          [0, 59, 5]],
                    'b': {'x': [[[71, 0, 0, 79],
                                 [0, 0, 13, 0]],

                                [[0, 89, 0, 0],
                                 [0, 0, 29, 0]],

                                [[0, 3, 93, 89],
                                 [0, 89, 85, 0]]]},
                })).all()
