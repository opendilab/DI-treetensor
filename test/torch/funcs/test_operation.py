import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchFuncsOperation:

    @choose_mark()
    def test_cat(self):
        t1 = torch.tensor([[21, 29, 17],
                           [16, 11, 16]])
        t2 = torch.tensor([[46, 46, 46],
                           [30, 47, 36]])

        t3 = torch.tensor([[51, 65, 65],
                           [54, 67, 57]])

        assert (ttorch.cat((t1, t2, t3)) == ttorch.tensor([[21, 29, 17],
                                                           [16, 11, 16],
                                                           [46, 46, 46],
                                                           [30, 47, 36],
                                                           [51, 65, 65],
                                                           [54, 67, 57]])).all()

        tt1 = ttorch.Tensor({
            'a': t1,
            'b': {'x': t2, 'y': t3},
        })
        tt2 = ttorch.Tensor({
            'a': t2,
            'b': {'x': t3, 'y': t1},
        })
        tt3 = ttorch.Tensor({
            'a': t3,
            'b': {'x': t1, 'y': t2},
        })
        assert (ttorch.cat((tt1, tt2, tt3)) == ttorch.tensor({
            'a': [[21, 29, 17],
                  [16, 11, 16],
                  [46, 46, 46],
                  [30, 47, 36],
                  [51, 65, 65],
                  [54, 67, 57]],
            'b': {
                'x': [[46, 46, 46],
                      [30, 47, 36],
                      [51, 65, 65],
                      [54, 67, 57],
                      [21, 29, 17],
                      [16, 11, 16]],
                'y': [[51, 65, 65],
                      [54, 67, 57],
                      [21, 29, 17],
                      [16, 11, 16],
                      [46, 46, 46],
                      [30, 47, 36]],
            }})).all()
        assert (ttorch.cat((tt1, tt2, tt3), dim=1) == ttorch.tensor({
            'a': [[21, 29, 17, 46, 46, 46, 51, 65, 65],
                  [16, 11, 16, 30, 47, 36, 54, 67, 57]],
            'b': {
                'x': [[46, 46, 46, 51, 65, 65, 21, 29, 17],
                      [30, 47, 36, 54, 67, 57, 16, 11, 16]],
                'y': [[51, 65, 65, 21, 29, 17, 46, 46, 46],
                      [54, 67, 57, 16, 11, 16, 30, 47, 36]],
            }})).all()

    @choose_mark()
    def test_split(self):
        t1 = torch.tensor([[59, 82],
                           [86, 42],
                           [71, 84],
                           [61, 58],
                           [82, 37],
                           [14, 31]])
        t1_a, t1_b, t1_c = ttorch.split(t1, (1, 2, 3))
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

        tt1_a, tt1_b, tt1_c = ttorch.split(tt1, (1, 2, 3))
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
    def test_chunk(self):
        t = torch.tensor([[54, 97, 12, 48, 62],
                          [92, 87, 28, 53, 54],
                          [65, 82, 40, 26, 61],
                          [75, 43, 86, 99, 7]])
        t_a, t_b = ttorch.chunk(t, 2)
        assert isinstance(t_a, torch.Tensor)
        assert isinstance(t_b, torch.Tensor)
        assert (t_a == torch.tensor([[54, 97, 12, 48, 62],
                                     [92, 87, 28, 53, 54]])).all()
        assert (t_b == torch.tensor([[65, 82, 40, 26, 61],
                                     [75, 43, 86, 99, 7]])).all()

        tt = ttorch.tensor({
            'a': [[80, 2, 15, 45, 48],
                  [38, 89, 34, 10, 34],
                  [18, 99, 33, 38, 20],
                  [43, 21, 35, 43, 37]],
            'b': {'x': [[[19, 17, 39, 68],
                         [41, 69, 33, 89],
                         [31, 88, 39, 14]],

                        [[27, 81, 84, 35],
                         [29, 65, 17, 72],
                         [53, 50, 75, 0]]]},
        })
        tt_a, tt_b = ttorch.chunk(tt, 2)
        assert (tt_a == ttorch.tensor({
            'a': [[80, 2, 15, 45, 48],
                  [38, 89, 34, 10, 34]],
            'b': {'x': [[[19, 17, 39, 68],
                         [41, 69, 33, 89],
                         [31, 88, 39, 14]]]},
        })).all()
        assert (tt_b == ttorch.tensor({
            'a': [[18, 99, 33, 38, 20],
                  [43, 21, 35, 43, 37]],
            'b': {'x': [[[27, 81, 84, 35],
                         [29, 65, 17, 72],
                         [53, 50, 75, 0]]]},
        })).all()

    @choose_mark()
    def test_stack(self):
        t1 = torch.tensor([[17, 15, 27],
                           [12, 17, 29]])
        t2 = torch.tensor([[45, 41, 47],
                           [37, 37, 36]])
        t3 = torch.tensor([[60, 50, 55],
                           [69, 54, 58]])
        assert (ttorch.stack((t1, t2, t3)) == torch.tensor([[[17, 15, 27],
                                                             [12, 17, 29]],

                                                            [[45, 41, 47],
                                                             [37, 37, 36]],

                                                            [[60, 50, 55],
                                                             [69, 54, 58]]])).all()

        tt1 = ttorch.tensor({
            'a': [[25, 22, 29],
                  [19, 21, 27]],
            'b': {'x': [[20, 17, 28, 10],
                        [28, 16, 27, 27],
                        [18, 21, 17, 12]]},
        })
        tt2 = ttorch.tensor({
            'a': [[40, 44, 41],
                  [39, 44, 40]],
            'b': {'x': [[44, 42, 38, 44],
                        [30, 44, 42, 31],
                        [36, 30, 33, 31]]}
        })
        assert (ttorch.stack((tt1, tt2)) == ttorch.tensor({
            'a': [[[25, 22, 29],
                   [19, 21, 27]],

                  [[40, 44, 41],
                   [39, 44, 40]]],
            'b': {'x': [[[20, 17, 28, 10],
                         [28, 16, 27, 27],
                         [18, 21, 17, 12]],

                        [[44, 42, 38, 44],
                         [30, 44, 42, 31],
                         [36, 30, 33, 31]]]},
        })).all()
        assert (ttorch.stack((tt1, tt2), dim=1) == ttorch.tensor({
            'a': [[[25, 22, 29],
                   [40, 44, 41]],

                  [[19, 21, 27],
                   [39, 44, 40]]],
            'b': {'x': [[[20, 17, 28, 10],
                         [44, 42, 38, 44]],

                        [[28, 16, 27, 27],
                         [30, 44, 42, 31]],

                        [[18, 21, 17, 12],
                         [36, 30, 33, 31]]]},
        })).all()

    @choose_mark()
    def test_reshape(self):
        t1 = ttorch.reshape(torch.tensor([[1, 2], [3, 4]]), (-1,))
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([1, 2, 3, 4])).all()

        t2 = ttorch.reshape(ttorch.tensor({
            'a': [[1, 2], [3, 4]],
            'b': {'x': [[2], [3], [5], [7], [11], [13]]},
        }), (-1,))
        assert (t2 == ttorch.tensor({
            'a': [1, 2, 3, 4],
            'b': {'x': [2, 3, 5, 7, 11, 13]},
        })).all()

    @choose_mark()
    def test_squeeze(self):
        t1 = torch.randint(100, (2, 1, 2, 1, 2))
        assert t1.shape == torch.Size([2, 1, 2, 1, 2])
        assert ttorch.squeeze(t1).shape == torch.Size([2, 2, 2])

        t2 = ttorch.randint(100, {
            'a': (2, 1, 2, 1, 2),
            'b': {'x': (2, 1, 1, 3)},
        })
        assert t2.shape == ttorch.Size({
            'a': (2, 1, 2, 1, 2),
            'b': {'x': (2, 1, 1, 3)},
        })
        assert ttorch.squeeze(t2).shape == ttorch.Size({
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })

    @choose_mark()
    def test_unsqueeze(self):
        t1 = torch.randint(100, (100,))
        assert t1.shape == torch.Size([100])
        assert ttorch.unsqueeze(t1, 0).shape == torch.Size([1, 100])

        tt1 = ttorch.randint(100, {
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })
        assert tt1.shape == ttorch.Size({
            'a': (2, 2, 2),
            'b': {'x': (2, 3)},
        })
        assert ttorch.unsqueeze(tt1, 1).shape == ttorch.Size({
            'a': (2, 1, 2, 2),
            'b': {'x': (2, 1, 3)},
        })

    @choose_mark()
    def test_where(self):
        t1 = ttorch.where(
            torch.tensor([[True, False], [False, True]]),
            torch.tensor([[2, 8], [16, 4]]),
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
        assert (ttorch.where(t2 % 2 == 1, t2,
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

    @choose_mark()
    def test_index_select(self):
        t = ttorch.index_select(
            torch.tensor([[0.2247, -0.1441, -1.2249, -0.2738],
                          [-0.1496, -0.4883, -1.2442, 0.6374],
                          [0.8017, 1.1220, -2.1013, -0.5951]]),
            1, torch.tensor([1, 2])
        )
        assert isinstance(t, torch.Tensor)
        assert (t == torch.tensor([[-0.1441, -1.2249],
                                   [-0.4883, -1.2442],
                                   [1.1220, -2.1013]])).all()

        tx = ttorch.tensor({
            'a': [[3.9724e-05, -3.3134e-01, -1.0441e+00, 7.9233e-01],
                  [-1.0035e-01, 2.3422e+00, 1.9307e+00, -1.7215e-01],
                  [1.9069e+00, 1.1852e+00, -1.0672e+00, 1.3463e+00]],
            'b': {
                'x': [[0.5200, -0.3595, -1.4235, -0.2655, 0.9504, -1.7564],
                      [-1.6577, -0.5516, 0.1660, -2.3273, -0.9811, -0.4677],
                      [0.7047, -1.6920, 0.3139, 0.6220, 0.4758, -1.2637],
                      [-0.3945, -2.1694, 0.8404, -0.4224, -1.4819, 0.3998],
                      [-0.0308, 0.9777, -0.7776, -0.0101, -1.0446, -1.1500]]
            }
        })
        tt = ttorch.index_select(tx, 1, torch.tensor([1, 2]))
        assert ttorch.isclose(tt, ttorch.tensor({
            'a': [[-0.3313, -1.0441],
                  [2.3422, 1.9307],
                  [1.1852, -1.0672]],
            'b': {
                'x': [[-0.3595, -1.4235],
                      [-0.5516, 0.1660],
                      [-1.6920, 0.3139],
                      [-2.1694, 0.8404],
                      [0.9777, -0.7776]],
            }
        }), atol=1e-4).all()

        tt = ttorch.index_select(tx, 1, ttorch.tensor({'a': [1, 2], 'b': {'x': [1, 3, 5]}}))
        assert ttorch.isclose(tt, ttorch.tensor({
            'a': [[-0.3313, -1.0441],
                  [2.3422, 1.9307],
                  [1.1852, -1.0672]],
            'b': {
                'x': [[-0.3595, -0.2655, -1.7564],
                      [-0.5516, -2.3273, -0.4677],
                      [-1.6920, 0.6220, -1.2637],
                      [-2.1694, -0.4224, 0.3998],
                      [0.9777, -0.0101, -1.1500]],
            }
        }), atol=1e-4).all()
