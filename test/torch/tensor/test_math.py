import math

import torch

import treetensor.torch as ttorch
from .base import choose_mark


# noinspection PyUnresolvedReferences
class TestTorchTensorMath:
    @choose_mark()
    def test_abs(self):
        t1 = ttorch.tensor([12, 0, -3]).abs()
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([12, 0, 3])).all()

        t2 = ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        }).abs()
        assert (t2 == ttorch.tensor({
            'a': [12, 0, 3],
            'b': {'x': [[3, 1], [0, 2]]},
        })).all()

    @choose_mark()
    def test_abs_(self):
        t1 = ttorch.tensor([12, 0, -3])
        t1r = t1.abs_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([12, 0, 3])).all()

        t2 = ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        })
        t2r = t2.abs_()
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [12, 0, 3],
            'b': {'x': [[3, 1], [0, 2]]},
        })).all()

    @choose_mark()
    def test_clamp(self):
        t1 = ttorch.tensor([-1.7120, 0.1734, -0.0478, 2.0922]).clamp(min=-0.5, max=0.5)
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([-0.5000, 0.1734, -0.0478, 0.5000])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [-1.7120, 0.1734, -0.0478, 2.0922],
            'b': {'x': [[-0.9049, 1.7029, -0.3697], [0.0489, -1.3127, -1.0221]]},
        }).clamp(min=-0.5, max=0.5)
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [-0.5000, 0.1734, -0.0478, 0.5000],
            'b': {'x': [[-0.5000, 0.5000, -0.3697],
                        [0.0489, -0.5000, -0.5000]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_clamp_(self):
        t1 = ttorch.tensor([-1.7120, 0.1734, -0.0478, 2.0922])
        t1r = t1.clamp_(min=-0.5, max=0.5)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([-0.5000, 0.1734, -0.0478, 0.5000])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [-1.7120, 0.1734, -0.0478, 2.0922],
            'b': {'x': [[-0.9049, 1.7029, -0.3697], [0.0489, -1.3127, -1.0221]]},
        })
        t2r = t2.clamp_(min=-0.5, max=0.5)
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [-0.5000, 0.1734, -0.0478, 0.5000],
            'b': {'x': [[-0.5000, 0.5000, -0.3697],
                        [0.0489, -0.5000, -0.5000]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_sign(self):
        t1 = ttorch.tensor([12, 0, -3]).sign()
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([1, 0, -1])).all()

        t2 = ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        }).sign()
        assert (t2 == ttorch.tensor({
            'a': [1, 0, -1],
            'b': {'x': [[-1, 1],
                        [0, -1]]},
        })).all()

    @choose_mark()
    def test_sign_(self):
        t1 = ttorch.tensor([12, 0, -3])
        t1r = t1.sign_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([1, 0, -1])).all()

        t2 = ttorch.tensor({
            'a': [12, 0, -3],
            'b': {'x': [[-3, 1], [0, -2]]},
        })
        t2r = t2.sign_()
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [1, 0, -1],
            'b': {'x': [[-1, 1],
                        [0, -1]]},
        })).all()

    @choose_mark()
    def test_round(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]).round()
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        }).round()
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-2., 3.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_round_(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        t1r = t1.round_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        })
        t2r = t2.round_()
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-2., 3.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_floor(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]).floor()
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-3., 2.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        }).floor()
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-3., 2.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 2.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_floor_(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        t1r = t1.floor_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[1., -2.],
                                               [-3., 2.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        })
        t2r = t2.floor_()
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[1., -2.],
                  [-3., 2.]],
            'b': {'x': [[1., -4., 1.],
                        [-5., -2., 2.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_ceil(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]]).ceil()
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[2., -1.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        }).ceil()
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[2., -1.],
                  [-2., 3.]],
            'b': {'x': [[1., -3., 2.],
                        [-4., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_ceil_(self):
        t1 = ttorch.tensor([[1.2, -1.8], [-2.3, 2.8]])
        t1r = t1.ceil_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([[2., -1.],
                                               [-2., 3.]])) < 1e-6).all()

        t2 = ttorch.tensor({
            'a': [[1.2, -1.8], [-2.3, 2.8]],
            'b': {'x': [[1.0, -3.9, 1.3], [-4.8, -2.0, 2.8]]},
        })
        t2r = t2.ceil_()
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [[2., -1.],
                  [-2., 3.]],
            'b': {'x': [[1., -3., 2.],
                        [-4., -2., 3.]]},
        })) < 1e-6).all()

    @choose_mark()
    def test_sigmoid(self):
        t1 = ttorch.tensor([1.0, 2.0, -1.5]).sigmoid()
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([0.7311, 0.8808, 0.1824])) < 1e-4).all()

        t2 = ttorch.tensor({
            'a': [1.0, 2.0, -1.5],
            'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        }).sigmoid()
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [0.7311, 0.8808, 0.1824],
            'b': {'x': [[0.6225, 0.7685],
                        [0.0759, 0.5622]]},
        })) < 1e-4).all()

    @choose_mark()
    def test_sigmoid_(self):
        t1 = ttorch.tensor([1.0, 2.0, -1.5])
        t1r = t1.sigmoid_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (ttorch.abs(t1 - ttorch.tensor([0.7311, 0.8808, 0.1824])) < 1e-4).all()

        t2 = ttorch.tensor({
            'a': [1.0, 2.0, -1.5],
            'b': {'x': [[0.5, 1.2], [-2.5, 0.25]]},
        })
        t2r = t2.sigmoid_()
        assert t2r is t2
        assert (ttorch.abs(t2 - ttorch.tensor({
            'a': [0.7311, 0.8808, 0.1824],
            'b': {'x': [[0.6225, 0.7685],
                        [0.0759, 0.5622]]},
        })) < 1e-4).all()

    @choose_mark()
    def test_add(self):
        t1 = ttorch.tensor([1, 2, 3]).add(
            ttorch.tensor([3, 5, 11]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([4, 7, 14])).all()

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [[3, 5], [9, 12]]},
        }).add(
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
    def test_add_(self):
        t1 = ttorch.tensor([1, 2, 3])
        t1r = t1.add_(ttorch.tensor([3, 5, 11]))
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([4, 7, 14])).all()

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [[3, 5], [9, 12]]},
        })
        t2r = t2.add_(ttorch.tensor({
            'a': [3, 5, 11],
            'b': {'x': [[31, -15], [13, 23]]},
        }))
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [4, 7, 14],
            'b': {'x': [[34, -10],
                        [22, 35]]},
        })).all()

    @choose_mark()
    def test_sub(self):
        t1 = ttorch.tensor([1, 2, 3]).sub(
            ttorch.tensor([3, 5, 11]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([-2, -3, -8])).all()

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [[3, 5], [9, 12]]},
        }).sub(
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
    def test_sub_(self):
        t1 = ttorch.tensor([1, 2, 3])
        t1r = t1.sub_(ttorch.tensor([3, 5, 11]))
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([-2, -3, -8])).all()

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [[3, 5], [9, 12]]},
        })
        t2r = t2.sub_(ttorch.tensor({
            'a': [3, 5, 11],
            'b': {'x': [[31, -15], [13, 23]]},
        }))
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [-2, -3, -8],
            'b': {'x': [[-28, 20],
                        [-4, -11]]},
        })).all()

    @choose_mark()
    def test_mul(self):
        t1 = ttorch.tensor([1, 2, 3]).mul(
            ttorch.tensor([3, 5, 11]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([3, 10, 33])).all()

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [[3, 5], [9, 12]]},
        }).mul(
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
    def test_mul_(self):
        t1 = ttorch.tensor([1, 2, 3])
        t1r = t1.mul_(ttorch.tensor([3, 5, 11]))
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([3, 10, 33])).all()

        t2 = ttorch.tensor({
            'a': [1, 2, 3],
            'b': {'x': [[3, 5], [9, 12]]},
        })
        t2r = t2.mul_(ttorch.tensor({
            'a': [3, 5, 11],
            'b': {'x': [[31, -15], [13, 23]]},
        }))
        assert t2r is t2
        assert (t2 == ttorch.tensor({
            'a': [3, 10, 33],
            'b': {'x': [[93, -75],
                        [117, 276]]},
        })).all()

    @choose_mark()
    def test_div(self):
        t1 = ttorch.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637]).div(0.5)
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([0.7620, 2.5548, -0.5944, -0.7438, 0.9274])).all()

        t2 = ttorch.tensor([1.3119, 0.0928, 0.4158, 0.7494, 0.3870]).div(
            ttorch.tensor([-1.7501, -1.4652, 0.1379, -1.1252, 0.0380]),
        )
        assert isinstance(t2, torch.Tensor)
        assert (ttorch.abs(t2 - ttorch.tensor([-0.7496, -0.0633, 3.0152, -0.6660, 10.1842])) < 1e-4).all()

        t3 = ttorch.tensor({
            'a': [0.3810, 1.2774, -0.2972, -0.3719, 0.4637],
            'b': {
                'x': [1.3119, 0.0928, 0.4158, 0.7494, 0.3870],
                'y': [[[1.9579, -0.0335, 0.1178],
                       [0.8287, 1.4520, -0.4696]],
                      [[-2.1659, -0.5831, 0.4080],
                       [0.1400, 0.8122, 0.5380]]],
            },
        }).div(
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
    def test_div_(self):
        t1 = ttorch.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637])
        t1r = t1.div_(0.5)
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([0.7620, 2.5548, -0.5944, -0.7438, 0.9274])).all()

        t2 = ttorch.tensor([1.3119, 0.0928, 0.4158, 0.7494, 0.3870])
        t2r = t2.div_(ttorch.tensor([-1.7501, -1.4652, 0.1379, -1.1252, 0.0380]))
        assert t2r is t2
        assert isinstance(t2, torch.Tensor)
        assert (ttorch.abs(t2 - ttorch.tensor([-0.7496, -0.0633, 3.0152, -0.6660, 10.1842])) < 1e-4).all()

        t3 = ttorch.tensor({
            'a': [0.3810, 1.2774, -0.2972, -0.3719, 0.4637],
            'b': {
                'x': [1.3119, 0.0928, 0.4158, 0.7494, 0.3870],
                'y': [[[1.9579, -0.0335, 0.1178],
                       [0.8287, 1.4520, -0.4696]],
                      [[-2.1659, -0.5831, 0.4080],
                       [0.1400, 0.8122, 0.5380]]],
            },
        })
        t3r = t3.div_(ttorch.tensor({
            'a': 0.5,
            'b': {
                'x': [-1.7501, -1.4652, 0.1379, -1.1252, 0.0380],
                'y': [[[-1.3136, 0.7785, -0.7290],
                       [0.6025, 0.4635, -1.1882]],
                      [[0.2756, -0.4483, -0.2005],
                       [0.9587, 1.4623, -2.8323]]],
            },
        }))
        assert t3r is t3
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
        t1 = ttorch.tensor([4, 3, 2, 6, 2]).pow(
            ttorch.tensor([4, 2, 6, 4, 3]),
        )
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([256, 9, 64, 1296, 8])).all()

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
        }).pow(
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
    def test_pow_(self):
        t1 = ttorch.tensor([4, 3, 2, 6, 2])
        t1r = t1.pow_(ttorch.tensor([4, 2, 6, 4, 3]))
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert (t1 == ttorch.tensor([256, 9, 64, 1296, 8])).all()

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
        t2r = t2.pow_(ttorch.tensor({
            'a': [4, 2, 6, 4, 3],
            'b': {
                'x': [[7, 4, 6],
                      [5, 2, 6]],
                'y': [[[7, 2, 2],
                       [2, 3, 2]],
                      [[5, 2, 6],
                       [7, 3, 4]]],
            },
        }))
        assert t2r is t2
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
        t1 = ttorch.tensor([4, 3, 2, 6, 2]).neg()
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
        }).neg()
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
        t1r = t1.neg_()
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
        t2r = t2.neg_()
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

    @choose_mark()
    def test_exp(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]).exp()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03]), rtol=1e-4).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        }).exp()
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03],
            'b': {'x': [[1.3534e-01, 3.3201e+00, 1.2840e+00],
                        [8.8861e+06, 4.2521e+01, 9.6328e-02]]},
        }), rtol=1e-4).all()

    @choose_mark()
    def test_exp_(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        t1r = t1.exp_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03]), rtol=1e-4).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        })
        t2r = t2.exp_()
        assert t2r is t2
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [1.8316e-02, 3.6788e-01, 1.0000e+00, 7.3891e+00, 1.2151e+02, 2.9810e+03],
            'b': {'x': [[1.3534e-01, 3.3201e+00, 1.2840e+00],
                        [8.8861e+06, 4.2521e+01, 9.6328e-02]]},
        }), rtol=1e-4).all()

    @choose_mark()
    def test_exp2(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]).exp2()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02]), rtol=1e-4).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        }).exp2()
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02],
            'b': {'x': [[2.5000e-01, 2.2974e+00, 1.1892e+00],
                        [6.5536e+04, 1.3454e+01, 1.9751e-01]]},
        }), rtol=1e-4).all()

    @choose_mark()
    def test_exp2_(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        t1r = t1.exp2_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02]), rtol=1e-4).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        })
        t2r = t2.exp2_()
        assert t2r is t2
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [6.2500e-02, 5.0000e-01, 1.0000e+00, 4.0000e+00, 2.7858e+01, 2.5600e+02],
            'b': {'x': [[2.5000e-01, 2.2974e+00, 1.1892e+00],
                        [6.5536e+04, 1.3454e+01, 1.9751e-01]]},
        }), rtol=1e-4).all()

    @choose_mark()
    def test_sqrt(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]).sqrt()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, 0.0000, 1.4142, 2.1909, 2.8284]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        }).sqrt()
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, 0.0000, 1.4142, 2.1909, 2.8284],
            'b': {'x': [[math.nan, 1.0954, 0.5000],
                        [4.0000, 1.9365, math.nan]]},
        }), rtol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_sqrt_(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        t1r = t1.sqrt_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, 0.0000, 1.4142, 2.1909, 2.8284]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        })
        t2r = t2.sqrt_()
        assert t2r is t2
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, 0.0000, 1.4142, 2.1909, 2.8284],
            'b': {'x': [[math.nan, 1.0954, 0.5000],
                        [4.0000, 1.9365, math.nan]]},
        }), rtol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_log(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]).log()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, -math.inf, 0.6931, 1.5686, 2.0794]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        }).log()
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, -math.inf, 0.6931, 1.5686, 2.0794],
            'b': {'x': [[math.nan, 0.1823, -1.3863],
                        [2.7726, 1.3218, math.nan]]},
        }), rtol=1e-4, atol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_log_(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        t1r = t1.log_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, -math.inf, 0.6931, 1.5686, 2.0794]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        })
        t2r = t2.log_()
        assert t2r is t2
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, -math.inf, 0.6931, 1.5686, 2.0794],
            'b': {'x': [[math.nan, 0.1823, -1.3863],
                        [2.7726, 1.3218, math.nan]]},
        }), rtol=1e-4, atol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_log2(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]).log2()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, -math.inf, 1.0000, 2.2630, 3.0000]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        }).log2()
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, -math.inf, 1.0000, 2.2630, 3.0000],
            'b': {'x': [[math.nan, 0.2630, -2.0000],
                        [4.0000, 1.9069, math.nan]]},
        }), rtol=1e-4, atol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_log2_(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        t1r = t1.log2_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, -math.inf, 1.0000, 2.2630, 3.0000]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        })
        t2r = t2.log2_()
        assert t2r is t2
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, -math.inf, 1.0000, 2.2630, 3.0000],
            'b': {'x': [[math.nan, 0.2630, -2.0000],
                        [4.0000, 1.9069, math.nan]]},
        }), rtol=1e-4, atol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_log10(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0]).log10()
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, -math.inf, 0.3010, 0.6812, 0.9031]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        }).log10()
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, -math.inf, 0.3010, 0.6812, 0.9031],
            'b': {'x': [[math.nan, 0.0792, -0.6021],
                        [1.2041, 0.5740, math.nan]]},
        }), rtol=1e-4, atol=1e-4, equal_nan=True).all()

    @choose_mark()
    def test_log10_(self):
        t1 = ttorch.tensor([-4.0, -1.0, 0, 2.0, 4.8, 8.0])
        t1r = t1.log10_()
        assert t1r is t1
        assert isinstance(t1, torch.Tensor)
        assert ttorch.isclose(t1, ttorch.tensor(
            [math.nan, math.nan, -math.inf, 0.3010, 0.6812, 0.9031]), rtol=1e-4, equal_nan=True).all()

        t2 = ttorch.tensor({
            'a': [-4.0, -1.0, 0, 2.0, 4.8, 8.0],
            'b': {'x': [[-2.0, 1.2, 0.25],
                        [16.0, 3.75, -2.34]]},
        })
        t2r = t2.log10_()
        assert t2r is t2
        assert ttorch.isclose(t2, ttorch.tensor({
            'a': [math.nan, math.nan, -math.inf, 0.3010, 0.6812, 0.9031],
            'b': {'x': [[math.nan, 0.0792, -0.6021],
                        [1.2041, 0.5740, math.nan]]},
        }), rtol=1e-4, atol=1e-4, equal_nan=True).all()
