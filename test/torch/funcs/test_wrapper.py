from unittest import skipUnless

import pytest
import torch
from hbutils.testing import vpip

import treetensor.torch as ttorch
from treetensor.torch import Size


@pytest.fixture()
def treetensor_x():
    return ttorch.randn({
        'a': (2, 5, 7),
        'b': {
            'x': (3, 4, 6),
        }
    })


@pytest.fixture()
def treetensor_y():
    return ttorch.randn({
        'a': (2, 5, 7),
        'b': {
            'x': (3, 4, 6),
        }
    })


@pytest.mark.unittest
class TestTorchTensorWrapper:
    @skipUnless(vpip('torch') >= '2', 'Torch 2 required.')
    def test_vmap(self, treetensor_x, treetensor_y):
        f = lambda x, y: (x.sum() + y.mean() * 2)
        n_pow = torch.vmap(f)
        batched_pow = ttorch.vmap(f)
        r = batched_pow(treetensor_x, treetensor_y)

        assert r.shape == Size({
            'a': (2,),
            'b': {
                'x': (3,)
            },
        })
        assert ttorch.isclose(
            r,
            ttorch.tensor({
                'a': n_pow(treetensor_x.a, treetensor_y.a),
                'b': {
                    'x': n_pow(treetensor_x.b.x, treetensor_y.b.x),
                }
            })
        ).all()

    @skipUnless(vpip('torch') >= '2', 'Torch 2 required.')
    def test_vmap_in_dims(self, treetensor_x, treetensor_y):
        f = lambda x, y: (x.sum() + y.mean() * 2)
        n_pow = torch.vmap(f, in_dims=1)
        batched_pow = ttorch.vmap(f, in_dims=1)
        r = batched_pow(treetensor_x, treetensor_y)

        assert r.shape == Size({
            'a': (5,),
            'b': {
                'x': (4,)
            },
        })
        assert ttorch.isclose(
            r,
            ttorch.tensor({
                'a': n_pow(treetensor_x.a, treetensor_y.a),
                'b': {
                    'x': n_pow(treetensor_x.b.x, treetensor_y.b.x),
                }
            })
        ).all()

    @skipUnless(vpip('torch') >= '2', 'Torch 2 required.')
    def test_vmap_nested(self, treetensor_x, treetensor_y):
        f = lambda x, y: (x.sum() + y.mean() * 2)
        n_pow = torch.vmap(torch.vmap(f))
        batched_pow = ttorch.vmap(ttorch.vmap(f))
        r = batched_pow(treetensor_x, treetensor_y)

        assert r.shape == Size({
            'a': (2, 5),
            'b': {
                'x': (3, 4)
            },
        })
        assert ttorch.isclose(
            r,
            ttorch.tensor({
                'a': n_pow(treetensor_x.a, treetensor_y.a),
                'b': {
                    'x': n_pow(treetensor_x.b.x, treetensor_y.b.x),
                }
            })
        ).all()

    @skipUnless(vpip('torch') < '2', 'Torch 1.x required.')
    def test_vmap_torch_1x(self, treetensor_x, treetensor_y):
        f = lambda x, y: (x.sum() + y.mean() * 2)
        with pytest.raises(NotImplementedError):
            _ = ttorch.vmap(f)
