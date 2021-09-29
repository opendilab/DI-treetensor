from math import nan
from types import FunctionType, MethodType

import pytest

import treetensor.torch as ttorch
from treetensor.common import Object
from .base import get_mark


def method_not_implemented(name: str, need_exist=True):
    mark_0 = pytest.mark.unittest if name not in ttorch.funcs.__all__ else pytest.mark.ignore
    mark_1 = get_mark(name=name)
    need_test = all(map(lambda x: x.name == 'unittest',
                        [mark_0, *((mark_1,) if need_exist else ())]))
    return pytest.mark.unittest if need_test else pytest.mark.ignore


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchTensorAuto:
    @method_not_implemented('arctanh')
    def test_u_arctanh(self):
        tt = ttorch.tensor({
            'a': [[0.8479, 1.0074, 0.2725],
                  [1.1674, 1.0784, 0.0655]],
            'b': {'x': [[0.2644, 0.7268, 0.2781, 0.6469],
                        [2.0015, 0.4448, 0.8814, 1.0063],
                        [0.1847, 0.5864, 0.4417, 0.2117]]},
        })
        ttc = tt.clone()

        assert isinstance(ttorch.Tensor.arctanh, FunctionType)
        assert isinstance(tt.arctanh, MethodType)
        assert ttorch.isclose(tt.arctanh(), ttorch.tensor({
            'a': [[1.2487, nan, 0.2796],
                  [nan, nan, 0.0656]],
            'b': {'x': [[0.2708, 0.9219, 0.2857, 0.7699],
                        [nan, 0.4782, 1.3821, nan],
                        [0.1868, 0.6722, 0.4743, 0.2150]]}
        }), atol=1e-4, equal_nan=True).all()
        assert ttorch.isclose(tt, ttc, atol=1e-4, equal_nan=True).all()

    @method_not_implemented('arctanh_')
    def test_u_arctanh_(self):
        tt = ttorch.tensor({
            'a': [[0.8479, 1.0074, 0.2725],
                  [1.1674, 1.0784, 0.0655]],
            'b': {'x': [[0.2644, 0.7268, 0.2781, 0.6469],
                        [2.0015, 0.4448, 0.8814, 1.0063],
                        [0.1847, 0.5864, 0.4417, 0.2117]]},
        })
        ttc = tt.clone()

        assert isinstance(ttorch.Tensor.arctanh_, FunctionType)
        assert isinstance(tt.arctanh_, MethodType)
        ttr = tt.arctanh_()
        assert ttr is tt
        assert ttorch.isclose(ttr, ttorch.tensor({
            'a': [[1.2487, nan, 0.2796],
                  [nan, nan, 0.0656]],
            'b': {'x': [[0.2708, 0.9219, 0.2857, 0.7699],
                        [nan, 0.4782, 1.3821, nan],
                        [0.1868, 0.6722, 0.4743, 0.2150]]}
        }), atol=1e-4, equal_nan=True).all()
        assert not ttorch.isclose(tt, ttc, atol=1e-4, equal_nan=True).all()

    @method_not_implemented('not_found', need_exist=False)
    def test_u_not_found(self):
        with pytest.raises(AttributeError):
            _ = ttorch.Tensor.not_found

        with pytest.raises(AttributeError):
            _ = ttorch.tensor({'a': [1, 2.0]}).not_found

    @method_not_implemented('is_cuda', need_exist=False)
    def test_u_is_cuda(self):
        tt = ttorch.tensor({
            'a': [[0.8479, 1.0074, 0.2725],
                  [1.1674, 1.0784, 0.0655]],
            'b': {'x': [[0.2644, 0.7268, 0.2781, 0.6469],
                        [2.0015, 0.4448, 0.8814, 1.0063],
                        [0.1847, 0.5864, 0.4417, 0.2117]]},
        })

        icuda = tt.is_cuda
        assert isinstance(icuda, Object)
        assert icuda == Object({
            'a': False, 'b': {'x': False}
        })
