import numpy as np
import pytest
import torch

from treetensor.common import ShapePrefixConstraint, shape_prefix


class NumpyShapePrefixConstraint(ShapePrefixConstraint):
    __type__ = np.ndarray


class TorchShapePrefixConstraint(ShapePrefixConstraint):
    __type__ = torch.Tensor


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestCommonConstraintsShape:
    def test_shape_prefix(self):
        c1 = shape_prefix(2, 3, 4)
        assert isinstance(c1, ShapePrefixConstraint)
        assert c1.prefix == (2, 3, 4)
        assert repr(c1) == '<ShapePrefixConstraint (2, 3, 4)>'

        c1.validate(np.random.rand(2, 3, 4))
        c1.validate(np.random.rand(2, 3, 4, 5))
        with pytest.raises(ValueError):
            c1.validate(np.random.rand(2, 3))
        with pytest.raises(ValueError):
            c1.validate(np.random.rand(2, 3, 3))
        with pytest.raises(ValueError):
            c1.validate(np.random.rand(2, 3, 3, 4))
        with pytest.raises(TypeError):
            c1.validate([2, 3, 4, 5])

        c1.validate(torch.randn(2, 3, 4))
        c1.validate(torch.randn(2, 3, 4, 5))
        with pytest.raises(ValueError):
            c1.validate(torch.randn(2, 3))
        with pytest.raises(ValueError):
            c1.validate(torch.randn(2, 3, 3))
        with pytest.raises(ValueError):
            c1.validate(torch.randn(2, 3, 3, 4))
        with pytest.raises(TypeError):
            c1.validate([2, 3, 4, 5])

        assert c1 == shape_prefix(2, 3, 4)
        assert not c1 != shape_prefix(2, 3, 4)
        assert c1 >= shape_prefix(2, 3, 4)
        assert c1 <= shape_prefix(2, 3, 4)
        assert not c1 > shape_prefix(2, 3, 4)
        assert not c1 < shape_prefix(2, 3, 4)

        assert not c1 == shape_prefix(2, 3)
        assert c1 != shape_prefix(2, 3)
        assert c1 >= shape_prefix(2, 3)
        assert not c1 <= shape_prefix(2, 3)
        assert c1 > shape_prefix(2, 3)
        assert not c1 < shape_prefix(2, 3)

        assert not c1 == shape_prefix(2, 3, 4, 5)
        assert c1 != shape_prefix(2, 3, 4, 5)
        assert not c1 >= shape_prefix(2, 3, 4, 5)
        assert c1 <= shape_prefix(2, 3, 4, 5)
        assert not c1 > shape_prefix(2, 3, 4, 5)
        assert c1 < shape_prefix(2, 3, 4, 5)

        assert not c1 == shape_prefix(2, 3, 3)
        assert c1 != shape_prefix(2, 3, 3)
        assert not c1 >= shape_prefix(2, 3, 3)
        assert not c1 <= shape_prefix(2, 3, 3)
        assert not c1 > shape_prefix(2, 3, 3)
        assert not c1 < shape_prefix(2, 3, 3)

        assert not c1 >= np.ndarray
        assert not c1 > np.ndarray
        assert not c1 >= torch.Tensor
        assert not c1 > torch.Tensor

    def test_shape_prefix_numpy(self):
        def nsp(*prefix):
            return shape_prefix(*prefix, type_=NumpyShapePrefixConstraint)

        c1 = nsp(2, 3, 4)
        assert isinstance(c1, NumpyShapePrefixConstraint)
        assert c1.prefix == (2, 3, 4)
        assert repr(c1) == '<NumpyShapePrefixConstraint (2, 3, 4)>'

        c1.validate(np.random.rand(2, 3, 4))
        c1.validate(np.random.rand(2, 3, 4, 5))
        with pytest.raises(ValueError):
            c1.validate(np.random.rand(2, 3))
        with pytest.raises(ValueError):
            c1.validate(np.random.rand(2, 3, 3))
        with pytest.raises(ValueError):
            c1.validate(np.random.rand(2, 3, 3, 4))
        with pytest.raises(TypeError):
            c1.validate([2, 3, 4, 5])

        with pytest.raises(TypeError):
            c1.validate(torch.randn(2, 3, 4))
        with pytest.raises(TypeError):
            c1.validate(torch.randn(2, 3, 4, 5))
        with pytest.raises(TypeError):
            c1.validate(torch.randn(2, 3))
        with pytest.raises(TypeError):
            c1.validate(torch.randn(2, 3, 3))
        with pytest.raises(TypeError):
            c1.validate(torch.randn(2, 3, 3, 4))
        with pytest.raises(TypeError):
            c1.validate([2, 3, 4, 5])

        assert c1 == nsp(2, 3, 4)
        assert not c1 != nsp(2, 3, 4)
        assert c1 >= nsp(2, 3, 4)
        assert c1 <= nsp(2, 3, 4)
        assert not c1 > nsp(2, 3, 4)
        assert not c1 < nsp(2, 3, 4)

        assert not c1 == nsp(2, 3)
        assert c1 != nsp(2, 3)
        assert c1 >= nsp(2, 3)
        assert not c1 <= nsp(2, 3)
        assert c1 > nsp(2, 3)
        assert not c1 < nsp(2, 3)

        assert not c1 == nsp(2, 3, 4, 5)
        assert c1 != nsp(2, 3, 4, 5)
        assert not c1 >= nsp(2, 3, 4, 5)
        assert c1 <= nsp(2, 3, 4, 5)
        assert not c1 > nsp(2, 3, 4, 5)
        assert c1 < nsp(2, 3, 4, 5)

        assert not c1 == nsp(2, 3, 3)
        assert c1 != nsp(2, 3, 3)
        assert not c1 >= nsp(2, 3, 3)
        assert not c1 <= nsp(2, 3, 3)
        assert not c1 > nsp(2, 3, 3)
        assert not c1 < nsp(2, 3, 3)

        assert not c1 == shape_prefix(2, 3, 4)
        assert c1 != shape_prefix(2, 3, 4)
        assert c1 >= shape_prefix(2, 3, 4)
        assert not c1 <= shape_prefix(2, 3, 4)
        assert c1 > shape_prefix(2, 3, 4)
        assert not c1 < shape_prefix(2, 3, 4)

        assert c1 >= np.ndarray
        assert c1 > np.ndarray
        assert not c1 >= torch.Tensor
        assert not c1 > torch.Tensor

    def test_shape_prefix_torch(self):
        def tsp(*prefix):
            return shape_prefix(*prefix, type_=TorchShapePrefixConstraint)

        c1 = tsp(2, 3, 4)
        assert isinstance(c1, TorchShapePrefixConstraint)
        assert c1.prefix == (2, 3, 4)
        assert repr(c1) == '<TorchShapePrefixConstraint (2, 3, 4)>'

        with pytest.raises(TypeError):
            c1.validate(np.random.rand(2, 3, 4))
        with pytest.raises(TypeError):
            c1.validate(np.random.rand(2, 3, 4, 5))
        with pytest.raises(TypeError):
            c1.validate(np.random.rand(2, 3))
        with pytest.raises(TypeError):
            c1.validate(np.random.rand(2, 3, 3))
        with pytest.raises(TypeError):
            c1.validate(np.random.rand(2, 3, 3, 4))
        with pytest.raises(TypeError):
            c1.validate([2, 3, 4, 5])

        c1.validate(torch.randn(2, 3, 4))
        c1.validate(torch.randn(2, 3, 4, 5))
        with pytest.raises(ValueError):
            c1.validate(torch.randn(2, 3))
        with pytest.raises(ValueError):
            c1.validate(torch.randn(2, 3, 3))
        with pytest.raises(ValueError):
            c1.validate(torch.randn(2, 3, 3, 4))
        with pytest.raises(TypeError):
            c1.validate([2, 3, 4, 5])

        assert c1 == tsp(2, 3, 4)
        assert not c1 != tsp(2, 3, 4)
        assert c1 >= tsp(2, 3, 4)
        assert c1 <= tsp(2, 3, 4)
        assert not c1 > tsp(2, 3, 4)
        assert not c1 < tsp(2, 3, 4)

        assert not c1 == tsp(2, 3)
        assert c1 != tsp(2, 3)
        assert c1 >= tsp(2, 3)
        assert not c1 <= tsp(2, 3)
        assert c1 > tsp(2, 3)
        assert not c1 < tsp(2, 3)

        assert not c1 == tsp(2, 3, 4, 5)
        assert c1 != tsp(2, 3, 4, 5)
        assert not c1 >= tsp(2, 3, 4, 5)
        assert c1 <= tsp(2, 3, 4, 5)
        assert not c1 > tsp(2, 3, 4, 5)
        assert c1 < tsp(2, 3, 4, 5)

        assert not c1 == tsp(2, 3, 3)
        assert c1 != tsp(2, 3, 3)
        assert not c1 >= tsp(2, 3, 3)
        assert not c1 <= tsp(2, 3, 3)
        assert not c1 > tsp(2, 3, 3)
        assert not c1 < tsp(2, 3, 3)

        assert not c1 == shape_prefix(2, 3, 4)
        assert c1 != shape_prefix(2, 3, 4)
        assert c1 >= shape_prefix(2, 3, 4)
        assert not c1 <= shape_prefix(2, 3, 4)
        assert c1 > shape_prefix(2, 3, 4)
        assert not c1 < shape_prefix(2, 3, 4)

        assert not c1 >= np.ndarray
        assert not c1 > np.ndarray
        assert c1 >= torch.Tensor
        assert c1 > torch.Tensor

    def test_shape_prefix_cross(self):
        c1 = shape_prefix(2, 3, 4, type_=NumpyShapePrefixConstraint)
        c2 = shape_prefix(2, 3, 4, type_=TorchShapePrefixConstraint)
        assert not c1 == c2
        assert c1 != c2
        assert not c1 >= c2
        assert not c1 > c2
        assert not c1 <= c2
        assert not c1 < c2
