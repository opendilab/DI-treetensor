import numpy as np
import pytest
import torch

import treetensor.torch as ttorch
from treetensor.torch import TensorShapePrefixConstraint, shape_prefix


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestCommonConstraintsShape:
    def test_shape_prefix(self):
        c1 = shape_prefix(2, 3, 4)
        assert isinstance(c1, TensorShapePrefixConstraint)
        assert c1.prefix == (2, 3, 4)
        assert repr(c1) == '<TensorShapePrefixConstraint (2, 3, 4)>'

        assert len(c1) == 3
        assert c1[0] == 2
        assert c1[1] == 3
        assert c1[2] == 4
        with pytest.raises(IndexError):
            _ = c1[3]
        assert c1[-1] == 4
        assert c1[-2] == 3
        assert c1[-3] == 2
        assert c1[1:] == (3, 4)

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
        assert c1 >= torch.Tensor
        assert c1 > torch.Tensor

    def test_pshape(self):
        tt = ttorch.tensor({
            'a': [[0.8479, 1.0074, 0.2725],
                  [1.1674, 1.0784, 0.0655]],
            'b': {'x': [[0.2644, 0.7268, 0.2781, 0.6469],
                        [2.0015, 0.4448, 0.8814, 1.0063],
                        [0.1847, 0.5864, 0.4417, 0.2117]]},
        })
        assert tt.pshape is None

        tt2 = tt.with_constraints(shape_prefix(2, 3), clear=False)
        assert tt2.pshape == (2, 3)
