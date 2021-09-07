import pytest

from treetensor import TreeNumpy

try:
    import numpy as np
except ImportError:
    need_fake = True
    from treetensor.numpy.fake import FakeTreeNumpy
else:
    need_fake = False

unittest_mark = pytest.mark.unittest if need_fake else pytest.mark.ignore


@unittest_mark
class TestNumpyFake:
    def test_base(self):
        assert TreeNumpy is FakeTreeNumpy
