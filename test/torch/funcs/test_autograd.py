import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchFuncsAutograd:

    @choose_mark()
    def test_detach(self):
        tt1 = ttorch.tensor({
            'a': [2, 3, 4.0],
            'b': {'x': [[5, 6], [7, 8.0]]}
        }, requires_grad=True)
        assert tt1.requires_grad.all()

        tt1r = ttorch.detach(tt1)
        assert tt1.requires_grad.all()
        assert tt1r is not tt1
        assert not tt1r.requires_grad.any()

    @choose_mark()
    def test_detach_(self):
        tt1 = ttorch.tensor({
            'a': [2, 3, 4.0],
            'b': {'x': [[5, 6], [7, 8.0]]}
        }, requires_grad=True)
        assert tt1.requires_grad.all()

        tt1r = ttorch.detach_(tt1)
        assert tt1r is tt1
        assert not tt1.requires_grad.any()
