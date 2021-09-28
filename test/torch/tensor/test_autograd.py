import treetensor.torch as ttorch
from .base import choose_mark


# noinspection DuplicatedCode,PyUnresolvedReferences
class TestTorchTensorAutograd:
    @choose_mark()
    def test_requires_grad(self):
        tt1 = ttorch.tensor({
            'a': [2, 3, 4.0],
            'b': {'x': [[5, 6], [7, 8.0]]}
        }, requires_grad=True)
        assert tt1.requires_grad.all()

        tt1.a.requires_grad_(False)
        assert not tt1.requires_grad.all()
        assert tt1.requires_grad.any()

        tt1.b.x.requires_grad_(False)
        assert not tt1.requires_grad.all()
        assert not tt1.requires_grad.any()

    @choose_mark()
    def test_requires_grad_(self):
        tt1 = ttorch.tensor({
            'a': [2, 3, 4.0],
            'b': {'x': [[5, 6], [7, 8.0]]}
        })
        assert not tt1.requires_grad.any()

        tt1.requires_grad_(True)
        assert tt1.requires_grad.all()

        tt1.a.requires_grad_(False)
        assert not tt1.requires_grad.all()
        assert tt1.requires_grad.any()

        tt1.b.x.requires_grad_(False)
        assert not tt1.requires_grad.all()
        assert not tt1.requires_grad.any()

    @choose_mark()
    def test_grad(self):
        tt1 = ttorch.tensor({
            'a': [2, 3, 4.0],
            'b': {'x': [[5, 6], [7, 8.0]]}
        }, requires_grad=True)

        mq = tt1.mean() ** 2
        mq.backward()
        assert ttorch.isclose(tt1.grad, ttorch.tensor({
            'a': [1.4286, 1.4286, 1.4286],
            'b': {'x': [[1.4286, 1.4286],
                        [1.4286, 1.4286]]},
        }), atol=1e-4).all()

    @choose_mark()
    def test_detach(self):
        tt1 = ttorch.tensor({
            'a': [2, 3, 4.0],
            'b': {'x': [[5, 6], [7, 8.0]]}
        }, requires_grad=True)
        assert tt1.requires_grad.all()

        tt1r = tt1.detach()
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

        tt1r = tt1.detach_()
        assert tt1r is tt1
        assert not tt1.requires_grad.any()
