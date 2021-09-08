import pytest
import torch

from treetensor.tensor import TreeTensor, zeros, zeros_like, ones, ones_like, randint, randint_like, randn, \
    randn_like, full, full_like, TreeSize
from treetensor.tensor import all as _tensor_all


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestTensorFuncs:
    def test_zeros(self):
        assert _tensor_all(zeros((2, 3)) == torch.zeros(2, 3)).all()
        assert _tensor_all(zeros({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }) == TreeTensor({
            'a': torch.zeros(2, 3),
            'b': torch.zeros(5, 6),
            'x': {
                'c': torch.zeros(2, 3, 4),
            }
        })).all()

    def test_zeros_like(self):
        assert _tensor_all(
            zeros_like(torch.tensor([[1, 2, 3], [4, 5, 6]])) ==
            torch.tensor([[0, 0, 0], [0, 0, 0]]),
        )
        assert _tensor_all(
            zeros_like(TreeTensor({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            })) == TreeTensor({
                'a': torch.tensor([[0, 0, 0], [0, 0, 0]]),
                'b': torch.tensor([0, 0, 0, 0]),
                'x': {
                    'c': torch.tensor([0, 0, 0]),
                    'd': torch.tensor([[[0, 0]]]),
                }
            })
        ).all()

    def test_ones(self):
        assert _tensor_all(ones((2, 3)) == torch.ones(2, 3))
        assert _tensor_all(ones({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }) == TreeTensor({
            'a': torch.ones(2, 3),
            'b': torch.ones(5, 6),
            'x': {
                'c': torch.ones(2, 3, 4),
            }
        })).all()

    def test_ones_like(self):
        assert _tensor_all(
            ones_like(torch.tensor([[1, 2, 3], [4, 5, 6]])) ==
            torch.tensor([[1, 1, 1], [1, 1, 1]])
        )
        assert _tensor_all(
            ones_like(TreeTensor({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            })) == TreeTensor({
                'a': torch.tensor([[1, 1, 1], [1, 1, 1]]),
                'b': torch.tensor([1, 1, 1, 1]),
                'x': {
                    'c': torch.tensor([1, 1, 1]),
                    'd': torch.tensor([[[1, 1]]]),
                }
            })
        ).all()

    def test_randn(self):
        _target = randn((200, 300))
        assert -0.02 <= _target.view(60000).mean().tolist() <= 0.02
        assert 0.98 <= _target.view(60000).std().tolist() <= 1.02
        assert _target.shape == torch.Size([200, 300])

        _target = randn({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    def test_randn_like(self):
        _target = randn_like(torch.ones(200, 300))
        assert -0.02 <= _target.view(60000).mean().tolist() <= 0.02
        assert 0.98 <= _target.view(60000).std().tolist() <= 1.02
        assert _target.shape == torch.Size([200, 300])

        _target = randn_like(TreeTensor({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
            'b': torch.tensor([1, 2, 3, 4], dtype=torch.float32),
            'x': {
                'c': torch.tensor([5, 6, 7], dtype=torch.float32),
                'd': torch.tensor([[[8, 9]]], dtype=torch.float32),
            }
        }))
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    def test_randint(self):
        _target = randint({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }, -10, 10)
        assert _tensor_all(_target < 10).all()
        assert _tensor_all(-10 <= _target).all()
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

        _target = randint({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }, 10)
        assert _tensor_all(_target < 10).all()
        assert _tensor_all(0 <= _target).all()
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    def test_randint_like(self):
        _target = randint_like(TreeTensor({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        }), -10, 10)
        assert _tensor_all(_target < 10).all()
        assert _tensor_all(-10 <= _target).all()
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

        _target = randint_like(TreeTensor({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        }), 10)
        assert _tensor_all(_target < 10).all()
        assert _tensor_all(0 <= _target).all()
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    def test_full(self):
        _target = full({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }, 233)
        assert _tensor_all(_target == 233).all()
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

    def test_full_like(self):
        _target = full_like(TreeTensor({
            'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'b': torch.tensor([1, 2, 3, 4]),
            'x': {
                'c': torch.tensor([5, 6, 7]),
                'd': torch.tensor([[[8, 9]]]),
            }
        }), 233)
        assert _tensor_all(_target == 233).all()
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })
