import pytest
import torch
from treevalue import TreeValue

from treetensor.tensor import TreeTensor, zeros, zeros_like, ones, ones_like, randint, randint_like, randn, \
    randn_like, full, full_like, TreeSize
from treetensor.tensor import all as _tensor_all


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestTensorFuncs:
    def test_zeros(self):
        assert _tensor_all(zeros((2, 3)) == torch.zeros(2, 3))
        assert _tensor_all(zeros(TreeValue({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })) == TreeTensor({
            'a': torch.zeros(2, 3),
            'b': torch.zeros(5, 6),
            'x': {
                'c': torch.zeros(2, 3, 4),
            }
        }))

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
        )

    def test_ones(self):
        assert _tensor_all(ones((2, 3)) == torch.ones(2, 3))
        assert _tensor_all(ones(TreeValue({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        })) == TreeTensor({
            'a': torch.ones(2, 3),
            'b': torch.ones(5, 6),
            'x': {
                'c': torch.ones(2, 3, 4),
            }
        }))

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
        )

    def test_randn(self):
        _target = randn((200, 300))
        assert -0.02 <= _target.view(60000).mean().tolist() <= 0.02
        assert 0.98 <= _target.view(60000).std().tolist() <= 1.02
        assert _target.shape == torch.Size([200, 300])

        _target = randn(TreeValue({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }))
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
        _target = randint(TreeValue({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }), -10, 10)
        assert _tensor_all(_target < 10)
        assert _tensor_all(-10 <= _target)
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([5, 6]),
            'x': {
                'c': torch.Size([2, 3, 4]),
            }
        })

        _target = randint(TreeValue({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }), 10)
        assert _tensor_all(_target < 10)
        assert _tensor_all(0 <= _target)
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
        assert _tensor_all(_target < 10)
        assert _tensor_all(-10 <= _target)
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
        assert _tensor_all(_target < 10)
        assert _tensor_all(0 <= _target)
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    def test_full(self):
        _target = full(TreeValue({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }), 233)
        assert _tensor_all(_target == 233)
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
        assert _tensor_all(_target == 233)
        assert _target.shape == TreeSize({
            'a': torch.Size([2, 3]),
            'b': torch.Size([4]),
            'x': {
                'c': torch.Size([3]),
                'd': torch.Size([1, 1, 2]),
            }
        })

    def test_all(self):
        r1 = _tensor_all(torch.tensor([1, 1, 1]) == 1)
        assert torch.is_tensor(r1)
        assert r1 == torch.tensor(True)

        r2 = _tensor_all(torch.tensor([1, 1, 2]) == 1)
        assert torch.is_tensor(r2)
        assert r2 == torch.tensor(False)

        r3 = _tensor_all(TreeTensor({
            'a': torch.Tensor([1, 2, 3]),
            'b': torch.Tensor([4, 5, 6]),
            'x': {
                'c': torch.Tensor([7, 8, 9])
            }
        }) == TreeTensor({
            'a': torch.Tensor([1, 2, 3]),
            'b': torch.Tensor([4, 5, 6]),
            'x': {
                'c': torch.Tensor([7, 8, 9])
            }
        }))
        assert torch.is_tensor(r3)
        assert r3 == torch.tensor(True)

        r4 = _tensor_all(TreeTensor({
            'a': torch.Tensor([1, 2, 3]),
            'b': torch.Tensor([4, 5, 6]),
            'x': {
                'c': torch.Tensor([7, 8, 9])
            }
        }) == TreeTensor({
            'a': torch.Tensor([1, 2, 3]),
            'b': torch.Tensor([4, 5, 6]),
            'x': {
                'c': torch.Tensor([7, 8, 8])
            }
        }))
        assert torch.is_tensor(r4)
        assert r4 == torch.tensor(False)
