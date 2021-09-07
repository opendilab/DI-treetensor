import pytest
import torch

from treetensor.tensor import TreeTensor, zeros, all_equal, zeros_like, ones, ones_like


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestTensorFuncs:
    def test_all_equal(self):
        assert all_equal(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        )
        assert all_equal(
            TreeTensor({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            }),
            TreeTensor({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            }),
        )

    def test_zeros(self):
        assert all_equal(zeros((2, 3)), torch.zeros(2, 3))
        assert all_equal(zeros({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }), TreeTensor({
            'a': torch.zeros(2, 3),
            'b': torch.zeros(5, 6),
            'x': {
                'c': torch.zeros(2, 3, 4),
            }
        }))

    def test_zeros_like(self):
        assert all_equal(
            zeros_like(TreeTensor({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            })),
            TreeTensor({
                'a': torch.tensor([[0, 0, 0], [0, 0, 0]]),
                'b': torch.tensor([0, 0, 0, 0]),
                'x': {
                    'c': torch.tensor([0, 0, 0]),
                    'd': torch.tensor([[[0, 0]]]),
                }
            })
        )

    def test_ones(self):
        assert all_equal(ones((2, 3)), torch.ones(2, 3))
        assert all_equal(ones({
            'a': (2, 3),
            'b': (5, 6),
            'x': {
                'c': (2, 3, 4),
            }
        }), TreeTensor({
            'a': torch.ones(2, 3),
            'b': torch.ones(5, 6),
            'x': {
                'c': torch.ones(2, 3, 4),
            }
        }))

    def test_ones_like(self):
        assert all_equal(
            ones_like(TreeTensor({
                'a': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'b': torch.tensor([1, 2, 3, 4]),
                'x': {
                    'c': torch.tensor([5, 6, 7]),
                    'd': torch.tensor([[[8, 9]]]),
                }
            })),
            TreeTensor({
                'a': torch.tensor([[1, 1, 1], [1, 1, 1]]),
                'b': torch.tensor([1, 1, 1, 1]),
                'x': {
                    'c': torch.tensor([1, 1, 1]),
                    'd': torch.tensor([[[1, 1]]]),
                }
            })
        )
