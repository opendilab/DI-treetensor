import unittest

import pytest
import torch
import torch.cuda

import treetensor.torch as ttorch

_CUDA_OK = torch.cuda.is_available()

N, M, T = 200, 2, 50
S1, S2, S3 = 32, 128, 512


# noinspection DuplicatedCode
@pytest.mark.unittest
class TestTorchStream:
    def test_simple(self):
        a = ttorch.randn({f'a{i}': (S1, S2) for i in range(N)})
        b = ttorch.randn({f'a{i}': (S2, S3) for i in range(N)})

        c = ttorch.matmul(a, b)

        for i in range(N):
            assert torch.isclose(
                c[f'a{i}'], torch.matmul(a[f'a{i}'], b[f'a{i}'])
            ).all(), f'Not match on item {f"a{i}"!r}.'

    @unittest.skipUnless(_CUDA_OK, 'CUDA required')
    def test_simple_with_cuda(self):
        a = ttorch.randn({f'a{i}': (S1, S2) for i in range(N)}, device='cuda')
        b = ttorch.randn({f'a{i}': (S2, S3) for i in range(N)}, device='cuda')
        torch.cuda.synchronize()

        c = ttorch.matmul(a, b)
        torch.cuda.synchronize()

        for i in range(N):
            assert torch.isclose(
                c[f'a{i}'], torch.matmul(a[f'a{i}'], b[f'a{i}'])
            ).all(), f'Not match on item {f"a{i}"!r}.'

    @unittest.skipUnless(not _CUDA_OK, 'No CUDA required')
    def test_stream_without_cuda(self):
        with pytest.raises(AssertionError):
            ttorch.stream(10)

    @unittest.skipUnless(_CUDA_OK, 'CUDA required')
    def test_stream_with_cuda(self):
        a = ttorch.randn({f'a{i}': (S1, S2) for i in range(N)}, device='cuda')
        b = ttorch.randn({f'a{i}': (S2, S3) for i in range(N)}, device='cuda')
        ttorch.stream(4)
        torch.cuda.synchronize()

        c = ttorch.matmul(a, b)
        torch.cuda.synchronize()

        for i in range(N):
            assert torch.isclose(
                c[f'a{i}'], torch.matmul(a[f'a{i}'], b[f'a{i}'])
            ).all(), f'Not match on item {f"a{i}"!r}.'
