import random
from typing import Optional, List

import torch

_stream_pool: Optional[List[torch.cuda.Stream]] = None
_global_streams: Optional[List[torch.cuda.Stream]] = None

__all__ = [
    'stream',
]


def stream(cnt):
    assert torch.cuda.is_available(), "CUDA is not supported."

    global _stream_pool, _global_streams
    if _stream_pool is None:
        _stream_pool = [torch.cuda.current_stream()]

    if cnt is None:  # close stream support by
        _global_streams = None
    else:  # use the given number of streams
        while len(_stream_pool) < cnt:
            _stream_pool.append(torch.cuda.Stream())

        _global_streams = _stream_pool[:cnt]


def stream_call(func, *args, **kwargs):
    if _global_streams is not None:
        with torch.cuda.stream(random.choice(_global_streams)):
            return func(*args, **kwargs)
    else:
        return func(*args, **kwargs)
