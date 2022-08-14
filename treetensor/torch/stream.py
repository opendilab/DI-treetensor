import itertools
from typing import Optional, List

import torch

_stream_pool: Optional[List[torch.cuda.Stream]] = None
_global_streams: Optional[List[torch.cuda.Stream]] = None

__all__ = [
    'stream', 'stream_call',
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


_stream_count = itertools.count()


def stream_call(func, *args, **kwargs):
    if _global_streams is not None:
        _stream_index = next(_stream_count) % len(_global_streams)
        _stream = _global_streams[_stream_index]
        with torch.cuda.stream(_stream):
            return func(*args, **kwargs)
    else:
        return func(*args, **kwargs)
