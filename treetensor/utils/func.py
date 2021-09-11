__all__ = [
    'replaceable_partial',
]


def replaceable_partial(func, **kws):
    def _new_func(*args, **kwargs):
        return func(*args, **{**kws, **kwargs})

    return _new_func
