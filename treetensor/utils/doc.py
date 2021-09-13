__all__ = [
    'doc_from',
]

_DOC_FROM_TAG = '__doc_from__'


def doc_from(src):
    def _decorator(obj):
        setattr(obj, _DOC_FROM_TAG, src)
        return obj

    return _decorator
