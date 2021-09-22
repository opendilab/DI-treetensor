from types import ModuleType

__all__ = [
    'removed', 'class_autoremove', 'module_autoremove',
]

_REMOVED_TAG = '__removed__'


def removed(obj):
    """
    Overview:
        Add ``__removed__`` attribute to the given object.
        The given ``object`` will be marked as removed, will be removed when
        :func:`class_autoremove` or :func:`module_autoremove` is used.

    Arguments:
        - obj: Given object to be marked.

    Returns:
        - marked: Marked object.
    """
    setattr(obj, _REMOVED_TAG, True)
    return obj


def _is_removed(obj) -> bool:
    return not not getattr(obj, _REMOVED_TAG, False)


def class_autoremove(cls: type) -> type:
    """
    Overview:
        Remove the items which are marked as removed in the given ``cls``.

    Arguments:
        - cls (:obj:`type`): Given class.

    Returns:
        - marked (:obj:`type`): Marked class.

    Examples::

        >>> @class_autoremove
        >>> class MyClass:
        >>>     pass
    """
    for _name in dir(cls):
        if _is_removed(getattr(cls, _name)):
            delattr(cls, _name)
    return cls


def module_autoremove(module: ModuleType):
    """
    Overview:
        Remove the items which are marked as removed in the given ``module``.

    Arguments:
        - module (:obj:`ModuleType`): Given module.

    Returns:
        - marked (:obj:`ModuleType`): Marked module.

    Examples::

        >>> # At the imports' part
        >>> import sys
        >>>
        >>> # At the very bottom of the module
        >>> sys.modules[__name__] = module_autoremove(sys.modules[__name__])
        >>>
    """
    if hasattr(module, '__all__'):
        names = getattr(module, '__all__')

        def names_postprocess(new_names):
            _names = getattr(module, '__all__')
            _names[:] = new_names[:]
            setattr(module, '__all__', _names)
    else:
        names = dir(module)

        # noinspection PyUnusedLocal
        def names_postprocess(new_names):
            pass

    _new_names = []
    for _name in names:
        if _is_removed(getattr(module, _name)):
            delattr(module, _name)
        else:
            _new_names.append(_name)

    names_postprocess(_new_names)
    return module
