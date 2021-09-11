import importlib
import types
from typing import List

_DOC_TAG = '__doc_names__'


def _is_tagged_name(clazz, name):
    return name in set(getattr(clazz, _DOC_TAG, set()))


def _find_class_members(clazz: type) -> List[str]:
    members = []
    for name in dir(clazz):
        item = getattr(clazz, name)
        if _is_tagged_name(clazz, name) and \
                getattr(item, '__name__', None) == name:  # should be public or protected
            members.append(name)

    return members


if __name__ == '__main__':
    package_name = input().strip()
    _module = importlib.import_module(package_name)
    _alls = getattr(_module, '__all__')

    print(package_name)
    print('=' * (len(package_name) + 5))
    print()

    print(f'.. automodule:: {package_name}')
    print()

    for _name in sorted(_alls):
        print(_name)
        print('-' * (len(_name) + 5))
        print()

        _item = getattr(_module, _name)
        if isinstance(_item, types.FunctionType):
            print(f'.. autofunction:: {package_name}.{_name}')
            print()
        elif isinstance(_item, type):
            print(f'.. autoclass:: {package_name}.{_name}')
            print(f'    :members: {", ".join(sorted(_find_class_members(_item)))}')
            print()
        else:
            print(f'.. autodata:: {package_name}.{_name}')
            print(f'    :annotation:')
            print()
