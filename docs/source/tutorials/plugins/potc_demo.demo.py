from potc import transvars

import treetensor.torch as ttorch

t_tensor = ttorch.randn({'a': (2, 3), 'b': (3, 4)})
t_i_tensor = ttorch.randint(-5, 10, {'a': (3, 4), 'x': {'b': (2, 3)}})
t_shape = t_i_tensor.shape

if __name__ == '__main__':
    _code = transvars(
        {
            't_tensor': t_tensor,
            't_i_tensor': t_i_tensor,
            't_shape': t_shape,
        },
        reformat='pep8'
    )
    print(_code)
