import builtins
import os
from functools import partial

import treetensor.torch as torch

print = partial(builtins.print, sep=os.linesep)

if __name__ == '__main__':
    t1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
    print('new native tensor:', t1)

    t2 = torch.tensor({
        'a': [1, 2, 3],
        'b': {'x': [[4, 5], [6, 7]]},
    })
    print('new tree tensor:', t2)

    t3 = torch.randn(2, 3)
    print('new random native tensor:', t3)

    t4 = torch.randn({
        'a': (2, 3),
        'b': {'x': (3, 4)},
    })
    print('new random tree tensor:', t4)
