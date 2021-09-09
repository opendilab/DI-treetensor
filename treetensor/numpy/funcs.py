from functools import partial

import numpy as np
from treevalue import func_treelize

from .numpy import TreeNumpy
from ..common import vreduce

_treelize = partial(func_treelize, return_type=TreeNumpy)

all = vreduce(all)(_treelize()(np.all))
equal = _treelize()(np.equal)
array_equal = _treelize()(np.array_equal)
