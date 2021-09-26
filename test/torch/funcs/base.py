import treetensor.torch as ttorch
from treetensor.utils import replaceable_partial
from ...tests import choose_mark_with_existence_check

choose_mark = replaceable_partial(choose_mark_with_existence_check, base=ttorch)
