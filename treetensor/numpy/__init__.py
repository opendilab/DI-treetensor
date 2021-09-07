try:
    import numpy as np
except ImportError:  # numpy not exist
    from .fake import FakeTreeNumpy as TreeNumpy
else:
    from .numpy import TreeNumpy as TreeNumpy
