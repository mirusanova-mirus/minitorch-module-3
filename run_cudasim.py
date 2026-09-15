import os
import sys

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numba.cuda
import numpy as np

if not hasattr(numba.cuda, "is_cuda_array"):
    numba.cuda.is_cuda_array = lambda x: not isinstance(x, np.ndarray)

import pytest
from hypothesis import settings


class SmallHypothesisProfile:
    def pytest_collection_finish(self, session):
        settings.register_profile("cudasim", max_examples=3, deadline=None)
        settings.load_profile("cudasim")


sys.exit(pytest.main(["-q", "tests", "-m", "task3_3 or task3_4"] + sys.argv[1:],
                     plugins=[SmallHypothesisProfile()]))
