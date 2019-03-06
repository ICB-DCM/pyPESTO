import unittest
import numpy as np
from theano.tests import unittest_tools as tutt

from .test_sbml_conversion import _load_model_objective
from pypesto.sample.sample import PosteriorTheano


class PosteriorTheanoTest(unittest.TestCase):
    def runTest(self):
        for example in ['conversion_reaction']:
            objective = _load_model_objective(example)
            posterior = PosteriorTheano(objective)
            tutt.verify_grad(self.op,
                             [np.random((2, 5))])
