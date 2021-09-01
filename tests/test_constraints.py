import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from constraints import Constraints
from causality_graphing import construct_structural_model

class TestConstraints(unittest.TestCase):
    def test_structural_model(self):
        constraint = Constraints()
        # self.assert()
        # TODO test constraints