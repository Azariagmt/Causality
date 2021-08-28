import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from constraints import Constraints

class TestConstraints(unittest.TestCase):
    def test_structural_model():
        constraint = Constraints()
        assert()