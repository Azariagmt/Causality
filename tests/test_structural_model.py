import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from constraints import Constraints
from causality_graphing import construct_structural_model
import pandas as pd
import preprocess_data
from causalnex.structure import notears
from causalnex.structure.structuremodel import StructureModel

data = pd.read_csv("../data/data.csv")

df = data[['perimeter_mean', 'concavity_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
        'diagnosis']]

df, non_numeric_cols = preprocess_data.check_numeric(df)
df = preprocess_data.label_encode(df, non_numeric_cols)


sm = construct_structural_model(df, tabu_parent_nodes=["diagnosis"])


class TestConstraints(unittest.TestCase):
    def test_structural_model(self):
        self.assertTrue(sm,StructureModel)


if __name__ == '__main__':
	unittest.main()