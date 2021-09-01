import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))
from constraints import Constraints
from causality_graphing import construct_structural_model, draw_graph
import pandas as pd
from constraints import Constraints
import preprocess_data


if (not os.path.isdir('../logs')):
    os.mkdir("../logs")

if (not os.path.isdir('../output')):
    os.mkdir("../output")



data = pd.read_csv("../data/data.csv")

df = data[['perimeter_mean', 'concavity_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
        'diagnosis']]

print("DataFrame loaded")
df, non_numeric_cols = preprocess_data.check_numeric(df)
df = preprocess_data.label_encode(df, non_numeric_cols)

print("DataFrame preprocessed")

class TestCausalGraphing(unittest.TestCase):
    def test_causal_graphing(self):
        sm = construct_structural_model(df, tabu_parent_nodes=["diagnosis"])
        draw_graph(sm, path="../output/causalnex.png")
        self.assertTrue(os.path.exists("../output/causalnex.png"))


if __name__ == "__main__":
    unittest.main()