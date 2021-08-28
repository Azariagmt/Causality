import mlflow
import dvc.api
from causality_graphing import construct_structural_model, draw_graph
from constraints import Constraints
from preprocess_data import check_numeric, label_encode
import pandas as pd
import matplotlib.pyplot as plt

from logs import log

logger = log(path="../logs/", file="causal-graph.logs")
logger.info("Starts Causal graph script")

train_store_path = 'data/data.csv'
repo = "../"

version = "'trainstorev1'"

train_store_url = dvc.api.get_url(
    path=train_store_path,
    repo=repo,
)



data = pd.read_csv("../data/data.csv")

df = data[['perimeter_mean', 'concavity_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
        'diagnosis']]

print("DataFrame loaded")
df, non_numeric_cols = check_numeric(df)
df = label_encode(df, non_numeric_cols)

print("DataFrame preprocessed")

mlflow.set_experiment('Breast cancer Causality')

if __name__ == "__main__":
    sm = construct_structural_model(df, tabu_parent_nodes=["diagnosis"])
    graph = draw_graph(sm)
    fig = plt.imread("graph.png")
    fig, ax = plt.subplots()
    mlflow.log_figure(fig,artifact_file="./graph.png")
    
    constraint = Constraints(structural_model = sm)
    constraint.add_edge("concavity_mean", "diagnosis")
    constraint.add_edge("area_worst", "diagnosis")
    constraint.add_edge("area_mean", "diagnosis")
    constraint.add_edge("concavity_worst", "diagnosis")
    constraint.add_edge("perimeter_worst", "diagnosis")
    sm_constrainted = constraint.get_model()





