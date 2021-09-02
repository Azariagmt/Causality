from networkx.algorithms.structuralholes import constraint
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure import notears
from causalnex.structure.notears import from_pandas, from_pandas_lasso
from constraints import Constraints
from logs import log
import preprocess_data
import os, sys


if (not os.path.isdir('../logs')):
    os.mkdir("../logs")

if (not os.path.isdir('../output')):
    os.mkdir("../output")

logs_path = "../logs/causal-graph.logs"
if not os.path.exists(logs_path):
    with open(logs_path, "w"):
        global logger
        logger = log(path="../logs/", file="causal-graph.logs")
        logger.info("Starts Causal graph script")
else:
    logger = log(path="../logs/", file="causal-graph.logs")
    logger.info("Starts Causal graph script")




data = pd.read_csv("../data/data.csv")

df = data[['perimeter_mean', 'concavity_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
        'diagnosis']]

print("DataFrame loaded")
df, non_numeric_cols = preprocess_data.check_numeric(df)
df = preprocess_data.label_encode(df, non_numeric_cols)

print("DataFrame preprocessed")

def construct_structural_model(df:pd.DataFrame, notears=from_pandas_lasso, tabu_parent_nodes=None)-> notears:
    """Constructs structural model to be used to draw causal graph

    Args:
        df (pd.DataFrame): Preprocessed DataFrame that will construct structural model
        notears ([type], optional): [description]. Defaults to from_pandas_lasso.
        tabu_parent_nodes (list) : List of features to not be the causes

    Returns:
        notears: structural model to draw graph
    """
    structural_model = notears(df, beta=0.8, tabu_parent_nodes=tabu_parent_nodes)
    return structural_model


def draw_graph(structural_model: from_pandas_lasso, path, prog="dot"):
    """Draws Causal graph

    Args:
        structural_model (from_pandas_lasso): Structural model of causalnex
        prog (str, optional): Graphics tool to draw pygraphiz graph. Defaults to "dot".

    Returns:
        image (png) : Causal graph img
    """
    viz = plot_structure(
    structural_model, prog="dot",
    graph_attributes={"scale": "2", "size": "2.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
    img = Image(viz.draw(format='png'))

    # TODO convert print log to use logger
    print("writing graph image")
    with open(f"{path}", "wb") as png:
        png.write(img.data)

    return img

if __name__ == "__main__":
    sm = construct_structural_model(df, tabu_parent_nodes=["diagnosis"])
    draw_graph(sm, path="../output/causalnex.png")
    constraint = Constraints(structural_model = sm)
    constraint.add_edge("concavity_mean", "diagnosis")
    constraint.add_edge("area_mean", "diagnosis")
    constraint.add_edge("perimeter_worst", "diagnosis")
    sm_constrainted = constraint.get_model()
    draw_graph(sm_constrainted, path="../output/constrainted-two.png")




    

# important_features = ["area_mean", "concavity_mean", "concavity_worst","area_se","texture_mean", "symmetry_worst", "diagnosis"]
# struct_data = data[important_features].copy()
 
# non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
# print(non_numeric_columns)

 
# le = LabelEncoder()
# for col in non_numeric_columns:
#     struct_data[col] = le.fit_transform(struct_data[col])
 
# struct_data.head(5)

# struct_data.head()

# struct_data.dtypes

# for column in struct_data.columns:
#   struct_data[column] = struct_data[column].fillna(0.0).astype(int)

# struct_data.dtypes



# sm = from_pandas(struct_data)

# # !apt install libgraphviz-dev
# # !pip install pygraphviz


# viz = plot_structure(
#     sm, prog="dot",
#     graph_attributes={"scale": "0.5"},
#     all_node_attributes=NODE_STYLE.WEAK,
#     all_edge_attributes=EDGE_STYLE.WEAK)
# img = Image(viz.draw(format='png'))

# print(len(img.data))


