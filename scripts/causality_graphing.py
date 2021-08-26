import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pylab
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

data = pd.read_csv("../data/data.csv")

data.head()

struct_data = data.copy()
 
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)

 
le = LabelEncoder()
for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])
 
struct_data.head(5)

struct_data.head()

struct_data.dtypes

for column in struct_data.columns:
  struct_data[column] = struct_data[column].fillna(0.0).astype(int)

struct_data.dtypes


from causalnex.structure.notears import from_pandas
sm = from_pandas(struct_data)

# !apt install libgraphviz-dev
# !pip install pygraphviz


viz = plot_structure(
    sm, prog="dot")
img = Image(viz.draw(format='png'))

print(img.data)

with open("graph.png", "wb") as png:
    png.write(img.data)

