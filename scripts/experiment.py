import mlflow
import mlflow.sklearn
import dvc.api
from causality_graphing import construct_structural_model, draw_graph
from constraints import Constraints
from preprocess_data import check_numeric, label_encode
from model import construct_model, evaluate, metrics
from plots import plot_cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, plot_confusion_matrix, plot_roc_curve


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

# TODO this needs to be replaced with versions of data using DVC
df = data[['perimeter_mean', 'concavity_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
        'diagnosis']]

print("DataFrame loaded")
df, non_numeric_cols = check_numeric(df)
df = label_encode(df, non_numeric_cols)

print("DataFrame preprocessed")

mlflow.set_experiment('Breast cancer Causality')

if __name__ == "__main__":
    mlflow.log_param('train_store_data_url', train_store_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows_shape', df.shape[0])
    mlflow.log_param('input_cols_shape', df.shape[1])

    sm = construct_structural_model(df, tabu_parent_nodes=["diagnosis"])
    graph = draw_graph(sm)
    mlflow.log_artifact("./graph.png")
    
    constraint = Constraints(structural_model = sm)
    constraint.add_edge("concavity_mean", "diagnosis")
    constraint.add_edge("area_worst", "diagnosis")
    constraint.add_edge("area_mean", "diagnosis")
    constraint.add_edge("concavity_worst", "diagnosis")
    constraint.add_edge("perimeter_worst", "diagnosis")
    sm_constrainted = constraint.get_model()


    # Modelling 
    model = construct_model(df)
    mlflow.sklearn.log_model(model, "Logistic Regression model")
    prediction, y_test = evaluate(model, df)
    accuracy, precision, recall = metrics(y_test,prediction)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })
    df_cm = confusion_matrix(y_test, prediction)
    image = sns.heatmap(df_cm,annot=True,fmt="d")
    import matplotlib.pyplot as plt
    # Plot and save metrics details
    from sklearn.model_selection import train_test_split
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 101)

    plot_confusion_matrix(model, X_train, y_train,
                              display_labels=['Benign', 'Malignant'],
                              cmap='magma')
    plt.title('Confusion Matrix')
    filename = f'../output/confusion_matrix.png'
    plt.savefig(filename)
    # log model artifacts
    mlflow.log_artifact(filename)