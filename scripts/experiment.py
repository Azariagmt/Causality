
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
from utils import jaccard_similarity, compute_jacards
import os
from logs import log

if (not os.path.isdir('../output')):
    os.mkdir("../output")

logger = log(path="../logs/", file="causal-graph.logs")
logger.info("Starts Causal graph script")

train_store_path = 'data/data.csv'
repo = "https://github.com/Azariagmt/Causality/"
# repo for local version
# repo = "../"

version = "v1"

train_store_url = dvc.api.get_url(
    path=train_store_path,
    repo=repo,
    rev=version
)

try:
    data = pd.read_csv("../data/data.csv")
    # TODO this needs to be replaced with versions of data using DVC
    df = data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']]
    logger.info("Dataframe loaded successfully")
except RuntimeError as err:
    logger.err(err)

try:
    df, non_numeric_cols = check_numeric(df)
    df = label_encode(df, non_numeric_cols)
    logger.info("Dataframe preprocessed successfully")
except RuntimeError as err:
    logger.err(err)


mlflow.set_experiment('Breast cancer Causality')


if __name__ == "__main__":
    mlflow.log_param('train_store_data_url', train_store_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows_shape', df.shape[0])
    mlflow.log_param('input_cols_shape', df.shape[1])

    sm = construct_structural_model(df, tabu_parent_nodes=["diagnosis"])
    graph = draw_graph(sm, path="../output/gt-graph.png")
    mlflow.log_artifact("../output/gt-graph.png")

    mlflow.log_metrics(compute_jacards(df, 6))

    # jaccard_similarity()
    selected_features = ['perimeter_mean', 'radius_worst', 'area_mean', 'perimeter_worst', 'diagnosis']

    constrainted_sm = construct_structural_model(df[selected_features], tabu_parent_nodes=["diagnosis"])
    constraint = Constraints(structural_model = constrainted_sm)
    constraint.add_edge("perimeter_mean", "diagnosis")
    constraint.add_edge("radius_worst", "diagnosis")
    constraint.add_edge("area_mean", "diagnosis")
    constraint.add_edge("perimeter_worst", "diagnosis")
    sm_constrainted = constraint.get_model()
    graph = draw_graph(sm_constrainted, path="../output/constrainted-graph.png")
    mlflow.log_artifact("../output/constrainted-graph.png")

    # Modelling 
    gt_model = construct_model(df)
    selected_df = df[selected_features]
    print(selected_df.head())
    selected_model = construct_model(selected_df)
    
    mlflow.sklearn.log_model(gt_model, "Logistic Regression GT model")
    mlflow.sklearn.log_model(selected_model, "Logistic Regression Selected var model")

    prediction, y_test = evaluate(gt_model, df)
    selected_prediction, selected_y_test = evaluate(selected_model, selected_df)
    accuracy, precision, recall = metrics(y_test,prediction)
    selected_accuracy, selected_precision, selected_recall = metrics(selected_y_test, selected_prediction)
    
    mlflow.log_metrics({
        "gt_accuracy": accuracy,
        "gt_precision": precision,
        "gt_recall": recall,
        "selected_accuracy": selected_accuracy,
        "selected_precision": selected_precision,
        "selected_recall": selected_recall
    })
    try:
        with open("../output/full_data.txt", 'w') as outfile:
            outfile.write("Accuracy: " + str(accuracy) + "\n")

        with open("../output/selected_data.txt", 'w') as outfile:
            outfile.write("Accuracy: " + str(selected_accuracy) + "\n")
        
        logger.info("Accuracy report written successfully")
    except RuntimeError as err:
        logger.err("Couldnt write report to txt file")

    import matplotlib.pyplot as plt
    # Plot and save metrics details
    from sklearn.model_selection import train_test_split
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 101)

    # TODO refactor plotting confusion matrix and saving file
    plot_confusion_matrix(gt_model, X_test, y_test,
                              display_labels=['Benign', 'Malignant'],
                              cmap='magma')
    plt.title('Confusion Matrix')
    filename = f'../output/gt_confusion_matrix.png'
    plt.savefig(filename)
    # log model artifacts
    mlflow.log_artifact(filename)

    # pop diagnosis because data has already been split
    selected_features.pop(-1)
    plot_confusion_matrix(selected_model, X_test[selected_features], y_test,
                              display_labels=['Benign', 'Malignant'],
                              cmap='magma')
    plt.title('Confusion Matrix')
    filename = f'../output/selected_confusion_matrix.png'
    plt.savefig(filename)
    # log model artifacts
    mlflow.log_artifact(filename)
