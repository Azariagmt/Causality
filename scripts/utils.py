import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split

def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)

def evaluate():
  predictions = model.predict(X_test)
  y_scores = model.predict_proba(X_test)
  cm = confusion_matrix(y_test, predictions)
  print ('Confusion Matrix:\n',cm, '\n')
  print('Accuracy:', accuracy_score(y_test, predictions))
  print("Overall Precision:",precision_score(y_test, predictions))
  print("Overall Recall:",recall_score(y_test, predictions))
  auc = roc_auc_score(y_test,y_scores[:,1])
  print('\nAUC: ' + str(auc))

  # calculate ROC curve
  fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

  # plot ROC curve
  fig = plt.figure(figsize=(6, 6))
  # Plot the diagonal 50% line
  plt.plot([0, 1], [0, 1], 'k--')
  # Plot the FPR and TPR achieved by our model
  plt.plot(fpr, tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.show()


def plot_ROC_curve(predictions, y_scores):
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    cm = confusion_matrix(y_test, predictions)
    print ('Confusion Matrix:\n',cm, '\n')
    print('Accuracy:', accuracy_score(y_test, predictions))
    print("Overall Precision:",precision_score(y_test, predictions))
    print("Overall Recall:",recall_score(y_test, predictions))
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('\nAUC: ' + str(auc))

    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

    # plot ROC curve
    fig = plt.figure(figsize=(6, 6))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


# from scripts.experiment import compute_jacards
import pandas as pd
# from utils import jaccard_similarity
from causality_graphing import construct_structural_model
from preprocess_data import check_numeric, label_encode

df = pd.read_csv("../data/data.csv", index_col="id")
# TODO this needs to be replaced with versions of data using DVC
df = df[['perimeter_mean', 'concavity_mean',
       'radius_worst', 'perimeter_worst', 'area_worst',
        'diagnosis']]

print("DataFrame loaded")

df, non_numeric_cols = check_numeric(df)
df = label_encode(df, non_numeric_cols)

print("DataFrame preprocessed")
def compute_jacards(df:pd.DataFrame, division:int):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        division (int): [description]

    Returns:
        [type]: [description]
    """
    print("SHAPEEE",df.shape[0])
    mods = round(df.shape[0] / division)
    print("MODS", mods)
    initial_mod = mods
    df_holder = {}
    struct_models = {}
    jaccard_sim = {}
    for num in range(division):
        df_holder[num] = df.iloc[:mods,:]
        struct_models[num] = construct_structural_model(df_holder[num], tabu_parent_nodes=["diagnosis"])
        if num > 0 :
            jaccard_sim[str(num)] = jaccard_similarity(struct_models[num-1], struct_models[num])
        else: continue
        mods = mods + initial_mod
    pass

    print("JACCAaAA", jaccard_sim)
    return jaccard_sim




# compute_jacards(df,6)
