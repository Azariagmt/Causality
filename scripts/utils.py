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

