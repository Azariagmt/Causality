import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
# TODO remove seaborn and implement confusion matrix using matplotlib
import seaborn as sns


def construct_model(data:pd.DataFrame)-> LogisticRegression:
    """Constructs classification model

    Args:
        data (pd.DataFrame): DataFrame to be used to train and evaluate classification model

    Returns:
        LogisticRegression: Logistic Regression model to be used for classification
    """
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 101)

    # Define preprocessing for numeric columns (normalize them so they're on the same scale)
    numeric_features = [x for x in range(data.shape[1]-1)]
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # Create preprocessing and training pipeline
    reg = 0.01
    # Create preprocessing and training pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('logregressor', RandomForestClassifier(n_estimators=100))])
    
    # fit the pipeline to train a logistic regression model on the training set
    model = pipeline.fit(X_train, (y_train))
    return model


def evaluate(model:LogisticRegression, data:pd.DataFrame):
    """Evaluates LR model

    Args:
        model (LogisticRegression): Model to be evaluated]
        data (pd.DataFrame): Dataset used for training and evaluation

    Returns:
        predictions (Nmupy.ndarray) : Numpy array of predictions
        y_test : y value of split test set 
    """

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 101)

    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    return predictions, y_test


def metrics(y_test:np.ndarray, predictions):
    """Returns metrics

    Args:
        y_test (np.ndarray): List of true values in test set
        predictions ([type]): Values model predicted

    Returns:
        accuracy (sklearn.metrics): Gives accuracy of model
        precision (sklearn.metrics): Gives precision of model
        recall (sklearn.metrics): Gives recall of model
    """
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return accuracy, precision, recall


def plot_cm(df_cm):
    # df_cm = pd.DataFrame(array, index=["stage 1", "stage 2", "stage 3", "stagte 4"], columns=["stage 1", "stage 2", "stage 3", "stagte 4"])
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    return plt