"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import logging


def train_test_dataframe(df: pd.DataFrame):
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_RF_model(X, y):
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(X, y)

    return rfc



def evaluate_model_rfc(rfc: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the score, f1_score and recall

    Args:
        rfc: Random Forest model.
        X_test: Testing data of independent features.
        y_test: Testing data : Target.
    """
    y_pred = rfc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a accuracy of %.3f on test data.", score)
    logger.info("Model has a f1 score of %.3f on test data.", f_score)
    logger.info("Model has a recall of %.3f on test data.", recall)
    
    return {"accuracy": score,
            "f1_score" : f_score,
            "recall" : recall}
 