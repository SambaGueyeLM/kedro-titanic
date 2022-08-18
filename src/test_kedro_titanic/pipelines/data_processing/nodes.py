"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def data_proc_titanic(dataset_titanic: pd.DataFrame)->pd.DataFrame:

    dataset_titanic.drop(columns=['Cabin', 'Name', 'PassengerId', 'Ticket' ], inplace=True)
    dataset_titanic['Embarked'].replace({'S':1, 'C':2, 'Q':3}, inplace=True)
    dataset_titanic['Sex'].replace({'male':1, 'female':2}, inplace=True)
    dataset_titanic.fillna(dataset_titanic.mean(), inplace=True)

    return dataset_titanic

