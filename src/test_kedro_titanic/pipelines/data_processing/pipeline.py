"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_proc_titanic

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=data_proc_titanic,
            inputs="dataset_titanic",
            outputs="preprocessed_dataset_titanic",
            name="preprocess_titanic_node",
        )
    ])
