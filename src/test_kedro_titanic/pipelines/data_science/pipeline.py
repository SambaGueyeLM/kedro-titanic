"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_test_dataframe, train_RF_model, evaluate_model_rfc

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=train_test_dataframe,
                inputs="preprocessed_dataset_titanic",
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
                name="Data_spliting_node",
            ),
            node(
                func=train_RF_model,
                inputs=['X_train', 'y_train'],
                outputs='rfc',
                name="train_RF_model_node",
            ),
            node(
                func=evaluate_model_rfc,
                inputs=['rfc', 'X_test', 'y_test'],
                outputs='rfc_metrics',
                name="evaluation_RF_model_node",
            )
        ]
    )

