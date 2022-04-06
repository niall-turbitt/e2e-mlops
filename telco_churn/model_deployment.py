from dataclasses import dataclass

from telco_churn.model_inference import ModelInference


@dataclass
class ModelDeployment:
    """
    Class to execute model deployment

    Attributes:
        model_registry_name (str):
        data_prep_params (dict):
        feature_store_params (dict):
        labels_table_params (dict):
    """
    model_registry_name: str
    reference_data: str

    def run(self):
        stage = 'staging'

        mlflow_params = {'model_registry_name': self.model_registry_name,
                         'model_registry_stage': stage}
        model_inference = ModelInference(mlflow_params=mlflow_params,
                                         data_input=self.reference_data)
