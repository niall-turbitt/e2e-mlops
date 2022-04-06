from dataclasses import dataclass

from telco_churn.model_inference import ModelInference


@dataclass
class ModelDeployment:
    """
    Class to execute model deployment

    Attributes:
        model_registry_name (str): Name of model to load in MLflow Model Registry
        reference_data (str):
    """
    model_registry_name: str
    reference_data: str

    def run(self):

        stage = 'staging'

        mlflow_params = {'model_registry_name': self.model_registry_name,
                         'model_registry_stage': stage}
        model_inference = ModelInference(mlflow_params=mlflow_params,
                                         inference_data=self.reference_data)
