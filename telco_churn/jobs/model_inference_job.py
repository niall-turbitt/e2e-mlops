import os
from typing import Dict

from telco_churn.common import Job
from telco_churn.model_inference import ModelInference
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelInferenceJob(Job):

    def _get_model_uri(self) -> str:
        model_name = os.getenv('model_name')
        model_registry_stage = self.conf['mlflow_params']['model_registry_stage']
        model_uri = f'models:/{model_name}/{model_registry_stage}'

        return model_uri

    @staticmethod
    def _get_inference_data() -> str:
        """
        Get the name of the table to perform inference on
        """
        inference_database_name = os.getenv('inference_database_name')
        inference_table_name = os.getenv('inference_table_name')
        return f'{inference_database_name}.{inference_table_name}'

    def _get_predictions_output_params(self) -> Dict:
        """
        Get a dictionary of delta_path, table_name, mode key-values to pass to run_and_write_batch of ModelInference
        """
        predictions_table_database_name = os.getenv('predictions_table_database_name')
        predictions_table_name = os.getenv('predictions_table_name')
        output_table_name = f'{predictions_table_database_name}.{predictions_table_name}'

        return {'delta_path': os.getenv('predictions_table_dbfs_path'),
                'table_name': output_table_name,
                'mode': self.conf['data_output']['mode']}

    def launch(self):
        _logger.info('Launching Batch ModelInferenceJob job')
        _logger.info(f'Running model-inference-batch in {os.getenv("DEPLOYMENT_ENV")} environment')
        ModelInference(model_uri=self._get_model_uri(),
                       inference_data=self._get_inference_data())\
            .run_and_write_batch(**self._get_predictions_output_params())
        _logger.info('Batch ModelInferenceJob job finished')


if __name__ == "__main__":
    job = ModelInferenceJob()
    job.launch()
