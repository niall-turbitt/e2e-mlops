import os

from telco_churn.common import Workload
from telco_churn.model_train import ModelTrain
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelTrainJob(Workload):

    def _get_mlflow_params(self):
        return {'experiment_path': os.getenv('model_train_experiment_path'),
                'run_name': self.conf['mlflow_params']['run_name'],
                'model_name': os.getenv('model_name')}

    @staticmethod
    def _get_data_input():
        feature_store_database_name = os.getenv('feature_store_database_name')
        feature_store_table_name = os.getenv('feature_store_table_name')
        feature_store_params = {'table_name': f'{feature_store_database_name}.{feature_store_table_name}',
                                'primary_keys': os.getenv('feature_store_table_primary_keys')}
        labels_table_database_name = os.getenv('labels_table_database_name')
        labels_table_name = os.getenv('labels_table_name')
        labels_table_params = {'table_name': f'{labels_table_database_name}.{labels_table_name}'}

        return {'feature_store_params': feature_store_params,
                'labels_table_params': labels_table_params}

    def _get_pipeline_params(self):
        return self.conf['pipeline_params']

    def _get_model_params(self):
        return self.conf['model_params']

    def launch(self):
        _logger.info('Launching ModelTrainJob job')
        _logger.info(f'Running model-train pipeline in {os.getenv("DEPLOYMENT_ENV")} environment')
        ModelTrain(mlflow_params=self._get_mlflow_params(),
                   data_input=self._get_data_input(),
                   pipeline_params=self._get_pipeline_params(),
                   model_params=self._get_model_params(),
                   conf=self.conf).run()
        _logger.info('ModelTrainJob job finished!')


if __name__ == '__main__':
    job = ModelTrainJob()
    job.launch()
