import os

from telco_churn.common import Job
from telco_churn.model_train import ModelTrain
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelTrainJob(Job):

    def _get_mlflow_params(self):
        return {'experiment_path': os.getenv('model_train_experiment_path'),
                'run_name': self.conf['run_name'],
                'model_registry_name': os.getenv('model_registry_name')}

    @staticmethod
    def _get_data_input():
        feature_store_database_name = os.getenv('feature_store_database_name')
        feature_store_table_name = os.getenv('feature_store_table_name')
        feature_store_params = {'table_name': f'{feature_store_database_name}.{feature_store_table_name}',
                                'primary_keys': os.getenv('feature_store_table_primary_keys')}
        labels_table_database_name = os.getenv('labels_table_database_name')
        labels_table_name = os.getenv('feature_store_table_name')
        labels_table_params = {'table_name': f'{labels_table_database_name}.{labels_table_name}'}

        return {'feature_store_params': feature_store_params,
                'labels_table_params': labels_table_params}

    def launch(self):
        _logger.info('Launching ModelTrainJob job')

        ModelTrain(mlflow_params=self._get_mlflow_params(),
                   data_input=self._get_data_input(),
                   pipeline_params=self.conf['pipeline_params'],
                   model_params=self.conf['model_params'],
                   conf=self.conf).run()
        _logger.info('ModelTrainJob job finished!')


if __name__ == "__main__":
    job = ModelTrainJob()
    job.launch()
