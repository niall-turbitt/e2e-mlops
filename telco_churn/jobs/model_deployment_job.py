import os
from typing import Dict

from telco_churn.common import Job
from telco_churn.model_deployment import ModelDeployment
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelDeploymentJob(Job):

    @staticmethod
    def _get_mlflow_params():
        return {'experiment_path': os.getenv('model_deploy_experiment_path'),
                'model_name': os.getenv('model_name')}

    @staticmethod
    def _get_reference_data() -> str:
        reference_table_database_name = os.getenv('reference_table_database_name')
        reference_table_name = os.getenv('reference_table_name')
        return f'{reference_table_database_name}.{reference_table_name}'

    @staticmethod
    def _get_reference_data_label_col():
        return os.getenv('reference_table_label_col')

    def _get_model_comparison_params(self) -> Dict:
        return self.conf['model_comparison_params']

    def launch(self):
        _logger.info('Launching ModelDeploymentJob job')
        _logger.info(f'Running model-deployment pipeline in {os.getenv("DEPLOYMENT_ENV")} environment')
        ModelDeployment(mlflow_params=self._get_mlflow_params(),
                        reference_data=self._get_reference_data(),
                        label_col=self._get_reference_data_label_col(),
                        comparison_metric=self._get_model_comparison_params()['metric'],
                        higher_is_better=self._get_model_comparison_params()['higher_is_better']).run()
        _logger.info('Launching ModelDeploymentJob job finished!')


if __name__ == '__main__':
    job = ModelDeploymentJob()
    job.launch()
