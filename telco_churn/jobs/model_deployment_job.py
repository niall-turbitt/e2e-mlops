from telco_churn.common import Workload
from telco_churn.model_deployment import ModelDeployment
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelDeploymentJob(Workload):

    def _get_mlflow_params(self):
        return {'experiment_path': self.env_vars['model_deploy_experiment_path'],
                'model_name': self.env_vars['model_name']}

    def _get_reference_data(self) -> str:
        reference_table_database_name = self.env_vars['reference_table_database_name']
        reference_table_name = self.env_vars['reference_table_name']
        return f'{reference_table_database_name}.{reference_table_name}'

    def _get_reference_data_label_col(self) -> str:
        return self.env_vars['reference_table_label_col']

    def _get_model_comparison_params(self) -> dict:
        return self.conf['model_comparison_params']

    def launch(self):
        _logger.info('Launching ModelDeploymentJob job')
        _logger.info(f'Running model-deployment pipeline in {self.env_vars["DEPLOYMENT_ENV"]} environment')
        ModelDeployment(mlflow_params=self._get_mlflow_params(),
                        reference_data=self._get_reference_data(),
                        label_col=self._get_reference_data_label_col(),
                        comparison_metric=self._get_model_comparison_params()['metric'],
                        higher_is_better=self._get_model_comparison_params()['higher_is_better']).run()
        _logger.info('Launching ModelDeploymentJob job finished!')


if __name__ == '__main__':
    job = ModelDeploymentJob()
    job.launch()
