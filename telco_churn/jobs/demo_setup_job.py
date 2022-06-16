import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from telco_churn.common import Job
from telco_churn.utils.logger_utils import get_logger

from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.utils.request_context import RequestContext

client = MlflowClient()
fs = FeatureStoreClient()
_logger = get_logger()


class DemoSetup(Job):

    @staticmethod
    def _check_mlflow_model_registry_exists(model_registry_name) -> bool:
        """
        Check if model exists in MLflow Model Registry.
        Returns True if model exists in Model Registry, False if not
        """
        try:
            client.get_registered_model(name=model_registry_name)
            _logger.info(f'MLflow Model Registry name: {model_registry_name} exists')
            return True
        except RestException:
            _logger.info(f'MLflow Model Registry name: {model_registry_name} DOES NOT exists')
            return False

    @staticmethod
    def _archive_registered_models(model_registry_name):
        """
        Archive any model versions which are not already under stage='Archived
        """
        registered_model = client.get_registered_model(name=model_registry_name)
        latest_versions_list = registered_model.latest_versions

        _logger.info(f'MLflow Model Registry name: {model_registry_name}')
        for model_version in latest_versions_list:
            if model_version.current_stage != 'Archived':
                _logger.info(f'Archiving model version: {model_version.version}')
                client.transition_model_version_stage(
                    name=model_registry_name,
                    version=model_version.version,
                    stage='Archived'
                )

    def _delete_registered_model(self, model_registry_name):
        """
        Delete an experiment from the backend store.
        """
        self._archive_registered_models(model_registry_name)
        client.delete_registered_model(name=model_registry_name)
        _logger.info(f'Deleted MLflow Model Registry model: {model_registry_name}')

    @staticmethod
    def _check_mlflow_experiments_exists() -> dict:
        """
        The demo workflow consists of creating 2 MLflow Tracking experiments:
            * train_experiment - Experiment used to track params, metrics, artifacts during model training
            * deploy_experiment - Experiment used to metrics when comparing models during the deploy model step

        This method checks the demo_setup config dict for either the experiment_id or experiment_path for both
        experiments.

        A dictionary containing the keys train_exp_exists and deploy_exp_exists along with boolean values is returned

        Returns
        -------
        Dictionary indicating whether train and deploy MLflow experiments currently exist
        """
        train_experiment_id = os.getenv('model_train_experiment_id')    # will be None if not passed
        train_experiment_path = os.getenv('model_train_experiment_path')
        deploy_experiment_id = os.getenv('model_deploy_experiment_id')
        deploy_experiment_path = os.getenv('model_deploy_experiment_path')

        def check_by_experiment_id(experiment_id):
            try:
                mlflow.get_experiment(experiment_id=experiment_id)
                _logger.info(f'MLflow Tracking experiment_id: {experiment_id} exists')
                return True
            except RestException:
                _logger.info(f'MLflow Tracking experiment_id: {experiment_id} DOES NOT exist')
                return False

        def check_by_experiment_path(experiment_path):
            experiment = mlflow.get_experiment_by_name(name=experiment_path)
            if experiment is not None:
                _logger.info(f'MLflow Tracking experiment_path: {experiment_path} exists')
                return True
            else:
                _logger.info(f'MLflow Tracking experiment_path: {experiment_path} DOES NOT exist')
                return False

        if train_experiment_id is not None:
            train_exp_exists = check_by_experiment_id(train_experiment_id)
        elif train_experiment_path is not None:
            train_exp_exists = check_by_experiment_path(train_experiment_path)
        else:
            raise RuntimeError('Either model_train_experiment_id or model_train_experiment_path should be passed in '
                               'deployment.yml')

        if deploy_experiment_id is not None:
            deploy_exp_exists = check_by_experiment_id(deploy_experiment_id)
        elif deploy_experiment_path is not None:
            deploy_exp_exists = check_by_experiment_path(deploy_experiment_path)
        else:
            raise RuntimeError('Either model_train_experiment_id or model_train_experiment_path should be passed in '
                               'deployment.yml')

        return {'train_exp_exists': train_exp_exists,
                'deploy_exp_exists': deploy_exp_exists}

    @staticmethod
    def _delete_mlflow_experiments(exp_exists_dict: dict):
        """
        Check exp_exists_dict if train_exp_exists or deploy_exp_exists is True. Delete experiments if they exist

        Parameters
        ----------
        exp_exists_dict : dict
            A dictionary containing the keys train_exp_exists and deploy_exp_exists along with boolean values
        """
        delete_experiments = [exp for exp, exists in exp_exists_dict.items() if exists == True]
        if len(delete_experiments) == 0:
            _logger.info(f'No existing experiments to delete')
        if 'train_exp_exists' in delete_experiments:
            if os.getenv('model_train_experiment_path') is not None:
                experiment = mlflow.get_experiment_by_name(name=os.getenv('model_train_experiment_path'))
                mlflow.delete_experiment(experiment_id=experiment.experiment_id)
                _logger.info(f'Deleted existing experiment_path: {os.getenv("model_train_experiment_path")}')
            elif os.getenv('model_train_experiment_id') is not None:
                mlflow.delete_experiment(experiment_id=os.getenv('model_train_experiment_id'))
                _logger.info(f'Deleted existing experiment_id: {os.getenv("model_train_experiment_id")}')
            else:
                raise RuntimeError('Either model_train_experiment_id or model_train_experiment_path should be passed '
                                   'in deployment.yml')

        if 'deploy_exp_exists' in delete_experiments:
            if os.getenv('model_deploy_experiment_path') is not None:
                experiment = mlflow.get_experiment_by_name(name=os.getenv('model_deploy_experiment_path'))
                mlflow.delete_experiment(experiment_id=experiment.experiment_id)
                _logger.info(
                    f'Deleted existing experiment_path: {os.getenv("model_deploy_experiment_path")}')
            elif os.getenv('model_deploy_experiment_id') is not None:
                mlflow.delete_experiment(experiment_id=os.getenv('model_deploy_experiment_id'))
                _logger.info(f'Deleted existing experiment_id: {os.getenv("model_deploy_experiment_id")}')

    def _check_feature_table_exists(self) -> bool:
        """
        Check if Feature Store feature table exists
        Returns True if feature table exists in Feature Store, False if not
        """
        try:
            fs.get_table(name=self.conf['feature_store_table'])
            _logger.info(f'Feature Store feature table: {self.conf["feature_store_table"]} exists')
            return True
        except (ValueError, Exception):
            _logger.info(f'Feature Store feature table: {self.conf["feature_store_table"]} DOES NOT exist')
            return False

    def _delete_feature_table(self):
        """
        Delete Feature Store feature table
        """
        try:
            fs.drop_table(
                name=self.conf['feature_store_table']
            )
            _logger.info(f'Deleted Feature Store feature table: {self.conf["feature_store_table"]}')
        except ValueError:
            _logger.info(f'Feature Store feature table: {self.conf["feature_store_table"]} does not exist')

    def _check_labels_delta_table_exists(self) -> bool:
        try:
            _logger.info(f'Labels Delta table: {self.conf["labels_table_path"]} exists')
            self.dbutils.fs.ls(self.conf['labels_table_path'])
            return True
        except:
            _logger.info(f'Labels Delta table: {self.conf["labels_table_path"]} does not exist')
            return False

    def _delete_labels_delta_table(self):
        self.dbutils.fs.rm(self.conf['labels_table_path'], True)
        _logger.info(f'Deleted labels Delta table: {self.conf["labels_table_path"]}')

    def setup(self):
        """
        Demo setup steps:
        * Delete Model Registry model if exists (archive any existing models)
        * Delete MLflow experiments if exists
        * Delete Feature Table if exists
        """
        _logger.info('==========Demo Setup=========')

        _logger.info('Checking MLflow Model Registry...')
        model_registry_name = os.getenv('model_registry_name')
        if self._check_mlflow_model_registry_exists(model_registry_name):
            self._delete_registered_model(model_registry_name)

        _logger.info('Checking MLflow Tracking...')
        exp_exists_dict = self._check_mlflow_experiments_exists()
        self._delete_mlflow_experiments(exp_exists_dict)

        _logger.info('Checking Feature Store...')
        if self._check_feature_table_exists():
            self._delete_feature_table()

        _logger.info('Checking existing labels table...')
        if self._check_labels_delta_table_exists():
            self._delete_labels_delta_table()

        _logger.info('==========Demo Setup Complete=========')

    def launch(self) -> None:
        """
        Launch DemoSetup job
        """
        _logger.info('Launching DemoSetup job')
        DemoSetup().setup()
        _logger.info('DemoSetup job finished!')

if __name__ == "__main__":
    job = DemoSetup()
    job.launch()
