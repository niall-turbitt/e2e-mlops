import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from telco_churn.common import Workload
from telco_churn.utils.logger_utils import get_logger

from databricks.feature_store.client import FeatureStoreClient

client = MlflowClient()
fs = FeatureStoreClient()
_logger = get_logger()


class DemoSetup(Workload):

    def _get_train_experiment_id(self):
        try:
            return self.env_vars['model_train_experiment_id']
        except KeyError:
            return None

    def _get_train_experiment_path(self):
        try:
            return self.env_vars['model_train_experiment_path']
        except KeyError:
            return None

    def _get_deploy_experiment_id(self):
        try:
            return self.env_vars['model_deploy_experiment_id']
        except KeyError:
            return None

    def _get_deploy_experiment_path(self):
        try:
            return self.env_vars['model_deploy_experiment_path']
        except KeyError:
            return None

    @staticmethod
    def _check_mlflow_model_registry_exists(model_name) -> bool:
        """
        Check if model exists in MLflow Model Registry.
        Returns True if model exists in Model Registry, False if not
        """
        try:
            client.get_registered_model(name=model_name)
            _logger.info(f'MLflow Model Registry name: {model_name} exists')
            return True
        except RestException:
            _logger.info(f'MLflow Model Registry name: {model_name} DOES NOT exists')
            return False

    @staticmethod
    def _archive_registered_models(model_name):
        """
        Archive any model versions which are not already under stage='Archived'
        """
        registered_model = client.get_registered_model(name=model_name)
        latest_versions_list = registered_model.latest_versions

        _logger.info(f'MLflow Model Registry name: {model_name}')
        for model_version in latest_versions_list:
            if model_version.current_stage != 'Archived':
                _logger.info(f'Archiving model version: {model_version.version}')
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage='Archived'
                )

    def _delete_registered_model(self, model_name):
        """
        Delete an experiment from the backend store.
        """
        self._archive_registered_models(model_name)
        client.delete_registered_model(name=model_name)
        _logger.info(f'Deleted MLflow Model Registry model: {model_name}')

    def _check_mlflow_experiments_exists(self) -> dict:
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
        train_experiment_id = self._get_train_experiment_id()
        train_experiment_path = self._get_train_experiment_path()
        deploy_experiment_id = self._get_deploy_experiment_id()
        deploy_experiment_path = self._get_deploy_experiment_path()

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

    def _delete_mlflow_experiments(self, exp_exists_dict: dict):
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
            if self.env_vars['model_train_experiment_path'] is not None:
                experiment = mlflow.get_experiment_by_name(name=self.env_vars['model_train_experiment_path'])
                mlflow.delete_experiment(experiment_id=experiment.experiment_id)
                _logger.info(f'Deleted existing experiment_path: {self.env_vars["model_train_experiment_path"]}')
            elif self.env_vars['model_train_experiment_id'] is not None:
                mlflow.delete_experiment(experiment_id=self.env_vars['model_train_experiment_id'])
                _logger.info(f'Deleted existing experiment_id: {self.env_vars["model_train_experiment_id"]}')
            else:
                raise RuntimeError('Either model_train_experiment_id or model_train_experiment_path should be passed '
                                   'in deployment.yml')

        if 'deploy_exp_exists' in delete_experiments:
            if self.env_vars['model_deploy_experiment_path'] is not None:
                experiment = mlflow.get_experiment_by_name(name=self.env_vars['model_deploy_experiment_path'])
                mlflow.delete_experiment(experiment_id=experiment.experiment_id)
                _logger.info(
                    f'Deleted existing experiment_path: {self.env_vars["model_deploy_experiment_path"]}')
            elif self.env_vars['model_deploy_experiment_id'] is not None:
                mlflow.delete_experiment(experiment_id=self.env_vars['model_deploy_experiment_id'])
                _logger.info(f'Deleted existing experiment_id: {self.env_vars["model_deploy_experiment_id"]}')

    @staticmethod
    def _check_feature_table_exists(feature_store_table) -> bool:
        """
        Check if Feature Store feature table exists
        Returns True if feature table exists in Feature Store, False if not
        """
        try:
            fs.get_table(name=feature_store_table)
            _logger.info(f'Feature Store feature table: {feature_store_table} exists')
            return True
        except (ValueError, Exception):
            _logger.info(f'Feature Store feature table: {feature_store_table} DOES NOT exist')
            return False

    @staticmethod
    def _drop_feature_table(feature_store_table):
        """
        Delete Feature Store feature table
        """
        try:
            fs.drop_table(
                name=feature_store_table
            )
            _logger.info(f'Deleted Feature Store feature table: {feature_store_table}')
        except ValueError:
            _logger.info(f'Feature Store feature table: {feature_store_table} does not exist')

    def _check_labels_delta_table_exists(self, labels_table_dbfs_path) -> bool:
        """
        Check if Delta table exists in DBFS

        Parameters
        ----------
        labels_table_dbfs_path : str
            Path to Delta table in DBFS

        Returns
        -------
        bool
        """
        try:
            self.dbutils.fs.ls(labels_table_dbfs_path)
            _logger.info(f'Labels Delta table: {labels_table_dbfs_path} exists')
            return True
        except:
            _logger.info(f'Labels Delta table: {labels_table_dbfs_path} DOES NOT exist')
            return False

    def _delete_labels_delta_table(self, labels_table_dbfs_path):
        self.dbutils.fs.rm(labels_table_dbfs_path, True)
        _logger.info(f'Deleted labels Delta table: {labels_table_dbfs_path}')

    def setup(self):
        """
        Demo setup steps:
        * Delete Model Registry model if exists (archive any existing models)
        * Delete MLflow experiments if exists
        * Delete Feature Table if exists
        """
        _logger.info('==========Demo Setup=========')
        _logger.info(f'Running demo-setup pipeline in {self.env_vars["env"]} environment')

        if self.conf['delete_model_registry']:
            _logger.info('Checking MLflow Model Registry...')
            model_name = self.env_vars['model_name']
            if self._check_mlflow_model_registry_exists(model_name):
                self._delete_registered_model(model_name)

        if self.conf['delete_mlflow_experiments']:
            _logger.info('Checking MLflow Tracking...')
            exp_exists_dict = self._check_mlflow_experiments_exists()
            self._delete_mlflow_experiments(exp_exists_dict)

        if self.conf['drop_feature_table']:
            _logger.info('Checking Feature Store...')
            feature_store_database_name = self.env_vars['feature_store_database_name']
            feature_store_table_name = self.env_vars['feature_store_table_name']
            feature_store_table = f'{feature_store_database_name}.{feature_store_table_name}'
            if self._check_feature_table_exists(feature_store_table=feature_store_table):
                self._drop_feature_table(feature_store_table=feature_store_table)

        if self.conf['drop_labels_table']:
            _logger.info('Checking existing labels table...')
            labels_table_dbfs_path = self.env_vars['labels_table_dbfs_path']
            if self._check_labels_delta_table_exists(labels_table_dbfs_path=labels_table_dbfs_path):
                self._delete_labels_delta_table(labels_table_dbfs_path=labels_table_dbfs_path)

        _logger.info('==========Demo Setup Complete=========')

    def launch(self) -> None:
        """
        Launch DemoSetup job
        """
        _logger.info('Launching DemoSetup job')
        DemoSetup().setup()
        _logger.info('DemoSetup job finished!')


if __name__ == '__main__':
    job = DemoSetup()
    job.launch()
