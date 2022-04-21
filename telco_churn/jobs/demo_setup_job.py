import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from telco_churn.common import Job
from telco_churn.utils.logger_utils import get_logger

client = MlflowClient()
_logger = get_logger()


class DemoSetup(Job):

    def _check_mlflow_model_registry_exists(self):
        pass

    def _delete_model_registry(self):
        client.delete_registered_model(name=self.conf['mlflow_params']['model_registry_name'])

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
        def check_by_experiment_id(experiment_id):
            try:
                mlflow.get_experiment(experiment_id=experiment_id)
                return True
            except RestException:
                return False

        def check_by_experiment_path(experiment_path):
            experiment = mlflow.get_experiment_by_name(name=experiment_path)
            if experiment is not None:
                return True
            else:
                return False

        mlflow_params = self.conf['mlflow_params']

        if 'train_experiment_id' in mlflow_params:
            train_exp_exists = check_by_experiment_id(mlflow_params['train_experiment_id'])
        elif 'train_experiment_path' in mlflow_params:
            train_exp_exists = check_by_experiment_path(mlflow_params['train_experiment_path'])
        else:
            raise RuntimeError('Either train_experiment_id or train_experiment_path should be passed in demo_setup.yml')

        if 'deploy_experiment_id' in mlflow_params:
            deploy_exp_exists = check_by_experiment_id(mlflow_params['deploy_experiment_id'])
        elif 'deploy_experiment_path' in mlflow_params:
            deploy_exp_exists = check_by_experiment_path(mlflow_params['deploy_experiment_path'])
        else:
            raise RuntimeError('Either train_experiment_id or train_experiment_path should be passed in demo_setup.yml')

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
            try:
                mlflow.delete_experiment(experiment_id=self.conf['mlflow_params']['train_experiment_id'])
                _logger.info(f'Deleted existing experiment_id: {self.conf["mlflow_params"]["train_experiment_id"]}')
            except KeyError:
                experiment = mlflow.get_experiment_by_name(name=self.conf['mlflow_params']['train_experiment_path'])
                mlflow.delete_experiment(experiment_id=experiment.experiment_id)
                _logger.info(f'Deleted existing experiment_path: {self.conf["mlflow_params"]["train_experiment_path"]}')
        if 'deploy_exp_exists' in delete_experiments:
            try:
                mlflow.delete_experiment(experiment_id=self.conf['mlflow_params']['deploy_experiment_id'])
                _logger.info(f'Deleted existing experiment_id: {self.conf["mlflow_params"]["deploy_experiment_id"]}')
            except KeyError:
                experiment = mlflow.get_experiment_by_name(name=self.conf['mlflow_params']['deploy_experiment_path'])
                mlflow.delete_experiment(experiment_id=experiment.experiment_id)
                _logger.info(f'Deleted existing experiment_path: {self.conf["mlflow_params"]["deploy_experiment_path"]}')

    def _check_feature_table_exists(self):
        pass

    def _archive_all_models(self):
        pass



    def _delete_feature_table(self):
        pass



    def setup(self):

        if self._check_mlflow_model_registry_exists():
            self._archive_all_models()
            self._delete_model_registry()



        exp_exists_dict = self._check_mlflow_experiments_exists()







        if self._check_feature_table_exists():
            self._delete_feature_table()

    def launch(self) -> None:
        """
        Launch DemoSetupJob
        """
        _logger.info("Launching DemoSetup job")
        DemoSetup().setup()
        _logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = DemoSetup()
    job.launch()
