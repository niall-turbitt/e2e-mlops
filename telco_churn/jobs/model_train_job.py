import mlflow

from telco_churn.common import Job
from telco_churn.model_train import ModelTrain


class ModelTrainJob(Job):

    def _set_experiment(self):
        if "experiment_id" in self.conf['mlflow_params']:
            exp = mlflow.get_experiment(self.conf['mlflow_params']["experiment_id"])
            mlflow.set_experiment(exp.name)
        elif "experiment_path" in self.conf['experiment_path']:
            mlflow.set_experiment(self.conf['mlflow_params']["experiment_path"])
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')

    def launch(self):

        self._set_experiment()

        with mlflow.start_run(run_name=self.conf['mlflow_params']['run_name']):

            # Log config file
            mlflow.log_dict(self.conf, 'conf.yml')

            ModelTrain(mlflow_params=self.conf['mlflow_params'],
                       data_input=self.conf['data_input'],
                       pipeline_params=self.conf['pipeline_params'],
                       model_params=self.conf['model_params']).run()


if __name__ == "__main__":
    job = ModelTrainJob()
    job.launch()
