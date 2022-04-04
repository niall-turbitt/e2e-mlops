from telco_churn.common import Job
from telco_churn.model_train import ModelTrain


class ModelTrainJob(Job):

    def launch(self):
        ModelTrain(mlflow_params=self.conf['mlflow_params'],
                   data_input=self.conf['data_input'],
                   pipeline_params=self.conf['pipeline_params'],
                   model_params=self.conf['model_params'],
                   conf=self.conf).run()


if __name__ == "__main__":
    job = ModelTrainJob()
    job.launch()
