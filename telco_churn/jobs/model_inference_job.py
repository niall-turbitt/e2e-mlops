from telco_churn.common import Job
from telco_churn.model_inference import ModelInference


class ModelInferenceJob(Job):

    def launch(self):
        ModelInference(mlflow_params=self.conf['mlflow_params'],
                       inference_data=self.conf['data_input']['table_name'],
                       data_output=self.conf['data_output']).run_and_write_batch()


if __name__ == "__main__":
    job = ModelInferenceJob()
    job.launch()
