from telco_churn.common import Job
from telco_churn.model_inference import ModelInference


class ModelInferenceJob(Job):

    def launch(self):
        model_registry_name = self.conf['mlflow_params']['model_registry_name']
        model_registry_stage = self.conf['mlflow_params']['model_registry_stage']
        model_uri = f'models:/{model_registry_name}/{model_registry_stage}'

        ModelInference(model_uri=model_uri,
                       inference_data=self.conf['data_input']['table_name'])\
            .run_and_write_batch(delta_path=self.conf['data_output']['delta_path'],
                                 table_name=self.conf['data_output']['table_name'])


if __name__ == "__main__":
    job = ModelInferenceJob()
    job.launch()
