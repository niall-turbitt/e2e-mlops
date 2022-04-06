from telco_churn.common import Job
from telco_churn.model_deployment import ModelDeployment


class ModelDeploymentJob(Job):

    def launch(self):
        pass


if __name__ == "__main__":
    job = ModelDeploymentJob()
    job.launch()
