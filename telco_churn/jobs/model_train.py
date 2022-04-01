"""placeholder for model train job"""
from telco_churn.common import Job


class ModelTrain(Job):

    def launch(self):
        pass


if __name__ == "__main__":
    job = ModelTrain()
    job.launch()
