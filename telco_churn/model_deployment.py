from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from telco_churn.model_inference import ModelInference
from telco_churn.utils.evaluation_utils import ModelEvaluation
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class ModelDeployment:
    """
    Class to execute model deployment

    Attributes:
        model_registry_name (str): Name of model to load in MLflow Model Registry
        reference_data (str): Name of table to use as a reference DataFrame to score loaded model on
            Must contain column(s) for lookup keys to join feature data from Feature Store
    """
    model_registry_name: str
    reference_data: str
    label_col: str
    comparison_metric: str
    higher_is_better: bool = True
    experiment_id: int = None
    experiment_path: int = None

    def _set_experiment(self):
        """
        Set MLflow experiment. Use one of either experiment_id or experiment_path
        """
        if self.experiment_id is not None:
            mlflow.set_experiment(experiment_id=self.experiment_id)
        elif self.experiment_path is not None:
            mlflow.set_experiment(experiment_name=self.experiment_path)
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')

    def _get_model_uri_by_stage(self, stage: str):
        """

        Parameters
        ----------
        stage

        Returns
        -------

        """
        return f'models:/{self.model_registry_name}/{stage}'

    def _batch_inference_by_stage(self, stage: str):
        """

        Parameters
        ----------
        stage

        Returns
        -------

        """
        model_uri = self._get_model_uri_by_stage(stage=stage)
        _logger.info(f'Computing batch inference using: {model_uri}')
        _logger.info(f'Reference data: {self.reference_data}')
        model_inference = ModelInference(model_uri=model_uri,
                                         inference_data=self.reference_data)

        return model_inference.run_batch()

    @staticmethod
    def _get_evaluation_metric(y_true: pd.Series, y_score: pd.Series, metric: str, stage: str) -> float:
        """

        Parameters
        ----------
        y_true
        y_score
        metric
        stage

        Returns
        -------

        """
        metric_prefix = stage + "_"
        eval_dict = ModelEvaluation().evaluate(y_true, y_score, metric_prefix=metric_prefix)
        mlflow.log_metrics(eval_dict)
        eval_metric = eval_dict[metric_prefix + metric]

        return eval_metric

    def _run_promotion_logic(self, staging_eval_metric: float, production_eval_metric: float):
        """

        Parameters
        ----------
        staging_eval_metric
        production_eval_metric

        Returns
        -------

        """
        client = MlflowClient()
        staging_model_version = client.get_latest_versions(name=self.model_registry_name, stages=['staging'])[0]

        _logger.info(f'metric={self.comparison_metric}')
        _logger.info(f'higher_is_better={self.higher_is_better}')
        if self.higher_is_better:
            if staging_eval_metric <= production_eval_metric:
                _logger.info('Candidate Staging model DOES NOT perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="archived"')
                client.transition_model_version_stage(name=self.model_registry_name,
                                                      version=staging_model_version.version,
                                                      stage="archived")

            elif staging_eval_metric > production_eval_metric:
                _logger.info('Candidate Staging model DOES perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="production"')
                _logger.info('Existing Production model will be archived')
                client.transition_model_version_stage(name=self.model_registry_name,
                                                      version=staging_model_version.version,
                                                      stage="production",
                                                      archive_existing_versions=True)

        else:
            if staging_eval_metric >= production_eval_metric:
                _logger.info('Candidate Staging model DOES NOT perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="archived"')
                client.transition_model_version_stage(name=self.model_registry_name,
                                                      version=staging_model_version.version,
                                                      stage="archived")

            elif staging_eval_metric < production_eval_metric:
                _logger.info('Candidate Staging model DOES perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="production"')
                _logger.info('Existing Production model will be archived')
                client.transition_model_version_stage(name=self.model_registry_name,
                                                      version=staging_model_version.version,
                                                      stage="production",
                                                      archive_existing_versions=True)

    def run(self):
        """

        Returns
        -------

        """
        self._set_experiment()
        with mlflow.start_run(run_name='Model Comparison'):

            staging_inference_pred_df = self._batch_inference_by_stage(stage='staging')
            prod_inference_pred_df = self._batch_inference_by_stage(stage='production')

            staging_inference_pred_pdf = staging_inference_pred_df.toPandas()
            prod_inference_pred_pdf = prod_inference_pred_df.toPandas()

            staging_eval_metric = self._get_evaluation_metric(y_true=staging_inference_pred_pdf[self.label_col],
                                                              y_score=staging_inference_pred_pdf['prediction'],
                                                              metric=self.comparison_metric,
                                                              stage='staging')
            _logger.info(f'Candidate Staging model (stage="staging") {self.comparison_metric}: {staging_eval_metric}')

            production_eval_metric = self._get_evaluation_metric(y_true=prod_inference_pred_pdf[self.label_col],
                                                                 y_score=prod_inference_pred_pdf['prediction'],
                                                                 metric=self.comparison_metric,
                                                                 stage='production')
            _logger.info(
                f'Current Production model (stage="production") {self.comparison_metric}: {production_eval_metric}')

            self._run_promotion_logic(staging_eval_metric, production_eval_metric)
