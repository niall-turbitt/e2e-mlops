from dataclasses import dataclass

import mlflow
import pandas as pd
import pyspark.sql
from mlflow.tracking import MlflowClient

from telco_churn.common import MLflowTrackingConfig
from telco_churn.model_inference import ModelInference
from telco_churn.utils.evaluation_utils import ModelEvaluation
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class ModelDeploymentConfig:
    """
    Attributes:
        mlflow_tracking_cfg (MLflowTrackingConfig)
            Configuration data class used to unpack MLflow parameters during a model training run.
        reference_data (str): Name of table to use as a reference DataFrame to score loaded model against.
            Must contain column(s) for lookup keys to join feature data from Feature Store
        label_col (str): Name of label column in input data
        comparison_metric (str): Name of evaluation metric to use when comparing models
        higher_is_better (bool): Boolean indicating whether a higher value for the evaluation metric equates to better
            model performance
    """
    mlflow_tracking_cfg: MLflowTrackingConfig
    reference_data: str
    label_col: str = 'churn'
    comparison_metric: str = 'roc_auc_score'
    higher_is_better: bool = True


class ModelDeployment:
    """
    Class to execute model deployment. This class orchestrates the comparison of the current Production model versus
    Staging model. The Production model will be the most recent model version under registered in the MLflow Model
    Registry under the provided model_name, for stage="Production". Likewise for Staging.

    Execution will involve loading the models and performing batch inference for a specified reference dataset.
    The two models will be compared using the specified comparison_metric.
    higher_is_better indicates whether a higher value for the evaluation metric equates to a better peforming model.
    Dependent on this comparison the candidate Staging model will be either promoted to Production (and the current
    Production model archived) if performing better, or the Staging model will be archived if it does not perform
    better than the current Production model.

    Metrics computed when comparing the two models will be logged to MLflow, under the provided experiment_id or
    experiment_path.
    """
    def __init__(self, cfg: ModelDeploymentConfig):
        self.cfg = cfg

    @staticmethod
    def _set_experiment(mlflow_tracking_cfg: MLflowTrackingConfig):
        """
        Set MLflow experiment. Use one of either experiment_id or experiment_path
        """
        if mlflow_tracking_cfg.experiment_id is not None:
            _logger.info(f'MLflow experiment_id: {mlflow_tracking_cfg.experiment_id}')
            mlflow.set_experiment(experiment_id=mlflow_tracking_cfg.experiment_id)
        elif mlflow_tracking_cfg.experiment_path is not None:
            _logger.info(f'MLflow experiment_path: {mlflow_tracking_cfg.experiment_path}')
            mlflow.set_experiment(experiment_name=mlflow_tracking_cfg.experiment_path)
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in MLflowTrackingConfig')

    def _get_model_uri_by_stage(self, stage: str):
        return f'models:/{self.cfg.mlflow_tracking_cfg.model_name}/{stage}'

    def _batch_inference_by_stage(self, stage: str) -> pyspark.sql.DataFrame:
        """
        Load and compute batch inference using model loaded from an MLflow Model Registry stage.
        Inference is computed on reference data specified. The model will use this reference data to look up feature
        values for primary keys, and use the loaded features as input for model scoring.
        The most recent model under the specified stage will be loaded. The registered model must have been logged to
        MLflow using the Feature Store API.

        Parameters
        ----------
        stage : str
            MLflow Model Registry stage

        Returns
        -------
        Spark DataFrame containing primary keys of the reference data, the loaded features from the feature store and
        prediction from model scoring
        """
        model_uri = self._get_model_uri_by_stage(stage=stage)
        _logger.info(f'Computing batch inference using: {model_uri}')
        _logger.info(f'Reference data: {self.cfg.reference_data}')
        model_inference = ModelInference(model_uri=model_uri,
                                         input_table_name=self.cfg.reference_data)

        return model_inference.run_batch()

    @staticmethod
    def _get_evaluation_metric(y_true: pd.Series, y_score: pd.Series, metric: str, stage: str) -> float:
        """
        Trigger evaluation, and return evaluation specified. A dictionary of evaluation metrics will be tracked to
        MLflow tracking.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
            True labels or binary label indicators
        y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target scores.
        metric : str
            Name of metric to retrieve from evaluation dictionary
        stage : str
            Name of the MLflow Registry model stage corresponding. Used as a prefix when logging metrics

        Returns
        -------
        Evaluation metric
        """
        metric_prefix = stage + "_"
        eval_dict = ModelEvaluation().evaluate(y_true, y_score, metric_prefix=metric_prefix)
        mlflow.log_metrics(eval_dict)
        eval_metric = eval_dict[metric_prefix + metric]

        return eval_metric

    def _run_promotion_logic(self, staging_eval_metric: float, production_eval_metric: float):
        """
        Basic logic to either promote a candidate Staging model performing better than the current Production model,
        or alternatively archive the Staging model if not outperforming Production model.

        Parameters
        ----------
        staging_eval_metric :  float
            Evaluation metric computed using Staging model
        production_eval_metric : float
            Evaluation metric computed using Production model
        """
        client = MlflowClient()
        model_name = self.cfg.mlflow_tracking_cfg.model_name
        staging_model_version = client.get_latest_versions(name=model_name, stages=['staging'])[0]

        _logger.info(f'metric={self.cfg.comparison_metric}')
        _logger.info(f'higher_is_better={self.cfg.higher_is_better}')
        if self.cfg.higher_is_better:
            if staging_eval_metric <= production_eval_metric:
                _logger.info('Candidate Staging model DOES NOT perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="archived"')
                client.transition_model_version_stage(name=model_name,
                                                      version=staging_model_version.version,
                                                      stage='archived')

            elif staging_eval_metric > production_eval_metric:
                _logger.info('Candidate Staging model DOES perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="production"')
                _logger.info('Existing Production model will be archived')
                client.transition_model_version_stage(name=model_name,
                                                      version=staging_model_version.version,
                                                      stage='production',
                                                      archive_existing_versions=True)

        else:
            if staging_eval_metric >= production_eval_metric:
                _logger.info('Candidate Staging model DOES NOT perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="archived"')
                client.transition_model_version_stage(name=model_name,
                                                      version=staging_model_version.version,
                                                      stage='archived')

            elif staging_eval_metric < production_eval_metric:
                _logger.info('Candidate Staging model DOES perform better than current Production model')
                _logger.info('Transition candidate model from stage="staging" to stage="production"')
                _logger.info('Existing Production model will be archived')
                client.transition_model_version_stage(name=model_name,
                                                      version=staging_model_version.version,
                                                      stage='production',
                                                      archive_existing_versions=True)

    def run(self):
        """
        Runner method to orchestrate model comparison and potential model promotion.

        Steps:
            1. Set MLflow Tracking experiment. Used to track metrics computed when comparing Staging versus Production
               models.
            2. Load Staging and Production models and score against reference dataset provided. The reference data
               specified must currently be a table.
            3. Compute evaluation metric for both Staging and Production model predictions against reference data
            4. If higher_is_better=True, the Staging model will be promoted in place of the Production model iff the
               Staging model evaluation metric is higher than the Production model evaluation metric.
               If higher_is_better=False, the Staging model will be promoted in place of the Production model iff the
               Staging model evaluation metric is lower than the Production model evaluation metric.

        """
        _logger.info('==========Running model deployment==========')

        _logger.info('==========Setting MLflow experiment==========')
        mlflow_tracking_cfg = self.cfg.mlflow_tracking_cfg
        self._set_experiment(mlflow_tracking_cfg)

        with mlflow.start_run(run_name=mlflow_tracking_cfg.run_name):

            _logger.info('==========Batch inference: staging model==========')
            staging_inference_pred_df = self._batch_inference_by_stage(stage='staging')
            staging_inference_pred_pdf = staging_inference_pred_df.toPandas()
            _logger.info('==========Batch inference: production model==========')
            prod_inference_pred_df = self._batch_inference_by_stage(stage='production')
            prod_inference_pred_pdf = prod_inference_pred_df.toPandas()

            _logger.info('==========Model evaluation: staging model==========')
            staging_eval_metric = self._get_evaluation_metric(y_true=staging_inference_pred_pdf[self.cfg.label_col],
                                                              y_score=staging_inference_pred_pdf['prediction'],
                                                              metric=self.cfg.comparison_metric,
                                                              stage='staging')
            _logger.info(f'Candidate Staging model (stage="staging") {self.cfg.comparison_metric}: {staging_eval_metric}')

            _logger.info('==========Model evaluation: production model==========')
            production_eval_metric = self._get_evaluation_metric(y_true=prod_inference_pred_pdf[self.cfg.label_col],
                                                                 y_score=prod_inference_pred_pdf['prediction'],
                                                                 metric=self.cfg.comparison_metric,
                                                                 stage='production')
            _logger.info(
                f'Current Production model (stage="production") {self.cfg.comparison_metric}: {production_eval_metric}')

            _logger.info('==========Model comparison: candidate staging model vs current production model==========')
            self._run_promotion_logic(staging_eval_metric, production_eval_metric)

            _logger.info('==========Model deployment completed==========')
