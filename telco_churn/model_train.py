from dataclasses import dataclass
from typing import List, Dict, Any
import pprint

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

import databricks
from databricks.feature_store import FeatureStoreClient, FeatureLookup

from telco_churn.common import MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from telco_churn.model_train_pipeline import ModelTrainPipeline
from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

fs = FeatureStoreClient()
_logger = get_logger()


@dataclass
class ModelTrainConfig:
    """
    Configuration data class used to execute ModelTrain pipeline.

    Attributes:
        mlflow_tracking_cfg (MLflowTrackingConfig)
            Configuration data class used to unpack MLflow parameters during a model training run.
        feature_store_table_cfg (FeatureStoreTableConfig):
            Configuration data class used to unpack parameters when loading the Feature Store table.
        labels_table_cfg (LabelsTableConfig):
            Configuration data class used to unpack parameters when loading labels table.
        pipeline_params (dict):
            Params to use in preprocessing pipeline. Read from model_train.yml
            - test_size: Proportion of input data to use as training data
            - random_state: Random state to enable reproducible train-test split
        model_params (dict):
            Dictionary of params for model. Read from model_train.yml
        conf (dict):
            [Optional] dictionary of conf file used to trigger pipeline. If provided will be tracked as a yml
            file to MLflow tracking.
        env_vars (dict):
            [Optional] dictionary of environment variables to trigger pipeline. If provided will be tracked as a yml
            file to MLflow tracking.
    """
    mlflow_tracking_cfg: MLflowTrackingConfig
    feature_store_table_cfg: FeatureStoreTableConfig
    labels_table_cfg: LabelsTableConfig
    pipeline_params: Dict[str, Any]
    model_params: Dict[str, Any]
    conf: Dict[str, Any] = None
    env_vars: Dict[str, str] = None


class ModelTrain:
    """
    Class to execute model training. Params, metrics and model artifacts will be tracking to MLflow Tracking.
    Optionally, the resulting model will be registered to MLflow Model Registry if provided.
    """
    def __init__(self, cfg: ModelTrainConfig):
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
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')

    def _get_feature_table_lookup(self) -> List[databricks.feature_store.entities.feature_lookup.FeatureLookup]:
        """
        Create list of FeatureLookup for single feature store table. The FeatureLookup is a value class used to specify
        features to use in a TrainingSet.

        Returns
        -------
        List[databricks.feature_store.entities.feature_lookup.FeatureLookup]
        """
        feature_store_table_cfg = self.cfg.feature_store_table_cfg

        _logger.info('Creating feature lookups...')
        feature_table_name = f'{feature_store_table_cfg.database_name}.{feature_store_table_cfg.table_name}'
        feature_lookup = FeatureLookup(table_name=feature_table_name,
                                       lookup_key=feature_store_table_cfg.primary_keys)
        # Lookup for single feature table
        feature_table_lookup = [feature_lookup]

        return feature_table_lookup

    def get_fs_training_set(self) -> databricks.feature_store.training_set.TrainingSet:
        """
        Create the Feature Store TrainingSet

        Returns
        -------
        databricks.feature_store.training_set.TrainingSet
        """
        feature_store_table_cfg = self.cfg.feature_store_table_cfg
        labels_table_cfg = self.cfg.labels_table_cfg
        labels_df = spark.table(f'{labels_table_cfg.database_name}.{labels_table_cfg.table_name}')

        feature_table_lookup = self._get_feature_table_lookup()
        _logger.info('Creating Feature Store training set...')
        return fs.create_training_set(df=labels_df,
                                      feature_lookups=feature_table_lookup,
                                      label=labels_table_cfg.label_col,
                                      exclude_columns=feature_store_table_cfg.primary_keys)

    def create_train_test_split(self, fs_training_set: databricks.feature_store.training_set.TrainingSet):
        """
        Load the TrainingSet for training. The loaded DataFrame has columns specified by fs_training_set.
        Loaded Spark DataFrame is converted to pandas DataFrame and split into train/test splits.

        Parameters
        ----------
        fs_training_set : databricks.feature_store.training_set.TrainingSet
            Feature Store TrainingSet

        Returns
        -------
        train-test splits
        """
        labels_table_cfg = self.cfg.labels_table_cfg

        _logger.info('Load training set from Feature Store, converting to pandas DataFrame')
        training_set_pdf = fs_training_set.load_df().toPandas()

        X = training_set_pdf.drop(labels_table_cfg.label_col, axis=1)
        y = training_set_pdf[labels_table_cfg.label_col]

        _logger.info(f'Splitting into train/test splits - test_size: {self.cfg.pipeline_params["test_size"]}')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=self.cfg.pipeline_params['random_state'],
                                                            test_size=self.cfg.pipeline_params['test_size'],
                                                            stratify=y)

        return X_train, X_test, y_train, y_test

    def fit_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series) -> sklearn.pipeline.Pipeline:
        """
        Create sklearn pipeline and fit pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data

        y_train : pd.Series
            Training labels

        Returns
        -------
        scikit-learn pipeline with fitted steps.
        """
        _logger.info('Creating sklearn pipeline...')
        pipeline = ModelTrainPipeline.create_train_pipeline(self.cfg.model_params)

        _logger.info('Fitting sklearn RandomForestClassifier...')
        _logger.info(f'Model params: {pprint.pformat(self.cfg.model_params)}')
        model = pipeline.fit(X_train, y_train)

        return model

    def run(self):
        """
        Method to trigger model training, and tracking to MLflow.

        Steps:
            1. Set MLflow experiment (creating a new experiment if it does not already exist)
            2. Start MLflow run
            3. Create Databricks Feature Store training set
            4. Create train-test splits to be used to train and evaluate the model
            5. Define sklearn pipeline using ModelTrainPipeline, and fit on train data
            6. Log trained model using the Databricks Feature Store API. Model will be logged to MLflow with associated
               feature table metadata.
            7. Register the model to MLflow model registry if model_name is provided in mlflow_params
        """
        _logger.info('==========Running model training==========')
        mlflow_tracking_cfg = self.cfg.mlflow_tracking_cfg

        _logger.info('==========Setting MLflow experiment==========')
        self._set_experiment(mlflow_tracking_cfg)
        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        _logger.info('==========Starting MLflow run==========')
        with mlflow.start_run(run_name=mlflow_tracking_cfg.run_name) as mlflow_run:

            if self.cfg.conf is not None:
                # Log config file
                mlflow.log_dict(self.cfg.conf, 'conf.yml')
            if self.cfg.env_vars is not None:
                # Log config file
                mlflow.log_dict(self.cfg.env_vars, 'env_vars.yml')

            # Create Feature Store Training Set
            _logger.info('==========Creating Feature Store training set==========')
            fs_training_set = self.get_fs_training_set()

            # Load and preprocess data into train/test splits
            _logger.info('==========Creating train/test splits==========')
            X_train, X_test, y_train, y_test = self.create_train_test_split(fs_training_set)

            # Fit pipeline with RandomForestClassifier
            _logger.info('==========Fitting RandomForestClassifier model==========')
            model = self.fit_pipeline(X_train, y_train)

            # Log model using Feature Store API
            _logger.info('Logging model to MLflow using Feature Store API')
            fs.log_model(
                model,
                'fs_model',
                flavor=mlflow.sklearn,
                training_set=fs_training_set,
                input_example=X_train[:100],
                signature=infer_signature(X_train, y_train))

            # Training metrics are logged by MLflow autologging
            # Log metrics for the test set
            _logger.info('==========Model Evaluation==========')
            _logger.info('Evaluating and logging metrics')
            test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix='test_')
            print(pd.DataFrame(test_metrics, index=[0]))

            # Register model to MLflow Model Registry if provided
            if mlflow_tracking_cfg.model_name is not None:
                _logger.info('==========MLflow Model Registry==========')
                _logger.info(f'Registering model: {mlflow_tracking_cfg.model_name}')
                mlflow.register_model(f'runs:/{mlflow_run.info.run_id}/fs_model',
                                      name=mlflow_tracking_cfg.model_name)

        _logger.info('==========Model training completed==========')
