from dataclasses import dataclass
import pprint
from typing import List

import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

import databricks
from databricks.feature_store import FeatureStoreClient, FeatureLookup

from telco_churn import data_ingest
from telco_churn.pipeline import PipelineCreator
from telco_churn.utils.logger_utils import get_logger

fs = FeatureStoreClient()
_logger = get_logger()


@dataclass
class ModelTrain:

    mlflow_params: dict
    data_input: dict
    pipeline_params: dict
    model_params: dict

    @staticmethod
    def _get_feature_table_lookup(feature_store_params: dict) \
            -> List[databricks.feature_store.entities.feature_lookup.FeatureLookup]:
        """

        Parameters
        ----------
        feature_store_params

        Returns
        -------
        List[databricks.feature_store.entities.feature_lookup.FeatureLookup]
        """
        _logger.info('Creating feature lookups...')
        feature_table_name = feature_store_params['table_name']
        feature_lookup = FeatureLookup(table_name=feature_table_name,
                                       lookup_key=feature_store_params['primary_keys'])

        # Lookup for single feature table
        feature_table_lookup = [feature_lookup]

        return feature_table_lookup

    def _get_fs_training_set(self) -> databricks.feature_store.training_set.TrainingSet:

        feature_store_params = self.data_input['feature_store_params']
        feature_table_lookup = self._get_feature_table_lookup(feature_store_params)
        labels_df = data_ingest.spark_load_table(self.data_input['labels_table_params']['table_name'])
        _logger.info('Creating Feature Store training set...')

        return fs.create_training_set(df=labels_df,
                                      feature_lookups=feature_table_lookup,
                                      label=self.pipeline_params['label_col'],
                                      exclude_columns=self.data_input['feature_store_params']['primary_keys'])

    def _data_preproc(self, fs_training_set: databricks.feature_store.training_set.TrainingSet):

        """

        Parameters
        ----------
        fs_training_set

        Returns
        -------
        train-test splits
        """
        _logger.info('Load training set from Feature Store, converting to pandas DataFrame')
        training_set_pdf = fs_training_set.load_df().toPandas()

        X = training_set_pdf.drop(self.pipeline_params['label_col'], axis=1)
        y = training_set_pdf[self.pipeline_params['label_col']]

        _logger.info(f'Splitting into train/val splits - val_size: {self.pipeline_params["test_size"]}')
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          random_state=self.pipeline_params['random_state'],
                                                          test_size=self.pipeline_params['test_size'],
                                                          stratify=y)

        return X_train, X_val, y_train, y_val

    def train_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> sklearn.pipeline.Pipeline:
        """
        Create baseline sklearn pipeline and fit pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data

        y_train : pd.Series
            Training targets

        Returns
        -------
        scikit-learn pipeline with fitted steps.
        """
        _logger.info('Creating baseline sklearn pipeline...')
        pipeline = PipelineCreator.make_baseline(self.model_params)
        _logger.info('Fitting XGBoostClassifier...')
        _logger.info(f'Model params: {pprint.pformat(self.model_params)}')
        model = pipeline.fit(X_train, y_train)

        return model

    def run(self):

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        # Create Feature Store Training Set
        fs_training_set = self._get_fs_training_set()

        # Load and preprocess data into train/val splits
        X_train, X_val, y_train, y_val = self._data_preproc(fs_training_set)

        model = self.train_baseline(X_train, y_train)

        _logger.info('Logging model to MLflow using Feature Store API')
        fs.log_model(
            model,
            'fs_model',
            flavor=mlflow.sklearn,
            training_set=fs_training_set,
            input_example=X_train[:100],
            signature=infer_signature(X_train, y_train))

        # Training metrics are logged by MLflow autologging
        # Log metrics for the validation set
        _logger.info('Evaluating and logging metrics')
        xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                               prefix='val_')
        print(pd.DataFrame(xgbc_val_metrics, index=[0]))

        if self.mlflow_params['model_registry_name']:
            _logger.info(f'Registering model: {self.mlflow_params["model_registry_name"]}')
            mlflow_run_id = mlflow.active_run().info.run_id
            mlflow.register_model(f'runs:/{mlflow_run_id}/fs_model',
                                  name=self.mlflow_params["model_registry_name"])

