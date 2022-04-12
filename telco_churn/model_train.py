from dataclasses import dataclass
from typing import List
import pprint

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

import databricks
from databricks.feature_store import FeatureStoreClient, FeatureLookup

from telco_churn.pipeline import PipelineCreator
from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

fs = FeatureStoreClient()
_logger = get_logger()


@dataclass
class ModelTrain:
    """
    Class to execute model training

    Attributes:
        mlflow_params (dict):
        data_input (dict):
        pipeline_params (dict):
        model_params (dict):
        conf (dict):
    """
    mlflow_params: dict
    data_input: dict
    pipeline_params: dict
    model_params: dict
    conf: dict = None

    def _set_experiment(self):
        """
        Set MLflow experiment. Use one of either experiment_id or experiment_path
        """
        if 'experiment_id' in self.mlflow_params:
            mlflow.set_experiment(experiment_id=self.mlflow_params['experiment_id'])
        elif 'experiment_path' in self.mlflow_params['experiment_path']:
            mlflow.set_experiment(experiment_name=self.mlflow_params['experiment_path'])
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')

    @staticmethod
    def _get_feature_table_lookup(feature_store_params: dict) \
            -> List[databricks.feature_store.entities.feature_lookup.FeatureLookup]:
        """
        Create list of FeatureLookup for single feature store table. The FeatureLookup is a value class used to specify
        features to use in a TrainingSet.

        Parameters
        ----------
        feature_store_params : dict
            Dictionary containing the Feature Store table name, and table primary keys

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
        """
        Get the Feature Store TrainingSet

        Returns
        -------
        databricks.feature_store.training_set.TrainingSet
        """
        feature_store_params = self.data_input['feature_store_params']
        feature_table_lookup = self._get_feature_table_lookup(feature_store_params)
        labels_df = spark.table(self.data_input['labels_table_params']['table_name'])
        _logger.info('Creating Feature Store training set...')

        return fs.create_training_set(df=labels_df,
                                      feature_lookups=feature_table_lookup,
                                      label=self.pipeline_params['label_col'],
                                      exclude_columns=self.data_input['feature_store_params']['primary_keys'])

    def _data_preproc(self, fs_training_set: databricks.feature_store.training_set.TrainingSet):
        """
        Load the TrainingSet fortraining. The loaded DataFrame has columns specified by fs_training_set.
        Loaded Spark DataFrame is converted to pandas DataFrame and split into train/val splits.


        Parameters
        ----------
        fs_training_set : databricks.feature_store.training_set.TrainingSet
            Feature Store TrainingSet

        Returns
        -------
        train-val splits
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
        self._set_experiment()
        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        with mlflow.start_run(run_name=self.mlflow_params['run_name']) as mlflow_run:

            if self.conf is not None:
                # Log config file
                mlflow.log_dict(self.conf, 'conf.yml')

            # Create Feature Store Training Set
            fs_training_set = self._get_fs_training_set()

            # Load and preprocess data into train/val splits
            X_train, X_val, y_train, y_val = self._data_preproc(fs_training_set)

            # Fit baseline pipeline with XGBoost
            model = self.train_baseline(X_train, y_train)

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
            # Log metrics for the validation set
            _logger.info('Evaluating and logging metrics')
            val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix='val_')
            print(pd.DataFrame(val_metrics, index=[0]))

            # TODO: log SHAP explainer

            if self.mlflow_params['model_registry_name']:
                _logger.info(f'Registering model: {self.mlflow_params["model_registry_name"]}')
                mlflow.register_model(f'runs:/{mlflow_run.info.run_id}/fs_model',
                                      name=self.mlflow_params["model_registry_name"])
