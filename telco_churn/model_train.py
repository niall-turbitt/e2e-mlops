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

from telco_churn.model_train_pipeline import ModelTrainPipeline
from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

fs = FeatureStoreClient()
_logger = get_logger()


@dataclass
class ModelTrain:
    """
    Class to execute model training. Params, metrics and model artifacts will be tracking to MLflow Tracking. Optionally,
    the resulting model will be registered to MLflow Model Registry if provided

    Attributes:
        mlflow_params : dict
            Dictionary of MLflow parameters to use (either experiment_id or experiment_path can be used):
                - experiment_id: ID of the MLflow experiment to be activated. If an experiment with this ID does not
                     exist, an exception is thrown.
                 - experiment_path: Case sensitive name of the experiment to be activated. If an experiment with this
                     name does not exist, a new experiment wth this name is created.
                - run_name: Name of MLflow run
                - model_registry_name: Name of the registered model under which to create a new model version.
                      If a registered model with the given name does not exist, it will be created automatically.
        data_input (dict): Dictionary of feature_store_params, labels_table_params. Each of which are themselves dicts.
            feature_store_params:
                - table_name: Name of Databricks Feature Store feature table in format <database_name>.<table_name>
                - primary_keys: (str or list) Name(s) of the primary key(s) column(s)
            labels_table_params:
                table_name: Name of labels table with columns primary_keys, and label_col. Table name in format
                     <database_name>.<table_name>
        pipeline_params (dict): Params to use in preprocessing pipeline
            - label_col: Name of label column
            - test_size: Proportion of input data to use as training data
            - random_state: Random state to enable reproducible train-test split
        model_params (dict): Dictionary of params for model
        conf (dict): Optional dictionary of conf file used to trigger pipeline. If provided will be tracked as a yml
            file to MLflow tracking.
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
            _logger.info(f'MLflow experiment_id: {self.mlflow_params["experiment_id"]}')
            mlflow.set_experiment(experiment_id=self.mlflow_params['experiment_id'])
        elif 'experiment_path' in self.mlflow_params:
            _logger.info(f'MLflow experiment_path: {self.mlflow_params["experiment_path"]}')
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

    def get_fs_training_set(self) -> databricks.feature_store.training_set.TrainingSet:
        """
        Create the Feature Store TrainingSet

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
        _logger.info('Load training set from Feature Store, converting to pandas DataFrame')
        training_set_pdf = fs_training_set.load_df().toPandas()

        X = training_set_pdf.drop(self.pipeline_params['label_col'], axis=1)
        y = training_set_pdf[self.pipeline_params['label_col']]

        _logger.info(f'Splitting into train/test splits - test_size: {self.pipeline_params["test_size"]}')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=self.pipeline_params['random_state'],
                                                            test_size=self.pipeline_params['test_size'],
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
        pipeline = ModelTrainPipeline.create_train_pipeline(self.model_params)

        _logger.info('Fitting sklearn RandomForestClassifier...')
        _logger.info(f'Model params: {pprint.pformat(self.model_params)}')
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
            7. Register the model to MLflow model registry if model_registry_name is provided in mlflow_params
        """
        _logger.info('==========Running model training==========')

        _logger.info('==========Setting MLflow experiment==========')
        self._set_experiment()
        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        _logger.info('==========Starting MLflow run==========')
        with mlflow.start_run(run_name=self.mlflow_params['run_name']) as mlflow_run:

            if self.conf is not None:
                # Log config file
                mlflow.log_dict(self.conf, 'conf.yml')

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
            if self.mlflow_params['model_registry_name']:
                _logger.info('==========MLflow Model Registry==========')
                _logger.info(f'Registering model: {self.mlflow_params["model_registry_name"]}')
                mlflow.register_model(f'runs:/{mlflow_run.info.run_id}/fs_model',
                                      name=self.mlflow_params['model_registry_name'])

        _logger.info('==========Model training completed==========')
