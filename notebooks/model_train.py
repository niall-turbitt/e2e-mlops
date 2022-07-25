# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # `model_train`
# MAGIC 
# MAGIC Pipeline to execute model training. Params, metrics and model artifacts will be tracking to MLflow Tracking.
# MAGIC Optionally, the resulting model will be registered to MLflow Model Registry if provided.

# COMMAND ----------

# DBTITLE 1,pip install requirements.txt
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# DBTITLE 1,Set env
dbutils.widgets.dropdown('env', 'dev', ['dev', 'staging', 'prod'], 'Environment Name')

# COMMAND ----------

# DBTITLE 1,Module Imports
from telco_churn.utils.notebook_utils import load_and_set_env_vars, load_config

from telco_churn.common import MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from telco_churn.model_train import ModelTrain, ModelTrainConfig
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()

# COMMAND ----------

# DBTITLE 1,Load pipeline config params
# Set pipeline name
pipeline_name = 'model_train'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name)

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# DBTITLE 1,Pipeline Config
# Set MLflowTrackingConfig
mlflow_tracking_cfg = MLflowTrackingConfig(run_name=pipeline_config['mlflow_params']['run_name'],
                                           experiment_path=env_vars['model_train_experiment_path'],
                                           model_name=env_vars['model_name'])


# Set FeatureStoreTableConfig
feature_store_table_cfg = FeatureStoreTableConfig(database_name=env_vars['feature_store_database_name'],
                                                  table_name=env_vars['feature_store_table_name'],
                                                  primary_keys=env_vars['feature_store_table_primary_keys'])

# Set LabelsTableConfig
labels_table_cfg = LabelsTableConfig(database_name=env_vars['labels_table_database_name'],
                                 table_name=env_vars['labels_table_name'],
                                 label_col=env_vars['labels_table_label_col'])

# Set pipeline_params
pipeline_params = pipeline_config['pipeline_params']

# Set model_params
model_params = pipeline_config['model_params']

# Define ModelTrainConfig
cfg = ModelTrainConfig(mlflow_tracking_cfg=mlflow_tracking_cfg,
                       feature_store_table_cfg=feature_store_table_cfg,
                       labels_table_cfg=labels_table_cfg,
                       pipeline_params=pipeline_params,
                       model_params=model_params,
                       conf=pipeline_config,    # Track pipeline_config to mlflow
                       env_vars=env_vars        # Track env_vars to mlflow
                      )

# COMMAND ----------

# DBTITLE 1,Execute Pipeline
# Instantiate pipeline
model_train_pipeline = ModelTrain(cfg)
model_train_pipeline.run()
