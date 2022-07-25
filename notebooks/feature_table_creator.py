# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # `feature_table_creator`
# MAGIC 
# MAGIC Pipeline to create a Feature Store table, and separate labels table

# COMMAND ----------

# DBTITLE 1,pip install requirements.txt
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# DBTITLE 1,Set env
dbutils.widgets.dropdown('env', 'dev', ['dev', 'staging', 'prod'], 'Environment Name')

# COMMAND ----------

# DBTITLE 1,Module Imports
from telco_churn.utils.notebook_utils import load_and_set_env_vars, load_config

from telco_churn.common import FeatureStoreTableConfig, LabelsTableConfig
from telco_churn.feature_table_creator import FeatureTableCreator, FeatureTableCreatorConfig
from telco_churn.featurize import FeaturizerConfig
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()

# COMMAND ----------

# DBTITLE 1,Load pipeline config params
# Set pipeline name
pipeline_name = 'feature_table_creator'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name)

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# DBTITLE 1,Pipeline Config
# Set FeaturizerConfig - data preparation config
featurizer_cfg = FeaturizerConfig(**pipeline_config['data_prep_params'])

# Set Feature Store feature table config
feature_store_table_cfg = FeatureStoreTableConfig(database_name=env_vars['feature_store_database_name'],
                                                  table_name=env_vars['feature_store_table_name'],
                                                  primary_keys=env_vars['feature_store_table_primary_keys'],
                                                  description=env_vars['feature_store_table_description'])

# Set Labels Table config
labels_table_cfg = LabelsTableConfig(database_name=env_vars['labels_table_database_name'],
                                     table_name=env_vars['labels_table_name'],
                                     label_col=env_vars['labels_table_label_col'],
                                     dbfs_path=env_vars['labels_table_dbfs_path'])

# Set FeatureTableCreatorConfig
cfg = FeatureTableCreatorConfig(input_table=pipeline_config['input_table'],
                                featurizer_cfg=featurizer_cfg,
                                feature_store_table_cfg=feature_store_table_cfg,
                                labels_table_cfg=labels_table_cfg)

# COMMAND ----------

# DBTITLE 1,Execute Pipeline
# Instantiate pipeline
feature_table_creator_pipeline = FeatureTableCreator(cfg)
feature_table_creator_pipeline.run()
