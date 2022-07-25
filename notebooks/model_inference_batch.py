# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # `model_inference_batch`
# MAGIC 
# MAGIC Pipeline to execute model inference.
# MAGIC Apply the model at the specified URI for batch inference on the table with name input_table_name,  writing results to the table with name output_table_name

# COMMAND ----------

# DBTITLE 1,pip install requirements.txt
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# DBTITLE 1,Set env
dbutils.widgets.dropdown('env', 'dev', ['dev', 'staging', 'prod'], 'Environment Name')

# COMMAND ----------

# DBTITLE 1,Module Imports
from typing import Dict

from telco_churn.utils.notebook_utils import load_and_set_env_vars, load_config
from telco_churn.model_inference import ModelInference
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()

# COMMAND ----------

# DBTITLE 1,Load pipeline config params
# Set pipeline name
pipeline_name = 'model_inference_batch'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name)

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# DBTITLE 1,Pipeline Config
# Fetch model_uri
model_name = env_vars['model_name']
model_registry_stage = pipeline_config['mlflow_params']['model_registry_stage']
model_uri = f'models:/{model_name}/{model_registry_stage}'
print(f'model_uri: {model_uri}')

# Set input table name
input_table_name = pipeline_config['data_input']['table_name']
print(f'input_table_name: {input_table_name}')

# Set output table name
predictions_table_database_name = env_vars['predictions_table_database_name']
predictions_table_name = f'{predictions_table_database_name}.{env_vars["predictions_table_name"]}'
print(f'predictions_table_name: {predictions_table_name}')

# COMMAND ----------

# DBTITLE 1,Execute Pipeline
# Instantiate model inference pipeline
model_inference_pipeline = ModelInference(model_uri=model_uri,
                                          input_table_name=input_table_name,
                                          output_table_name=predictions_table_name)

model_inference_pipeline.run_and_write_batch(mode=pipeline_config['data_output']['mode'])
