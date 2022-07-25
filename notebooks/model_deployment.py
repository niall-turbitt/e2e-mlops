# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # `model_deployment`
# MAGIC 
# MAGIC Pipeline to execute model deployment. This class orchestrates the comparison of the current Production model versus Staging model. 
# MAGIC The Production model will be the most recent model version under registered in the MLflow Model Registry under the provided model_name, for stage="Production". Likewise for Staging.
# MAGIC Execution will involve loading the models and performing batch inference for a specified reference dataset.
# MAGIC The two models will be compared using the specified comparison_metric.
# MAGIC `higher_is_better` indicates whether a higher value for the evaluation metric equates to a better peforming model.
# MAGIC Dependent on this comparison the candidate Staging model will be either promoted to Production (and the current
# MAGIC Production model archived) if performing better, or the Staging model will be archived if it does not perform better than the current Production model.
# MAGIC 
# MAGIC Metrics computed when comparing the two models will be logged to MLflow, under the provided experiment_id or experiment_path.
# MAGIC 
# MAGIC **Pipeline Steps**:
# MAGIC 1. Set MLflow Tracking experiment. Used to track metrics computed when comparing Staging versus Production
# MAGIC        models.
# MAGIC 1. Load Staging and Production models and score against reference dataset provided. The reference data specified must currently be a table.
# MAGIC 1. Compute evaluation metric for both Staging and Production model predictions against reference data
# MAGIC 1. If higher_is_better=True, the Staging model will be promoted in place of the Production model iff the Staging model evaluation metric is higher than the Production model evaluation metric. If higher_is_better=False, the Staging model will be promoted in place of the Production model iff the Staging model evaluation metric is lower than the Production model evaluation metric.

# COMMAND ----------

# DBTITLE 1,pip install requirements.txt
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# DBTITLE 1,Set env
dbutils.widgets.dropdown('env', 'dev', ['dev', 'staging', 'prod'], 'Environment Name')

# COMMAND ----------

# DBTITLE 1,Module Imports
from telco_churn.utils.notebook_utils import load_and_set_env_vars, load_config

from telco_churn.common import MLflowTrackingConfig
from telco_churn.model_deployment import ModelDeployment, ModelDeploymentConfig
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()

# COMMAND ----------

# DBTITLE 1,Load pipeline config params
# Set pipeline name
pipeline_name = 'model_deployment'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name)

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# DBTITLE 1,Pipeline Config
# Set MLflowTrackingConfig - comparison metrics logged to MLflow
mlflow_tracking_cfg = MLflowTrackingConfig(experiment_path=env_vars['model_deploy_experiment_path'],
                                           run_name='staging_vs_prod_comparison',
                                           model_name=env_vars['model_name'])

# Define reference dataset 
reference_table_database_name = env_vars['reference_table_database_name']
reference_table_name = f'{reference_table_database_name}.{env_vars["reference_table_name"]}'   

# Set label col from reference dataset
label_col = env_vars['reference_table_label_col']

# Params defining how to compare staging vs prod models
model_comparison_params = pipeline_config['model_comparison_params']

# Define ModelDeploymentConfig
cfg = ModelDeploymentConfig(mlflow_tracking_cfg=mlflow_tracking_cfg,
                            reference_data=reference_table_name,
                            label_col=label_col,
                            comparison_metric=model_comparison_params['metric'],
                            higher_is_better=model_comparison_params['higher_is_better'])

# COMMAND ----------

# DBTITLE 1,Execute Pipeline
# Instantiate pipeline
model_deployment_pipeline = ModelDeployment(cfg)
model_deployment_pipeline.run()
