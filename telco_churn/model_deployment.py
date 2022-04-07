# from dataclasses import dataclass
#
# from telco_churn.model_inference import ModelInference
#
#
# @dataclass
# class ModelDeployment:
#     """
#     Class to execute model deployment
#
#     Attributes:
#         model_registry_name (str): Name of model to load in MLflow Model Registry
#         reference_data (str): Name of table to use as a reference DataFrame to score loaded model on
#             Must contain column(s) for lookup keys to join feature data from Feature Store
#     """
#     model_registry_name: str
#     reference_data: str
#
#     def _batch_inference_by_stage(self, stage: str):
#
#         mlflow_params = {'model_registry_name': self.model_registry_name,
#                          'model_registry_stage': stage}
#
#         model_inference = ModelInference(mlflow_params=mlflow_params,
#                                          inference_data=self.reference_data)
#
#         return model_inference.run_batch()
#
#     def run(self):
#
#         staging_inference_pred_df = self._batch_inference_by_stage(stage='staging')
#         prod_inference_pred_df = self._batch_inference_by_stage(stage='production')
#
#         # Evaluate
#
#         # Log comparison metrics to MLflow
#
#         # If staging > prod metric , promote staging model to production
