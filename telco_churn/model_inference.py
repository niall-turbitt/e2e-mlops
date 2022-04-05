from dataclasses import dataclass

import pyspark.sql.dataframe
from databricks.feature_store import FeatureStoreClient

from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class ModelInference:

    mlflow_params: dict
    data_input: dict
    data_output: dict

    def _load_inference_df(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Load Spark DataFrame containing lookup keys to join feature data from Feature Store

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        inference_table_name = self.data_input['inference_data']['table_name']
        _logger.info(f'Loading lookup keys from table: {inference_table_name}')
        return spark.table(inference_table_name)

    def _fs_score_batch(self, df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        """
        Load and apply model from MLflow Model Registry using Feature Store API. Features will be automatically
        retrieved from the Feature Store. This method requires that the registered model must have been logged
        with FeatureStoreClient.log_model(), which packages the model with feature metadata. Unless present in df ,
        these features will be looked up from Feature Store and joined with df prior to scoring the model.

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame

            The DataFrame to score the model on. Feature Store features will be joined with df prior to scoring the
            model. df must:

                1. Contain columns for lookup keys required to join feature data from Feature Store, as specified in
                   the feature_spec.yaml artifact.
                2. Contain columns for all source keys required to score the model, as specified in the
                   feature_spec.yaml artifact.
                3. Not contain a column prediction, which is reserved for the modelʼs predictions. df may contain
                   additional columns.

        Returns
        -------
        pyspark.sql.dataframe.DataFrame:
            A Spark DataFrame containing:
                1. All columns of df.
                2. All feature values retrieved from Feature Store.
                3. A column prediction containing the output of the model.
        """
        fs = FeatureStoreClient()
        model_uri = f'models:/{self.mlflow_params["model_registry_name"]}/{self.mlflow_params["model_registry_stage"]}'
        _logger.info(f'Loading model from Model Registry: {model_uri}')

        return fs.score_batch(model_uri, df)

    def run_batch(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Load inference lookup keys, feature data from Feature Store, and score using the loaded model from MLflow
        model registry

        Returns
        -------
        pyspark.sql.dataframe.DataFrame:
            A Spark DataFrame containing:
                1. All columns of df.
                2. All feature values retrieved from Feature Store.
                3. A column prediction containing the output of the model.
        """
        inference_df = self._load_inference_df()
        inference_pred_df = self._fs_score_batch(inference_df)

        return inference_pred_df

    def run_and_write_batch(self, mode: str = 'overwrite'):
        """
        Run batch inference and create predictions table

        Parameters
        ----------
        mode : str
            Specify behavior when predictions data already exists.
            Options include:

                * 'append': Append contents of this :class:`DataFrame` to existing data.
                * 'overwrite': Overwrite existing data.
                * 'error' or `errorifexists`: Throw an exception if data already exists.
                * 'ignore': Silently ignore this operation if data already exists.
        """
        inference_pred_df = self.run_batch()

        predictions_path = self.data_output['predictions_data']['delta_path']
        _logger.info(f'Writing predictions to DBFS: {predictions_path}')
        inference_pred_df.write.format('delta').mode(mode).save(predictions_path)

        predictions_table_name = self.data_output['predictions_data']['table_name']
        if mode == 'overwrite':
            spark.sql(f'DROP TABLE IF EXISTS {predictions_table_name};')
        # TODO: handle appends to predictions table

        spark.sql(f"""CREATE TABLE {predictions_table_name}
                      USING DELTA LOCATION '{predictions_path}';""")
        _logger.info(f'Created predictions table: {predictions_table_name}')
