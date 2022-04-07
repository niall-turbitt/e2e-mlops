from dataclasses import dataclass

import pyspark.sql.dataframe
from databricks.feature_store import FeatureStoreClient

from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class ModelInference:
    """
    Class to execute model inference

    Attributes:
        model_uri (str): MLflow model uri. Model model must have been logged using the Feature Store API
        inference_data (str): Table names to load as a Spark DataFrame to score the model on. Must contain column(s)
            for lookup keys to join feature data from Feature Store
    """
    model_uri: str
    inference_data: str = None

    def _load_inference_df(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Load Spark DataFrame containing lookup keys to join feature data from Feature Store

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        inference_table_name = self.inference_data
        _logger.info(f'Loading lookup keys from table: {inference_table_name}')
        return spark.table(inference_table_name)

    def fs_score_batch(self, df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
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
        _logger.info(f'Loading model from Model Registry: {self.model_uri}')

        return fs.score_batch(self.model_uri, df)

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
        inference_pred_df = self.fs_score_batch(inference_df)

        return inference_pred_df

    def run_and_write_batch(self, delta_path: str = None, table_name: str = None, mode: str = 'overwrite'):
        """
        Run batch inference, save as Delta table (and optionally) create predictions table

        Parameters
        ----------
        delta_path :  str
            Path to save resulting predictions to (as delta)
        table_name : str
            Name of predictions table
        mode : str
            Specify behavior when predictions data already exists.
            Options include:

                * 'append': Append contents of this :class:`DataFrame` to existing data.
                * 'overwrite': Overwrite existing data.
                * 'error' or `errorifexists`: Throw an exception if data already exists.
                * 'ignore': Silently ignore this operation if data already exists.
        """
        if delta_path is None:
            raise RuntimeError('Provide a path to delta_path to save predictions table to')

        inference_pred_df = self.run_batch()

        _logger.info(f'Writing predictions to DBFS: {delta_path}')
        inference_pred_df.write.format('delta').mode(mode).save(delta_path)

        if table_name:
            if mode == 'overwrite':
                spark.sql(f'DROP TABLE IF EXISTS {table_name};')
            # TODO: handle appends to predictions table

            spark.sql(f"""CREATE TABLE {table_name}
                          USING DELTA LOCATION '{delta_path}';""")
            _logger.info(f'Created predictions table: {table_name}')
