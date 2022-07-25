import pyspark.sql.dataframe
from databricks.feature_store import FeatureStoreClient

from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelInference:
    """
    Class to execute model inference.
    Apply the model at the specified URI for batch inference on the table with name input_table_name,
    writing results to the table with name output_table_name
    """
    def __init__(self, model_uri: str, input_table_name: str, output_table_name: str = None):
        """

        Parameters
        ----------
        model_uri : str
            MLflow model uri. Model model must have been logged using the Feature Store API
        input_table_name : str
            Table name to load as a Spark DataFrame to score the model on. Must contain column(s)
            for lookup keys to join feature data from Feature Store
        output_table_name : str
            Output table name to write results to
        """
        self.model_uri = model_uri
        self.input_table_name = input_table_name
        self.output_table_name = output_table_name

    def _load_input_table(self) -> pyspark.sql.DataFrame:
        """
        Load Spark DataFrame containing lookup keys to join feature data from Feature Store

        Returns
        -------
        pyspark.sql.DataFrame
        """
        input_table_name = self.input_table_name
        _logger.info(f"Loading lookup keys from input table: {input_table_name}")
        return spark.table(input_table_name)

    def fs_score_batch(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Load and apply model from MLflow Model Registry using Feature Store API. Features will be automatically
        retrieved from the Feature Store. This method requires that the registered model must have been logged
        with FeatureStoreClient.log_model(), which packages the model with feature metadata. Unless present in df ,
        these features will be looked up from Feature Store and joined with df prior to scoring the model.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
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
        pyspark.sql.DataFrame:
            A Spark DataFrame containing:
                1. All columns of df.
                2. All feature values retrieved from Feature Store.
                3. A column prediction containing the output of the model.
        """
        fs = FeatureStoreClient()
        _logger.info(f"Loading model from Model Registry: {self.model_uri}")

        return fs.score_batch(self.model_uri, df)

    def run_batch(self) -> pyspark.sql.DataFrame:
        """
        Load inference lookup keys, feature data from Feature Store, and score using the loaded model from MLflow
        model registry

        Returns
        -------
        pyspark.sql.DataFrame:
            A Spark DataFrame containing:
                1. All columns of inference df.
                2. All feature values retrieved from Feature Store.
                3. A column prediction containing the output of the model.
        """
        input_df = self._load_input_table()
        pred_df = self.fs_score_batch(input_df)

        return pred_df

    def run_and_write_batch(self, mode: str = 'overwrite') -> None:
        """
        Run batch inference, save as Delta table to `self.output_table_name`

        Parameters
        ----------
        mode : str
            Specify behavior when predictions data already exists.
                        Options include:
                            * "append": Append contents of this :class:`DataFrame` to existing data.
                            * "overwrite": Overwrite existing data.

        Returns
        -------

        """
        _logger.info("==========Running batch model inference==========")
        pred_df = self.run_batch()

        _logger.info("==========Writing predictions==========")
        _logger.info(f"mode={mode}")
        _logger.info(f"Predictions written to {self.output_table_name}")
        # Model predictions are written to the Delta table provided as input.
        # Delta is the default format in Databricks Runtime 8.0 and above.
        pred_df.write.format("delta").mode(mode).saveAsTable(self.output_table_name)

        _logger.info("==========Batch model inference completed==========")
