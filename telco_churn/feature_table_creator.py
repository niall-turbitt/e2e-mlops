from dataclasses import dataclass

import pyspark.sql.dataframe

from telco_churn import data_ingest
from telco_churn.data_prep import DataPreprocessor
from telco_churn.utils import feature_store_utils
from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class FeatureTableCreator:
    """
    Class to execute a pipeline to create a Feature Store table, and labels table
    """
    data_ingest_params: dict
    data_prep_params: dict
    feature_store_params: dict
    labels_table_params: dict

    @staticmethod
    def setup(database_name: str, table_name: str) -> None:
        """
        Set up database to use. Create the database {database_name} if it doesn't exist, and drop the table {table_name}
        if it exists

        Parameters
        ----------
        database_name : str
            Database to create if it doesn't exist. Otherwise use database of the name provided
        table_name : str
            Drop table if it already exists
        """
        _logger.info(f'Creating database {database_name} if not exists')
        spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name};')
        spark.sql(f'USE {database_name};')
        spark.sql(f'DROP TABLE IF EXISTS {table_name};')

    def run_data_ingest(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Run data ingest step

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            Input Spark DataFrame
        """
        return data_ingest.spark_load_table(table=self.data_ingest_params['input_table'])

    def run_data_prep(self, input_df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        """
        Run data preparation step

        Parameters
        ----------
        input_df : pyspark.sql.dataframe.DataFrame
            Input Spark DataFrame

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
        """
        data_preprocessor = DataPreprocessor(**self.data_prep_params)
        preproc_df = data_preprocessor.run(input_df)

        return preproc_df

    def run_feature_table_create(self, df: pyspark.sql.dataframe.DataFrame,
                                 database_name: str, table_name: str) -> None:
        """

        Parameters
        ----------
        df
        database_name
        table_name
        """
        # TODO: handle existing feature table more gracefully.
        #  Currently to start afresh involves deleting feature table via UI
        # Create database if not exists, drop table if it already exists
        self.setup(database_name=database_name, table_name=table_name)
        # TODO: add method to delete table in FeatureStore if exists

        # Store only features for each customerID, storing customerID, churn in separate churn_labels table
        # During model training we will use the churn_labels table to join features into
        features_df = df.drop(self.labels_table_params['label_col'])
        feature_table_name = f'{database_name}.{table_name}'
        _logger.info(f'Creating and writing features to feature table: {feature_table_name}')
        feature_store_utils.create_and_write_feature_table(features_df,
                                                           feature_table_name,
                                                           primary_keys=self.feature_store_params['primary_keys'],
                                                           description=self.feature_store_params['description'])

    def run_labels_table_create(self, df: pyspark.sql.dataframe.DataFrame) -> None:
        """

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame containing primary keys column and label column
        """

        if type(self.feature_store_params['primary_keys']) is str:
            labels_table_cols = [self.feature_store_params['primary_keys'],
                                 self.labels_table_params['label_col']]
        elif type(self.feature_store_params['primary_keys']) is list:
            labels_table_cols = self.feature_store_params['primary_keys'] + \
                                [self.labels_table_params['label_col']]
        else:
            raise RuntimeError(
                f'{self.feature_store_params["primary_keys"]} must be of either str of list type')

        labels_database_name = self.labels_table_params['database_name']
        labels_table_name = self.labels_table_params['table_name']
        labels_dbfs_path = self.labels_table_params['dbfs_path']
        # Create database if not exists, drop table if it already exists
        self.setup(database_name=labels_database_name, table_name=labels_table_name)
        # DataFrame of customerID/churn labels
        labels_df = df.select(labels_table_cols)
        _logger.info(f'Writing labels to DBFS: {labels_dbfs_path}')
        labels_df.write.format('delta').mode('overwrite').save(labels_dbfs_path)
        spark.sql(f"""CREATE TABLE {labels_database_name}.{labels_table_name}
                      USING DELTA LOCATION '{labels_dbfs_path}';""")
        _logger.info(f'Created labels table: {labels_database_name}.{labels_table_name}')

    def run(self) -> None:
        """
        Run feature table creation pipeline
        """
        # TODO - insert check to see if feature table already exists
        # TODO - delete feature table if already exisits?

        _logger.info('==========Data Ingest==========')
        input_df = self.run_data_ingest()

        _logger.info('==========Data Prep==========')
        preproc_df = self.run_data_prep(input_df)

        _logger.info('==========Create Feature Table==========')
        fs_database_name = self.feature_store_params['database_name']
        fs_table_name = self.feature_store_params['table_name']
        self.run_feature_table_create(preproc_df, database_name=fs_database_name, table_name=fs_table_name)

        _logger.info('==========Create Labels Table==========')
        self.run_labels_table_create(preproc_df)
