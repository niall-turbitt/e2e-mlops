import pyspark.sql.dataframe

from telco_churn.common import Job
from telco_churn import data_ingest
from telco_churn.data_prep import DataPreprocessor
from telco_churn.utils import feature_store_utils
from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class FeatureTableCreator(Job):

    def setup(self, database_name: str, table_name: str) -> None:
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
        self.spark.sql(f'create database if not exists {database_name};')
        self.spark.sql(f'use {database_name};')
        self.spark.sql(f'drop table if exists {table_name};')

    def run_data_ingest(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Run data ingest step

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            Input Spark DataFrame
        """
        return data_ingest.spark_load_table(table=self.conf['data_ingest_params']['input_table'])

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
        data_preprocessor = DataPreprocessor(**self.conf['data_prep_params'])
        preproc_df = data_preprocessor.run(input_df)

        return preproc_df

    def launch(self) -> None:
        """
        Launch FeatureStoreTableCreator job
        """
        _logger.info("Launching FeatureTableCreator job")

        # TODO - insert check to see if feature table already exists
        # TODO - delete feature table if already exisits?

        _logger.info('==========Data Ingest==========')
        input_df = self.run_data_ingest()

        _logger.info('==========Data Prep==========')
        preproc_df = self.run_data_prep(input_df)

        _logger.info('==========Create Feature Table==========')
        fs_database_name = self.conf['feature_store_params']['database_name']
        fs_table_name = self.conf['feature_store_params']['table_name']
        feature_table_name = f'{fs_database_name}.{fs_table_name}'
        # TODO: handle existing feature table more gracefully.
        #  Currently to start afresh involves deleting feature table via UI
        # Create database if not exists, drop table if it already exists
        self.setup(database_name=fs_database_name, table_name=fs_table_name)
        # TODO: add method to delete table in FeatureStore if exists

        # TODO: refactor the following
        # Store only features for each customerID, storing customerID, churn in separate churn_labels table
        # During model training we will use the churn_labels table to join features into
        churn_features_df = preproc_df.drop(self.conf['data_prep_params']['label_col'])
        _logger.info(f'Creating and writing features to feature table: {fs_database_name}.{fs_table_name}')
        feature_store_utils.create_and_write_feature_table(churn_features_df,
                                                           feature_table_name,
                                                           primary_keys=self.conf['feature_store_params']['primary_keys'],
                                                           description=self.conf['feature_store_params']['description'])

        _logger.info('==========Create Labels Table==========')
        if type(self.conf['feature_store_params']['primary_keys']) is str:
            labels_table_cols = [self.conf['feature_store_params']['primary_keys'],
                                 self.conf['labels_table_params']['label_col']]
        elif type(self.conf['feature_store_params']['primary_keys']) is list:
            labels_table_cols = self.conf['feature_store_params']['primary_keys'] + \
                                [self.conf['labels_table_params']['label_col']]
        else:
            raise RuntimeError(f'{self.conf["feature_store_params"]["primary_keys"]} must be of either str of list type')

        labels_database_name = self.conf['labels_table_params']['database_name']
        labels_table_name = self.conf['labels_table_params']['table_name']
        labels_dbfs_path = self.conf["labels_table_params"]["dbfs_path"]
        # Create database if not exists, drop table if it already exists
        self.setup(database_name=labels_database_name, table_name=labels_table_name)
        # DataFrame of customerID/churn labels
        labels_df = preproc_df.select(labels_table_cols)
        _logger.info(f'Writing labels to DBFS: {labels_dbfs_path}')
        labels_df.write.format('delta').mode('overwrite').save(labels_dbfs_path)
        spark.sql(f"""CREATE TABLE {labels_database_name}.{labels_table_name} 
                      USING DELTA LOCATION '{labels_dbfs_path}';""")
        _logger.info(f'Created labels table: {labels_database_name}.{labels_table_name}')

        _logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = FeatureTableCreator()
    job.launch()
