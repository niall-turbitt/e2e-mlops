import pyspark.sql.dataframe

from telco_churn.common import Job
from telco_churn import data_ingest
from telco_churn.data_prep import DataPreprocessor
from telco_churn.utils import feature_store_utils


class FeatureTableCreator(Job):

    def setup(self):

        # TODO: handle existing feature table more gracefully.
        #  Currently to start afresh involves deleting feature table via UI

        database_name = self.conf['feature_store_params']['database_name']
        table_name = self.conf['feature_store_params']['table_name']

        self.logger.info(f'Creating database {database_name} if not exists')
        self.spark.sql(f'create database if not exists {database_name};')
        self.spark.sql(f'use {database_name};')

        self.spark.sql(f'drop table if exists {table_name};')

    def run_data_ingest(self) -> pyspark.sql.dataframe.DataFrame:
        """
        Returns
        -------
        """
        return data_ingest.spark_load_table(table=self.conf['data_ingest_params']['input_table'])

    def run_data_prep(self, input_df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        """
        Parameters
        ----------
        input_df
        Returns
        -------
        """
        data_preprocessor = DataPreprocessor(cat_cols=self.conf['data_prep_params']['cat_cols'],
                                             label_col=self.conf['data_prep_params']['label_col'],
                                             drop_missing=self.conf['data_prep_params']['drop_missing'])
        preproc_df = data_preprocessor.run(input_df)

        return preproc_df

    def launch(self):
        self.logger.info("Launching FeatureTableCreator job")
        self.setup()

        # TODO - insert check to see if feature table already exists
        # TODO - delete feature table if already exisits?

        self.logger.info('=======Data Ingest=======')
        input_df = self.run_data_ingest()

        self.logger.info('=======Data Prep=======')
        preproc_df = self.run_data_prep(input_df)

        self.logger.info('=======Create Feature Table=======')
        database_name = self.conf['feature_store_params']['database_name']
        table_name = self.conf['feature_store_params']['table_name']
        feature_table_name = f'{database_name}.{table_name}'

        feature_store_utils.create_and_write_feature_table(preproc_df,
                                                           feature_table_name,
                                                           primary_keys=self.conf['feature_store_params']['primary_keys'],
                                                           description=self.conf['feature_store_params']['description'])

        self.logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = FeatureTableCreator()
    job.launch()
