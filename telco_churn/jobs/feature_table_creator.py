from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from telco_churn.common import Job
from telco_churn.data_ingest import spark_load_table
from telco_churn.data_prep import DataPreprocessor, create_and_write_feature_table


class FeatureTableCreator(Job):

    def setup(self):

        # TODO: handle existing feature table more gracefully. Currently to start afresh involves deleting feature table via UI

        database_name = self.conf['feature_store_params']['database_name']
        table_name = self.conf['feature_store_params']['table_name']

        self.logger.info(f'Creating database {database_name} if not exists')
        self.spark.sql(f'create database if not exists {database_name};')
        self.spark.sql(f'use {database_name};')

        self.spark.sql(f'drop table if exists {table_name};')

    def run_data_ingest(self):
        """

        Returns
        -------

        """
        return spark_load_table(table=self.conf['data_ingest_params']['input_table'])

    def run_data_prep(self, input_df: SparkDataFrame) -> SparkDataFrame:
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
        preproc_psdf = data_preprocessor.run(input_df)
        preproc_df = preproc_psdf.to_spark()

        return preproc_df

    def launch(self):
        self.logger.info("Launching FeatureTableCreator job")
        self.setup()

        # TODO - insert check to see if feature table already exists

        self.logger.info('=======Data Ingest=======')
        input_df = self.run_data_ingest()

        self.logger.info('=======Data Prep=======')
        preproc_df = self.run_data_prep(input_df)

        self.logger.info('=======Create Feature Table=======')
        database_name = self.conf['feature_store_params']['database_name']
        table_name = self.conf['feature_store_params']['table_name']
        feature_table_name = f'{database_name}.{table_name}'

        create_and_write_feature_table(preproc_df,
                                       feature_table_name,
                                       keys=self.conf['feature_store_params']['keys'],
                                       description=self.conf['feature_store_params']['description'],
                                       mode='overwrite')

        self.logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = FeatureTableCreator()
    job.launch()
