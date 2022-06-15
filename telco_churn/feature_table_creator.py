from dataclasses import dataclass

import pyspark.sql.dataframe

from telco_churn import featurize
from telco_churn.utils import feature_store_utils
from telco_churn.utils.get_spark import spark
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class FeatureTableCreator:
    """
    Class to execute a pipeline to create a Feature Store table, and separate labels table

    Attributes:
        data_ingest_params (dict): Dictionary containing input_table key. The value for input_table should be the name
            of the table to use as input for creating features
        data_prep_params (dict): Dictionary containing label_col, ohe, cat_cols (optional) and drop_missing keys.
            label_col: name of column from input_table to use as the label column
            ohe: boolean to indicate whether or not to one hot encode categorical columns
            cat_cols: only required if ohe=True. Names of categorical columns to one hot encode
            drop_missing: boolean to indicate whether or not to drop rows with missing values
        feature_store_params (dict): Dictionary containing database_name, table_name, primary_keys and description keys.
            database_name: name of database to use for creating the feature table
            table_name: name of feature table
            primary_keys: string or list of columns to use as the primary key(s). Use single column (customerID) as the
                primary key for the telco churn example.
            description: string containing description to attribute to the feature table in the Feature Store.
        labels_table_params (dict): Dictionary containing database_name, table_name, label_col and dbfs_path
            database_name: name of database to use for creating the labels table
            table_name: name of labels table
            label_col: name of column to use as the label column (in telco churn example we rename this column to 'churn')
            dbfs_path: DBFS path to use for the labels table (saving as a Delta table)
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
        return spark.table(self.data_ingest_params['input_table'])

    def run_data_prep(self, input_df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        """
        Run data preparation step, using Featurizer to run featurization logic to create features from the input
        DataFrame.

        Parameters
        ----------
        input_df : pyspark.sql.dataframe.DataFrame
            Input Spark DataFrame

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            Processed Spark DataFrame containing features
        """
        featurizer = featurize.Featurizer(**self.data_prep_params)
        proc_df = featurizer.run(input_df)

        return proc_df

    def run_feature_table_create(self, df: pyspark.sql.dataframe.DataFrame,
                                 database_name: str, table_name: str) -> None:
        """
        Method to create feature table in Databricks Feature Store. When run, this method will create from scratch the
        feature table. As such, we first create (if it doesn't exist) the database specified, and drop the table if it
        already exists.

        The feature table is created from the Spark DataFrame provided, dropping the label column if it exists in the
        DataFrame. The label column cannot be present in the feature table when later constructing a feature store
        training set from the feature table. The feature table will be created using the primary keys and description
        proivided via feature_store_params.

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame from which to create the feature table.
        database_name :  str
            Name of database to use for creating the feature table
        table_name :  str
            Name of feature table
        """
        # Create database if not exists, drop table if it already exists
        self.setup(database_name=database_name, table_name=table_name)

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
        Method to create labels table. This table will consist of the columns primary_key, label_col

        Create table using params specified in labels_table_params. Will create Delta table at dbfs_path, and further
        create a table using this Delta location.

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            Spark DataFrame containing primary keys column and label column
        """
        if isinstance(self.feature_store_params['primary_keys'], str):
            labels_table_cols = [self.feature_store_params['primary_keys'],
                                 self.labels_table_params['label_col']]
        elif isinstance(self.feature_store_params['primary_keys'], list):
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
        _logger.info('==========Data Ingest==========')
        input_df = self.run_data_ingest()

        _logger.info('==========Data Prep==========')
        proc_df = self.run_data_prep(input_df)

        _logger.info('==========Create Feature Table==========')
        fs_database_name = self.feature_store_params['database_name']
        fs_table_name = self.feature_store_params['table_name']
        self.run_feature_table_create(proc_df, database_name=fs_database_name, table_name=fs_table_name)

        _logger.info('==========Create Labels Table==========')
        self.run_labels_table_create(proc_df)
