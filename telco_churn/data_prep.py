from dataclasses import dataclass
from typing import Union

import pyspark.pandas as ps
from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import entities as fs_entities

from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class DataPreprocessor:
    """
    Data preprocessing class

    Attributes:
        label_col (str): Name of original label column in input data
        ohe (bool): Flag to indicate whether or not to one hot encode categorical columns
        cat_cols (list): List of categorical columns. Only required if ohe=True
        drop_missing (bool): Flag to indicate whether or not to drop missing values
    """
    label_col: str = 'churnString'
    ohe: bool = False
    cat_cols: list = None
    drop_missing: bool = True

    @staticmethod
    def pyspark_pandas_ohe(psdf: ps.DataFrame, cat_cols: list) -> ps.DataFrame:
        """
        Take a pyspark.pandas DataFrame and convert a list of categorical variables (columns) into dummy/indicator
        variables, also known as one hot encoding.

        Parameters
        ----------
        psdf : ps.DataFrame
            pyspark.pandas DataFrame
        cat_cols : list
            List of categorical features

        Returns
        -------
        ps.DataFrame
        """
        return ps.get_dummies(psdf, columns=cat_cols, dtype='int64')

    def process_label(self, psdf: ps.DataFrame, rename_to: str = 'churn') -> ps.DataFrame:
        """
        Convert label to int and rename label column

        TODO: add test

        Parameters
        ----------
        psdf : ps.DataFrame
            pyspark.pandas DataFrame
        rename_to : str
            Name of new label column name

        Returns
        -------
        ps.DataFrame
        """
        psdf[self.label_col] = psdf[self.label_col].map({'Yes': 1, 'No': 0})
        psdf = psdf.astype({self.label_col: 'int32'})
        psdf = psdf.rename(columns={self.label_col: rename_to})

        return psdf

    @staticmethod
    def process_col_names(psdf: ps.DataFrame) -> ps.DataFrame:
        """
        Strip parentheses and spaces from existing column names, replacing spaces with '_'

        TODO: add test

        Parameters
        ----------
        psdf : ps.DataFrame
            pyspark.pandas DataFrame

        Returns
        -------
        ps.DataFrame
        """
        cols = psdf.columns.to_list()
        new_col_names = [col.replace(' ', '').replace('(', '_').replace(')', '') for col in cols]

        # Update column names to new column names
        psdf.columns = new_col_names

        return psdf

    @staticmethod
    def drop_missing_values(psdf: ps.DataFrame) -> ps.DataFrame:
        """
        Remove missing values

        Parameters
        ----------
        psdf

        Returns
        -------
        ps.DataFrame
        """
        return psdf.dropna()

    def run(self, df: SparkDataFrame) -> ps.DataFrame:
        """
        Parameters
        ----------
        df

        Returns
        -------

        """
        _logger.info('Running Data Preprocessing steps...')

        # Convert Spark DataFrame to koalas
        psdf = df.to_pandas_on_spark()

        # Convert label to int and rename column
        _logger.info(f'Processing label: {self.label_col}')
        psdf = self.process_label(psdf, rename_to='churn')

        # OHE
        if self.ohe:
            _logger.info('Applying one-hot-encoding')
            if self.cat_cols is None:
                raise RuntimeError('cat_cols must be provided if ohe=True')
            psdf = self.pyspark_pandas_ohe(psdf, self.cat_cols)

            # Clean up column names resulting from OHE
            _logger.info(f'Renaming columns')
            psdf = self.process_col_names(psdf)

        # Drop missing values
        if self.drop_missing:
            _logger.info(f'Dropping missing values')
            psdf = self.drop_missing_values(psdf)

        return psdf


def create_and_write_feature_table(df: SparkDataFrame,
                                   feature_table_name: str,
                                   keys: Union[str, list],
                                   description: str,
                                   mode: str = 'overwrite') -> fs_entities.feature_table.FeatureTable:
    """

    Parameters
    ----------
    df
    feature_table_name
    keys
    description
    mode

    Returns
    -------
    databricks.feature_store.entities.feature_table.FeatureTable
    """
    fs = FeatureStoreClient()

    feature_table = fs.create_table(
        name=feature_table_name,
        primary_keys=keys,
        schema=df.schema,
        description=description
    )

    fs.write_table(df=df, name=feature_table_name, mode=mode)

    return feature_table
