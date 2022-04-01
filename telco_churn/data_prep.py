from dataclasses import dataclass

import pyspark.pandas as ps
from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from databricks.feature_store import feature_table

from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class DataPreprocessor:
    """
    Data preprocessing class

    Attributes:
        cat_cols (list): List of categorical columns
        label_col (str): Name of original label column in input data
        drop_missing (bool): Flag to indicate whether or not to drop missing values
    """
    cat_cols: list
    label_col: str = 'churnString'
    drop_missing: bool = True

    def pyspark_pandas_ohe(self, psdf: ps.DataFrame) -> ps.DataFrame:
        """
        Take a pyspark.pandas DataFrame and convert a list of categorical variables (columns) into dummy/indicator
        variables, also known as one hot encoding.

        Parameters
        ----------
        psdf : ps.DataFrame
            pyspark.pandas DataFrame

        Returns
        -------
        ps.DataFrame
        """
        return ps.get_dummies(psdf, columns=self.cat_cols, dtype='int64')

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
        Method to chain
        Parameters
        ----------
        df

        Returns
        -------

        """

        _logger.info('Running Data Preprocessing steps...')

        # Convert Spark DataFrame to koalas
        psdf = df.to_pandas_on_spark()

        # OHE
        _logger.info('Applying one-hot-encoding')
        ohe_psdf = self.pyspark_pandas_ohe(psdf)

        # Convert label to int and rename column
        _logger.info(f'Processing label: {self.label_col}')
        ohe_psdf = self.process_label(ohe_psdf, rename_to='churn')

        # Clean up column names
        _logger.info(f'Renaming columns')
        ohe_psdf = self.process_col_names(ohe_psdf)

        # Drop missing values
        if self.drop_missing:
            _logger.info(f'Dropping missing values')
            ohe_psdf = self.drop_missing_values(ohe_psdf)

        return ohe_psdf
