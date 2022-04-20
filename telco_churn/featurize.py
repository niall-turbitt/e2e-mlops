from dataclasses import dataclass

import pyspark
import pyspark.pandas as ps

from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class Featurizer:
    """
    Class containing featurization logic to apply to input Spark DataFrame

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
    def pyspark_pandas_ohe(psdf: ps.DataFrame, cat_cols: list) -> pyspark.pandas.DataFrame:
        """
        Take a pyspark.pandas DataFrame and convert a list of categorical variables (columns) into dummy/indicator
        variables, also known as one hot encoding.
        Parameters
        ----------
        psdf : pyspark.pandas.DataFrame
            pyspark.pandas DataFrame to OHE
        cat_cols : list
            List of categorical features
        Returns
        -------
        pyspark.pandas.DataFrame
        """
        return ps.get_dummies(psdf, columns=cat_cols, dtype='int64')

    def process_label(self, psdf: pyspark.pandas.DataFrame, rename_to: str = 'churn') -> pyspark.pandas.DataFrame:
        """
        Convert label to int and rename label column
        TODO: add test
        Parameters
        ----------
        psdf : pyspark.pandas.DataFrame
            pyspark.pandas DataFrame
        rename_to : str
            Name of new label column name
        Returns
        -------
        pyspark.pandas.DataFrame
        """
        psdf[self.label_col] = psdf[self.label_col].map({'Yes': 1, 'No': 0})
        psdf = psdf.astype({self.label_col: 'int32'})
        psdf = psdf.rename(columns={self.label_col: rename_to})

        return psdf

    @staticmethod
    def process_col_names(psdf: pyspark.pandas.DataFrame) -> pyspark.pandas.DataFrame:
        """
        Strip parentheses and spaces from existing column names, replacing spaces with '_'
        TODO: add test
        Parameters
        ----------
        psdf : pyspark.pandas.DataFrame
            pyspark.pandas DataFrame
        Returns
        -------
        pyspark.pandas.DataFrame
        """
        cols = psdf.columns.to_list()
        new_col_names = [col.replace(' ', '').replace('(', '_').replace(')', '') for col in cols]

        # Update column names to new column names
        psdf.columns = new_col_names

        return psdf

    @staticmethod
    def drop_missing_values(psdf: pyspark.pandas.DataFrame) -> pyspark.pandas.DataFrame:
        """
        Remove missing values
        Parameters
        ----------
        psdf
        Returns
        -------
        pyspark.pandas.DataFrame
        """
        return psdf.dropna()

    def run(self, df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
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

        # Return as Spark DataFrame
        preproc_df = psdf.to_spark()

        return preproc_df
