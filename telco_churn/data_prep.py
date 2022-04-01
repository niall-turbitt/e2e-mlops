from dataclasses import dataclass

import pyspark.pandas as ps
from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from databricks.feature_store import feature_table


@dataclass
class DataPreprocessor:
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
        Convert label to int and rename column

        TODO: add test

        Parameters
        ----------
        psdf : ps.DataFrame
            pyspark.pandas DataFrame
        label_col : str
            Name of the original label column
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

    def run(self, df: SparkDataFrame):
        # Convert Spark DataFrame to koalas
        psdf = df.to_pandas_on_spark()

        # OHE
        ohe_psdf = self.pyspark_pandas_ohe(psdf)

        # Convert label to int and rename column
        ohe_psdf = self.process_label(ohe_psdf, rename_to='churn')

        # Clean up column names
        ohe_psdf = self.process_col_names(ohe_psdf)

        # Drop missing values
        if self.drop_missing:
            ohe_psdf = self.drop_missing_values(ohe_psdf)

        return ohe_psdf
