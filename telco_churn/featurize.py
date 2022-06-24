from dataclasses import dataclass

import pyspark
import pyspark.pandas as ps

from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class FeaturizerConfig:
    """
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


class Featurizer:
    """
    Class containing featurization logic to apply to input Spark DataFrame
    """
    def __init__(self, cfg: FeaturizerConfig):
        self.cfg = cfg 

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
        psdf[self.cfg.label_col] = psdf[self.cfg.label_col].map({'Yes': 1, 'No': 0})
        psdf = psdf.astype({self.cfg.label_col: 'int32'})
        psdf = psdf.rename(columns={self.cfg.label_col: rename_to})

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

    def run(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Run all data preprocessing steps. Consists of the following:
    
            1. Convert PySpark DataFrame to pandas_on_spark DataFrame 
            2. Process the label column - converting to int and renaming col to 'churn'
            3. Apply OHE if specified in the config
            4. Drop any missing values if specified in the config
            5. Return resulting preprocessed dataset as a PySpark DataFrame
            
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input PySpark DataFrame to preprocess

        Returns
        -------
        pyspark.sql.DataFrame
            Preprocessed dataset of features and label column
        """
        _logger.info('Running Data Preprocessing steps...')

        # Convert Spark DataFrame to pandas on Spark DataFrame
        psdf = df.pandas_api()

        # Convert label to int and rename column
        _logger.info(f'Processing label: {self.cfg.label_col}')
        psdf = self.process_label(psdf, rename_to='churn')

        # OHE
        if self.cfg.ohe:
            _logger.info('Applying one-hot-encoding')
            if self.cfg.cat_cols is None:
                raise RuntimeError('cat_cols must be provided if ohe=True')
            psdf = self.pyspark_pandas_ohe(psdf, self.cfg.cat_cols)

            # Clean up column names resulting from OHE
            _logger.info(f'Renaming columns')
            psdf = self.process_col_names(psdf)

        # Drop missing values
        if self.cfg.drop_missing:
            _logger.info(f'Dropping missing values')
            psdf = self.drop_missing_values(psdf)

        # Return as Spark DataFrame
        preproc_df = psdf.to_spark()

        return preproc_df
