import pyspark.pandas as ps
from pyspark.sql.dataframe import DataFrame as SparkDataFrame

from databricks.feature_store import feature_table


def pyspark_pandas_ohe(psdf: ps.DataFrame, columns: list) -> ps.DataFrame:
    """
    Take a pyspark.pandas DataFrame and convert a list of categorical variables (columns) into dummy/indicator
    variables, also known as one hot encoding.

    Parameters
    ----------
    psdf : ps.DataFrame
        pyspark.pandas DataFrame
    columns : list
        List of columns to one-hot encode

    Returns
    -------
    ps.DataFrame
    """
    ohe_psdf = ps.get_dummies(psdf, columns=columns, dtype='int64')

    return ohe_psdf


def process_label(psdf: ps.DataFrame, label_col: str, rename_to: str = 'churn') -> ps.DataFrame:
    """
    Convert label to int and rename column

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
    psdf[label_col] = psdf[label_col].map({'Yes': 1, 'No': 0})
    psdf = psdf.astype({label_col: 'int32'})
    psdf = psdf.rename(columns={label_col: rename_to})

    return psdf


def process_col_names(psdf: ps.DataFrame) -> ps.DataFrame:
    """
    Strip parentheses and spaces from existing column names, replacing spaces with '_'

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


def compute_churn_features(df: SparkDataFrame):
    # Convert Spark DataFrame to koalas
    psdf = df.to_pandas_on_spark()

    cat_cols = ['gender', 'partner', 'dependents',
                'phoneService', 'multipleLines', 'internetService',
                'onlineSecurity', 'onlineBackup', 'deviceProtection',
                'techSupport', 'streamingTV', 'streamingMovies',
                'contract', 'paperlessBilling', 'paymentMethod']
    label_col = 'churnString'
    drop_missing = True

    # OHE
    ohe_psdf = pyspark_pandas_ohe(psdf, columns=cat_cols)

    # Convert label to int and rename column
    ohe_psdf = process_label(ohe_psdf, label_col=label_col, rename_to='churn')

    # Clean up column names
    ohe_psdf = process_col_names(ohe_psdf)

    # Drop missing values
    if drop_missing:
        ohe_psdf = drop_missing_values(ohe_psdf)

    return ohe_psdf
