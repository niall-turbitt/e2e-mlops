from typing import Union, List

import pyspark

import databricks
from databricks.feature_store import FeatureStoreClient


def create_and_write_feature_table(df: pyspark.sql.DataFrame,
                                   feature_table_name: str,
                                   primary_keys: Union[str, List[str]],
                                   description: str) -> databricks.feature_store.entities.feature_table.FeatureTable:
    """
    Create and return a feature table with the given name and primary keys, writing the provided Spark DataFrame to the
    feature table

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Data to create this feature table
    feature_table_name : str
        A feature table name of the form <database_name>.<table_name>, for example dev.user_features.
    primary_keys : Union[str, List[str]]
        The feature table’s primary keys. If multiple columns are required, specify a list of column names, for example
        ['customer_id', 'region'].
    description : str
        Description of the feature table.
    Returns
    -------
    databricks.feature_store.entities.feature_table.FeatureTable
    """
    fs = FeatureStoreClient()

    feature_table = fs.create_table(
        name=feature_table_name,
        primary_keys=primary_keys,
        schema=df.schema,
        description=description
    )

    fs.write_table(df=df, name=feature_table_name, mode='overwrite')

    return feature_table
